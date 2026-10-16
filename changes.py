"""What changed in the room? Compare two phone videos of the same space in 3D.

    python changes.py before.mp4 after.mp4 --out results/my_room

Steps: pick sharp frames from each video -> one joint Depth Anything 3 reconstruction
(both videos share a world frame because the static room ties them together) -> metric
scale from DA3-Metric -> visibility test per 3D point -> cluster changed points -> CLIP
labels and removed/added matching -> HTML viewer + changes.json + report.md.
"""
import argparse, base64, json, os, time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent


# ---------------------------------------------------------------- frames
def sharp_frames(path, n, step=3):
    """n frames spread over the video, the sharpest (Laplacian variance) in each window.
    A folder of images also works: n evenly spaced images (all of them if fewer)."""
    if os.path.isdir(path):
        files = sorted(f for f in Path(path).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int))]
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > 0, f"cannot read {path}"
    edges = np.linspace(0, total, n + 1)
    best = [(-1.0, None)] * n
    for i in range(total):
        if not cap.grab():
            break
        if i % step:
            continue
        ok, f = cap.retrieve()
        if not ok:
            continue
        w = min(int(np.searchsorted(edges, i, side="right")) - 1, n - 1)
        g = cv2.cvtColor(cv2.resize(f, (320, round(320 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
        s = cv2.Laplacian(g, cv2.CV_64F).var()
        if s > best[w][0]:
            best[w] = (s, f)
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in best if f is not None]


# ---------------------------------------------------------------- geometry
def reconstruct(frames, model_name, res=504):
    from depth_anything_3.api import DepthAnything3

    m = DepthAnything3.from_pretrained(model_name).to(DEV).eval()
    p = m.inference(frames, process_res=res)
    del m
    torch.cuda.empty_cache()
    E = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))
    E[:, :3, :] = p.extrinsics[:, :3, :]  # world -> camera (OpenCV axes)
    return dict(depth=p.depth.astype(np.float32), conf=p.conf.astype(np.float32), w2c=E,
                K=p.intrinsics.astype(np.float32), rgb=p.processed_images)


def metric_scale(frames, rec):
    """DA3 any-view depth is up to scale; DA3-Metric gives metres (depth = f * out / 300)."""
    from depth_anything_3.api import DepthAnything3

    m = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(DEV).eval()
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    ratios = []
    for i in idx:
        out = m.inference([frames[i]]).depth[0]
        H, W = rec["depth"][i].shape
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
        K = rec["K"][i]
        metric = out * (K[0, 0] + K[1, 1]) / 2 / 300.0
        good = rec["conf"][i] > np.percentile(rec["conf"][i], 50)
        ratios.append(np.median(metric[good] / rec["depth"][i][good]))
    del m
    torch.cuda.empty_cache()
    return float(np.median(ratios)), [float(r) for r in ratios]


@torch.no_grad()
def dino_maps(rgb, dim=128, name="facebook/dinov2-base"):
    """DINOv2 patch features (14 px) for every processed frame, projected to `dim`, unit length.
    Uncentred projection: centring removes what all views of a surface share."""
    from transformers import AutoModel

    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    N, H, W, _ = rgb.shape
    h, w = H // 14, W // 14
    feats = []
    for i in range(N):
        x = torch.from_numpy(rgb[i : i + 1, : h * 14, : w * 14]).to(DEV).permute(0, 3, 1, 2).float() / 255
        out = m(pixel_values=((x - mean) / std).half(), interpolate_pos_encoding=True).last_hidden_state
        feats.append(out[0, -h * w :].float())
    del m
    torch.cuda.empty_cache()
    F = torch.cat(feats)
    _, _, V = torch.pca_lowrank(F, q=dim, center=False)
    F = torch.nn.functional.normalize(F @ V, dim=-1)
    return F.view(N, h, w, dim).half().cpu().numpy()


def feat_at(fmap, v, u):
    h, w = fmap.shape[:2]
    return fmap[np.minimum(v // 14, h - 1), np.minimum(u // 14, w - 1)]


def backproject(rec, i, mask):
    v, u = np.nonzero(mask)
    z = rec["depth"][i][v, u]
    K = rec["K"][i]
    Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)])
    Xw = (np.linalg.inv(rec["w2c"][i]) @ Xc)[:3].T
    return Xw.astype(np.float32), rec["rgb"][i][v, u]


def source_points(rec, frames, cfg):
    """Confident, non-edge pixels of the given frames as world points."""
    P, C, F, Fe = [], [], [], []
    for i in frames:
        d, c = rec["depth"][i], rec["conf"][i]
        k = np.ones((3, 3), np.uint8)
        jump = (cv2.dilate(d, k) - cv2.erode(d, k)) / d  # flying pixels at depth edges
        m = (c > cfg.conf_thr) & (jump < 0.05)
        sub = np.zeros_like(m)
        sub[:: cfg.stride, :: cfg.stride] = True
        X, rgb = backproject(rec, i, m & sub)
        v, u = np.nonzero(m & sub)
        P.append(X), C.append(rgb), F.append(np.full(len(X), i, np.int16)), Fe.append(feat_at(rec["dino"][i], v, u))
    return np.concatenate(P), np.concatenate(C), np.concatenate(F), np.concatenate(Fe)


def visibility_votes(X, Fe, rec, frames, cfg):
    """For each world point, count target frames that see *through* it (free) or *at* it (same).

    free  : the target camera's depth along that ray is clearly farther -> the point is gone
    same  : the target depth matches -> the point is still there
    neither (occluded, out of view, low confidence) gives no evidence.
    diff  : 'same' geometry but the DINOv2 feature there disagrees -> a flat object changed
            (a moved book barely changes depth, but it changes what the surface looks like)
    """
    diff = np.zeros(len(X), np.int16)
    per_frame = []
    free = np.zeros(len(X), np.int16)
    same = np.zeros(len(X), np.int16)
    k = np.ones((5, 5), np.uint8)
    Xh = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    for j in frames:
        D, Cf, K = rec["depth"][j], rec["conf"][j], rec["K"][j]
        H, W = D.shape
        dmin, dmax = cv2.erode(D, k), cv2.dilate(D, k)
        Xc = Xh @ rec["w2c"][j].T
        z = Xc[:, 2]
        ok = z > 0.05
        u = np.full(len(X), -1, np.int64)
        v = np.full(len(X), -1, np.int64)
        u[ok] = np.round(Xc[ok, 0] / z[ok] * K[0, 0] + K[0, 2])
        v[ok] = np.round(Xc[ok, 1] / z[ok] * K[1, 1] + K[1, 2])
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv, zz = u[ok], v[ok], z[ok]
        lo, hi = dmin[vv, uu], dmax[vv, uu]
        margin = lambda d: cfg.tau_abs + cfg.tau_rel * d
        good = Cf[vv, uu] > cfg.conf_thr
        f = good & (zz < lo - margin(lo))
        s = (zz >= lo - margin(lo)) & (zz <= hi + margin(hi))
        cos = np.einsum("nd,nd->n", Fe[ok][s].astype(np.float32), feat_at(rec["dino"][j], vv[s], uu[s]).astype(np.float32))
        free[ok] += f
        same[ok] += s
        idx = np.nonzero(ok)[0][s]
        per_frame.append((idx[good[s]], cos[good[s]]))
    # self-calibrating: "looks different" = among the lowest app_pct % similarities in this scene
    if per_frame and cfg.app_pct > 0:
        thr = np.percentile(np.concatenate([c for _, c in per_frame]), cfg.app_pct)
        for idx, cos in per_frame:
            diff[idx[cos < thr]] += 1
    return free, same, diff


def classify(free, same, diff, cfg):
    """0 unobserved, 1 unchanged, 2 changed, 3 weakly changed (kept only if touching a 2, like Canny)."""
    strong = (free >= cfg.min_votes) & (free >= cfg.free_ratio * (free + same))
    strong |= (diff >= cfg.min_votes) & (diff >= cfg.free_ratio * same)
    weak = ((free >= 1) & (free >= 0.5 * (free + same))) | ((diff >= 1) & (diff >= 0.5 * same))
    return np.where(strong, 2, np.where(weak, 3, np.where(same > 0, 1, 0)))


def floor_frame(X, w2c, iters=500, tol=0.02, seed=0):
    """RANSAC the floor among unchanged points. Returns T (world -> floor frame): x right,
    y down (floor at y=0), z forward, origin under the first camera. Identity if no floor."""
    rng = np.random.default_rng(seed)
    P = X[rng.choice(len(X), min(len(X), 100_000), replace=False)]
    c2w = np.linalg.inv(w2c)
    # where "up" probably is: the cameras' -y axis when filming a room, or back along the
    # viewing direction when looking down at a table
    for up, cone in [(-c2w[:, :3, 1].mean(0), 30), (-c2w[:, :3, 2].mean(0), 45)]:
        up = up / np.linalg.norm(up)
        best = (0, None)
        for _ in range(iters):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            n = n if n @ up > 0 else -n
            if n @ up < np.cos(np.radians(cone)):
                continue
            h = (P - a) @ n
            below = np.mean(h < -3 * tol)
            inl = np.sum(np.abs(h) < tol)
            if below < 0.03 and inl > best[0]:
                best = (inl, (n, a))
        if best[1] is not None and best[0] >= 0.05 * len(P):
            break
    else:
        return np.eye(4, dtype=np.float32), None
    n, a = best[1]
    c0 = np.linalg.inv(w2c[0])
    x = c0[:3, 0] - (c0[:3, 0] @ n) * n
    x /= np.linalg.norm(x)
    y = -n
    z = np.cross(x, y)
    R = np.stack([x, y, z])
    o = c0[:3, 3] - ((c0[:3, 3] - a) @ n) * n
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, -R @ o
    return T, float(best[0] / len(P))


def cluster(X, F, strong, cfg):
    """Connected components of changed points on a voxel grid; a component counts if it holds
    enough strongly changed voxels. Returns a list of index arrays."""
    if len(X) == 0:
        return []
    vox = np.floor(X / cfg.voxel).astype(np.int64)
    keys, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.ravel()
    centers = (keys + 0.5) * cfg.voxel
    pairs = cKDTree(centers).query_pairs(cfg.voxel * 2.01, output_type="ndarray")
    from scipy.sparse import coo_matrix

    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(keys),) * 2)
    _, lab = connected_components(G, directed=False)
    plab = lab[inv]
    out = []
    for l in np.unique(lab):
        idx = np.nonzero(plab == l)[0]
        st = idx[strong[idx]]
        nvox = len(np.unique(inv[st]))
        ext = X[idx].max(0) - X[idx].min(0)
        if nvox >= cfg.min_voxels and len(np.unique(F[st])) >= cfg.min_frames and ext.max() >= cfg.min_extent:
            out.append(idx)
    return sorted(out, key=len, reverse=True)[: cfg.max_objects]


# ---------------------------------------------------------------- views of an object
def project(rec, i, X):
    Xc = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ rec["w2c"][i].T
    z = Xc[:, 2]
    K = rec["K"][i]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = Xc[:, 0] / z * K[0, 0] + K[0, 2]
        v = Xc[:, 1] / z * K[1, 1] + K[1, 2]
    H, W = rec["depth"][i].shape
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def best_view(rec, frames, X, visible=True, cfg=None):
    """Frame where the points are most visible (or merely in view), and their pixel bbox."""
    best = (-1, None, None)
    for i in frames:
        u, v, z, ok = project(rec, i, X)
        if visible:
            D = rec["depth"][i]
            ui, vi = u[ok].astype(int), v[ok].astype(int)
            vis = np.abs(z[ok] - D[vi, ui]) < cfg.tau_abs + cfg.tau_rel * D[vi, ui]
            ok[np.nonzero(ok)[0][~vis]] = False
        # prefer many points, then the object being central in the image
        score = ok.sum()
        if score > 0:
            H, W = rec["depth"][i].shape
            score -= 0.2 * ok.sum() * np.hypot(u[ok].mean() / W - 0.5, v[ok].mean() / H - 0.5)
        if score > best[0]:
            best = (score, i, (u[ok], v[ok]) if ok.any() else None)
    return best[1], best[2]


def crop(frames_full, rec, i, uv, pad=0.35, min_px=64):
    img = frames_full[i]
    H, W = rec["depth"][i].shape
    sx, sy = img.shape[1] / W, img.shape[0] / H
    u, v = uv
    u0, u1 = np.percentile(u, [2, 98]) * sx
    v0, v1 = np.percentile(v, [2, 98]) * sy
    cx, cy = (u0 + u1) / 2, (v0 + v1) / 2
    s = max(u1 - u0, v1 - v0, min_px * sx) * (1 + 2 * pad) / 2
    x0, x1 = int(max(cx - s, 0)), int(min(cx + s, img.shape[1]))
    y0, y1 = int(max(cy - s, 0)), int(min(cy + s, img.shape[0]))
    box = (int(u0 - x0), int(v0 - y0), int(u1 - x0), int(v1 - y0))
    return img[y0:y1, x0:x1].copy(), box


def object_only(frames_full, rec, i, uv, pad=0.1):
    """Tight crop with everything outside the object's projected points greyed out, so CLIP
    describes the object rather than its surroundings."""
    img = frames_full[i]
    H, W = rec["depth"][i].shape
    m = np.zeros((H, W), np.uint8)
    m[uv[1].astype(int), uv[0].astype(int)] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.where(m[..., None], img, 128).astype(np.uint8)
    ys, xs = np.nonzero(m)
    p = int(pad * max(np.ptp(xs), np.ptp(ys)))
    return out[max(ys.min() - p, 0) : ys.max() + p, max(xs.min() - p, 0) : xs.max() + p]


def sam_masks(frames, long=960):
    """Segment Anything (automatic masks) on every frame: object outlines, whatever their height."""
    from PIL import Image
    from transformers import pipeline

    gen = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if DEV == "cuda" else -1)
    out = []
    for f in frames:
        s = long / max(f.shape[:2])
        im = Image.fromarray(cv2.resize(f, (round(f.shape[1] * s), round(f.shape[0] * s))))
        out.append([np.asarray(m, bool) for m in gen(im, points_per_batch=64, pred_iou_thresh=0.88)["masks"]])
    del gen
    torch.cuda.empty_cache()
    return out


def object_votes(rec, full, own, other, masks, clip, cfg, views=5):
    """Flat objects barely change the geometry, so test them by appearance, one SAM object at a time:
    look at the object's 3D spot from the other video's cameras (where the geometry says that spot
    is visible), crop the same region, and compare with CLIP. A watch that became bare table looks
    different from every angle. Returns world points of objects that changed, and their frames."""
    H, W = rec["depth"][own[0]].shape
    k5 = np.ones((5, 5), np.uint8)
    jobs = []  # (source crop, [target crops], points, frame)
    for i, ms in zip(own, masks):
        good = rec["conf"][i] > cfg.conf_thr
        for m in ms:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not (cfg.obj_min_area < m.mean() < cfg.obj_max_area):
                continue
            m = cv2.erode(m.astype(np.uint8), k5).astype(bool) & good  # stay off the outline
            if m.sum() < 20:
                continue
            X, _ = backproject(rec, i, m)
            cand = []
            for j in other:
                u, v, z, ok = project(rec, j, X)
                if ok.mean() < 0.9:
                    continue
                D = rec["depth"][j][v[ok].astype(int), u[ok].astype(int)]
                vis = np.abs(z[ok] - D) < cfg.tau_abs + cfg.tau_rel * D  # spot visible, not blocked
                if vis.mean() > 0.6:
                    cand.append((vis.mean(), j, (u[ok], v[ok])))
            if len(cand) < 2:
                continue
            cand = sorted(cand, key=lambda c: -c[0])[:views]
            v_, u_ = np.nonzero(m)
            # shrink at once to CLIP's input size: thousands of full-res 4K crops would fill RAM
            small = lambda c: cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)
            jobs.append((small(object_only(full, rec, i, (u_, v_))), [small(object_only(full, rec, j, uv)) for _, j, uv in cand], X, i))
    if not jobs:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int16), []
    src = clip.image([j[0] for j in jobs])
    tgt = clip.image([c for j in jobs for c in j[1]])
    sims, k = [], 0
    for n, job in enumerate(jobs):
        t = tgt[k : k + len(job[1])]
        k += len(job[1])
        sims.append(float((t @ src[n]).median()))
    sims = np.array(sims)
    keep = [n for n in range(len(jobs)) if sims[n] < cfg.obj_sim]
    X = np.concatenate([jobs[n][2] for n in keep]) if keep else np.zeros((0, 3), np.float32)
    F = np.concatenate([np.full(len(jobs[n][2]), jobs[n][3], np.int16) for n in keep]) if keep else np.zeros(0, np.int16)
    return X[:: cfg.stride], F[:: cfg.stride], sims


# ---------------------------------------------------------------- semantics
VOCAB = """chair, office chair, stool, table, desk, coffee table, sofa, cushion, pillow, blanket, bed,
lamp, floor lamp, plant, potted plant, vase, flowers, box, cardboard box, storage box, bin, trash can,
basket, laundry basket, bag, backpack, handbag, suitcase, shoe, shoes, boots, jacket, coat, towel,
clothes, hat, book, books, notebook, magazine, paper, laptop, computer monitor, keyboard, computer mouse,
phone, tablet, headphones, speaker, television, remote control, game controller, cable, charger,
bottle, water bottle, cup, mug, glass, bowl, plate, kettle, toaster, microwave, pan, pot, fruit,
apple, banana, food, cereal box, jar, can, toy, stuffed animal, ball, football, guitar, umbrella,
fan, heater, clock, picture frame, painting, mirror, poster, candle, tissue box, toilet paper,
bucket, broom, vacuum cleaner, squeeze bottle, sauce bottle, beer bottle, food can, tin can,
snack box, cracker box, small box, packet, container, plastic tub, jar of mints, chewing gum pack, ladder, tool box, drill, hammer, scissors, tape, pen, cone, bicycle,
helmet, skateboard, dumbbell, yoga mat, guitar case, printer, router, monitor stand, shelf,
drawer unit, cabinet, door, curtain, rug, mat, wall, floor, window""".replace("\n", " ")
VOCAB = [w.strip() for w in VOCAB.split(",") if w.strip()]
NOT_OBJECTS = {"wall", "floor", "window", "door", "curtain", "rug", "mat", "shelf", "cabinet"}


def _t(o):  # transformers 5 returns an output object, 4.x a tensor
    return o if torch.is_tensor(o) else o.pooler_output


class Clip:
    def __init__(self, name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor

        self.m = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        self.p = CLIPProcessor.from_pretrained(name)
        with torch.no_grad():
            t = self.p(text=[f"a photo of a {w}" for w in VOCAB], return_tensors="pt", padding=True).to(DEV)
            self.text = torch.nn.functional.normalize(_t(self.m.get_text_features(**t)).float(), dim=-1)

    @torch.no_grad()
    def image(self, imgs, bs=64):
        out = []
        for k in range(0, len(imgs), bs):
            x = self.p(images=imgs[k : k + bs], return_tensors="pt").to(DEV)
            out.append(_t(self.m.get_image_features(pixel_values=x["pixel_values"].half())).float())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)

    def label(self, feat):
        p = (100 * feat @ self.text.T).softmax(-1)
        top = p.topk(3, dim=-1)
        return [[(VOCAB[j], float(s)) for s, j in zip(ss, jj)] for ss, jj in zip(top.values, top.indices)]


@torch.no_grad()
def vlm_names(crops, boxes, name="Qwen/Qwen3-VL-2B-Instruct"):
    """Name each object with a small vision-language model: the crop with a red box drawn
    around the object, answer in a few words. Free text, so no fixed word list."""
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    p = AutoProcessor.from_pretrained(name)
    q = "What is the object inside the red box? Answer with a short name only (1-4 words)."
    names = []
    for im, b in zip(crops, boxes):
        im = im.copy()
        cv2.rectangle(im, b[:2], b[2:], (255, 0, 0), max(2, im.shape[1] // 150))
        pil = Image.fromarray(im)
        pil.thumbnail((448, 448))
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": q}]}]
        x = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(DEV)
        out = m.generate(**x, max_new_tokens=12, do_sample=False)
        names.append(p.decode(out[0, x["input_ids"].shape[1] :], skip_special_tokens=True).strip(" .'\"\n").lower())
    del m
    torch.cuda.empty_cache()
    return names


GENERIC = set("""black white grey gray red green blue yellow orange pink purple brown silver gold
clear dark light plastic metal leather wooden wood glass paper cardboard small large big round square
the a an of and with""".split())


def same_thing(g, n, cfg):
    """With free-text names, both ends of a move must share a meaningful word
    ('mustard bottle' / "french's mustard"); colours and materials don't count."""
    if cfg.namer != "vlm":
        return True
    words = lambda o: {w.strip("'s") for w in o["label"].replace("'s", "").split()} - GENERIC
    return bool(words(g) & words(n))


# ---------------------------------------------------------------- language
def direction(d):
    """Describe a displacement in the floor frame (x right, y down, z forward from the first view)."""
    parts = []
    for val, pos, neg in [(d[0], "right", "left"), (-d[1], "up", "down"), (d[2], "further away", "closer")]:
        if abs(val) >= 0.1 and abs(val) >= 0.3 * np.abs(d).max():
            parts.append(f"{abs(val):.2f} m {pos if val > 0 else neg}")
    return ", ".join(parts) or "slightly"


def where(P, floor=True, surface="floor"):
    c = P.mean(0)
    side = "left" if c[0] < -0.3 else "right" if c[0] > 0.3 else "centre"
    s = f"{c[2]:.1f} m ahead, {side} ({c[0]:+.1f} m)"
    if floor:
        h = -P[:, 1].max()  # y points down, floor at 0
        s += f", on the {surface}" if h < 0.1 else f", {h:.1f} m above the {surface}"
    return s


# ---------------------------------------------------------------- viewer
def to_b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def jpg_b64(img, box=None, color=(255, 255, 255)):
    img = img.copy()
    if box is not None:
        t = max(2, img.shape[1] // 120)
        cv2.rectangle(img, box[:2], box[2:], color, t)
    s = 320 / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def voxel_down(X, C, size, extra=None):
    _, keep = np.unique(np.floor(X / size).astype(np.int64), axis=0, return_index=True)
    return (X[keep], C[keep]) + ((extra[keep],) if extra is not None else ())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before"), ap.add_argument("after")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--frames", type=int, default=20, help="frames per video (40 total fits an 8 GB GPU)")
    ap.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    ap.add_argument("--res", type=int, default=None, help="reconstruction resolution (long side, multiple of 14)")
    ap.add_argument("--scale", type=float, default=None, help="override metric scale")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf-pct", type=float, default=30, help="confidence percentile treated as unreliable")
    ap.add_argument("--tau-abs", type=float, default=None, help="m, depth tolerance (default 2%% of median depth)")
    ap.add_argument("--tau-rel", type=float, default=0.05, help="fraction of depth, depth tolerance")
    ap.add_argument("--namer", choices=["vlm", "clip"], default="vlm", help="name objects with Qwen3-VL-2B or the CLIP word list")
    ap.add_argument("--objects", type=int, default=0, help="1 = also test SAM objects by appearance: finds flat objects, more false positives, ~3x slower")
    ap.add_argument("--obj-sim", type=float, default=0.75, help="CLIP similarity below which an object's spot looks different")
    ap.add_argument("--obj-min-area", type=float, default=0.0005)
    ap.add_argument("--obj-max-area", type=float, default=0.08)
    ap.add_argument("--app-pct", type=float, default=0, help="lowest %% of DINOv2 similarities that count as 'looks different' (0 = off)")
    ap.add_argument("--min-free", type=float, default=2.0, help="mean see-through votes a change needs")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--free-ratio", type=float, default=0.7)
    ap.add_argument("--voxel", type=float, default=None, help="m (default 1.2%% of median depth)")
    ap.add_argument("--min-voxels", type=int, default=25)
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--min-extent", type=float, default=None, help="m (default 3%% of median depth)")
    ap.add_argument("--max-objects", type=int, default=12)
    ap.add_argument("--query", default=None, help="image folder reconstructed with the scene but not used "
                    "for detection (for evaluation); poses/depth saved in scene.npz")
    ap.add_argument("--match-sim", type=float, default=0.70, help="CLIP similarity (and mutual best match) to call it 'moved'")
    cfg = ap.parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    def frames(path, n):  # decoding 4K HEVC is slow, so keep the picked frames next to the output
        cache = out / f".frames_{Path(path).stem}_{n}_{int(os.path.getmtime(path))}.npy"
        if cache.exists():
            return list(np.load(cache))
        fr = sharp_frames(path, n)
        if len({f.shape for f in fr}) == 1:
            np.save(cache, np.stack(fr))
        return fr

    fa, fb = frames(cfg.before, cfg.frames), frames(cfg.after, cfg.frames)
    fq = sharp_frames(cfg.query, 10**6) if cfg.query else []
    if fa[0].shape != fb[0].shape or (fq and fq[0].shape != fa[0].shape):
        # one video portrait, one landscape: the joint reconstruction needs one shape, so take
        # the centre square of every frame
        sq = lambda f: f[(f.shape[0] - min(f.shape[:2])) // 2 :][: min(f.shape[:2])][:, (f.shape[1] - min(f.shape[:2])) // 2 :][:, : min(f.shape[:2])]
        fa, fb, fq = [sq(f) for f in fa], [sq(f) for f in fb], [sq(f) for f in fq]
        log("videos differ in shape -> centre-square crops")
    full = fa + fb
    A, B = list(range(len(fa))), list(range(len(fa), len(full)))
    Q = list(range(len(full), len(full) + len(fq)))
    log(f"frames: {len(fa)} before, {len(fb)} after, {full[0].shape[1]}x{full[0].shape[0]}")

    # DA3 resizes the long side to --res; square frames cost ~1.8x the memory of 16:9 ones
    res = cfg.res or (392 if full[0].shape[0] == full[0].shape[1] else 504)
    rec = reconstruct(full + fq, cfg.model, res)
    log("joint reconstruction done", rec["depth"].shape)
    if cfg.scale is None:
        cfg.scale, ratios = metric_scale(full, rec)
        log(f"metric scale {cfg.scale:.3f} (per-frame {np.round(ratios, 3).tolist()})")
    rec["depth"] *= cfg.scale
    rec["w2c"][:, :3, 3] *= cfg.scale
    cfg.conf_thr = float(np.percentile(rec["conf"], cfg.conf_pct))
    # tolerances follow the scene size: a desk seen from 0.5 m needs finer voxels than a room
    med = float(np.median(rec["depth"][rec["conf"] > cfg.conf_thr]))
    cfg.voxel = cfg.voxel or 0.012 * med
    cfg.tau_abs = cfg.tau_abs or 0.02 * med
    cfg.min_extent = cfg.min_extent or 0.03 * med
    log(f"median depth {med:.2f} m -> voxel {cfg.voxel * 100:.1f} cm, tau {cfg.tau_abs * 100:.1f} cm + {cfg.tau_rel:.0%}")

    # points from each video tested against the other video's cameras
    # optional appearance cue (off by default: on 3DGS-CD it cost more than it gained, see README)
    rec["dino"] = dino_maps(rec["rgb"]) if cfg.app_pct > 0 else np.zeros((len(rec["rgb"]), 1, 1, 1), np.float16)
    XA, CA, FA, EA = source_points(rec, A, cfg)
    XB, CB, FB, EB = source_points(rec, B, cfg)
    vA, vB = visibility_votes(XA, EA, rec, B, cfg), visibility_votes(XB, EB, rec, A, cfg)
    sA, sB = classify(*vA, cfg), classify(*vB, cfg)  # changed in A = gone in B; changed in B = new

    if cfg.objects:  # appearance test per SAM object, added as extra "changed" points
        # crops for this test come from 1280 px copies: 4K crops by the thousand are slow
        lowres = [cv2.resize(f, (round(f.shape[1] * 1280 / max(f.shape[:2])), round(f.shape[0] * 1280 / max(f.shape[:2]))), interpolation=cv2.INTER_AREA) for f in full]
        masks = sam_masks([full[i] for i in A + B])
        clip = Clip()
        add = []
        for own, other, ms in [(A, B, masks[: len(A)]), (B, A, masks[len(A) :])]:
            X, F, sims = object_votes(rec, lowres, own, other, ms, clip, cfg)
            add.append((X, F))
            log(f"object test: {len(sims)} SAM objects checked, {np.sum(np.array(sims) < cfg.obj_sim)} look different "
                f"(CLIP sim percentiles 5/25/50: {np.round(np.percentile(sims, [5, 25, 50]), 2).tolist() if len(sims) else []})")
        del clip
        torch.cuda.empty_cache()
        pseudo = lambda n: (np.full(n, cfg.min_free + 1, np.int16), np.zeros(n, np.int16), np.zeros(n, np.int16))
        (X2, F2), (X3, F3) = add
        XA, FA, sA = np.concatenate([XA, X2]), np.concatenate([FA, F2]), np.concatenate([sA, np.full(len(X2), 2)])
        XB, FB, sB = np.concatenate([XB, X3]), np.concatenate([FB, F3]), np.concatenate([sB, np.full(len(X3), 2)])
        CA = np.concatenate([CA, np.zeros((len(X2), 3), CA.dtype)])
        CB = np.concatenate([CB, np.zeros((len(X3), 3), CB.dtype)])
        vA = tuple(np.concatenate([a, b]) for a, b in zip(vA, pseudo(len(X2))))
        vB = tuple(np.concatenate([a, b]) for a, b in zip(vB, pseudo(len(X3))))
    log(f"before pts {len(XA)}: unchanged {np.mean(sA == 1):.1%} gone {np.mean(sA == 2):.2%} unobserved {np.mean(sA == 0):.1%}")
    log(f"after  pts {len(XB)}: unchanged {np.mean(sB == 1):.1%} new  {np.mean(sB == 2):.2%} unobserved {np.mean(sB == 0):.1%}")

    # put the floor at y=0 so "up", heights and directions mean something
    T, frac = floor_frame(np.concatenate([XA[sA == 1], XB[sB == 1]]), rec["w2c"])
    log(f"floor plane: {'not found, using camera frame' if frac is None else f'{frac:.0%} of points'}")
    XA, XB = XA @ T[:3, :3].T + T[:3, 3], XB @ T[:3, :3].T + T[:3, 3]
    rec["w2c"] = rec["w2c"] @ np.linalg.inv(T)
    cfg.has_floor = frac is not None

    mA, mB = sA >= 2, sB >= 2
    gone = [np.nonzero(mA)[0][c] for c in cluster(XA[mA], FA[mA], sA[mA] == 2, cfg)]
    new = [np.nonzero(mB)[0][c] for c in cluster(XB[mB], FB[mB], sB[mB] == 2, cfg)]
    log(f"clusters: {len(gone)} gone, {len(new)} new")

    # crops: the object where it is, and the same 3D spot in the other video
    objs = []
    for kind, idxs, X, own, other, (fr, sm, _) in [("gone", gone, XA, A, B, vA), ("new", new, XB, B, A, vB)]:
        for idx in idxs:
            P = X[idx]
            ev = dict(free=float(fr[idx].mean()), same=float(sm[idx].mean()), n=len(idx))
            if ev["free"] < cfg.min_free:  # weakly supported: usually depth noise near edges
                continue
            i, uv = best_view(rec, own, P, True, cfg)
            j, uv2 = best_view(rec, other, P, False, cfg)
            o = dict(kind=kind, idx=idx, pts=P, center=P.mean(0), size=P.max(0) - P.min(0), frame=i, ev=ev)
            o["crop"], o["box"] = crop(full, rec, i, uv)
            o["obj"] = object_only(full, rec, i, uv)
            if uv2 is not None:
                o["crop2"], o["box2"] = crop(full, rec, j, uv2)
            objs.append(o)

    changes = []
    if objs:
        clip = Clip()
        feats = clip.image([o["obj"] for o in objs])
        labels = clip.label(feats)
        for o, f, l in zip(objs, feats, labels):
            o["feat"], o["labels"] = f, l
            o["label"] = next((w for w, _ in l if w not in NOT_OBJECTS), l[0][0])
        del clip
        torch.cuda.empty_cache()
        if cfg.namer == "vlm":
            try:
                for o, n in zip(objs, vlm_names([o["crop"] for o in objs], [o["box"] for o in objs])):
                    o["label"] = n or o["label"]
            except Exception as e:  # keep the CLIP names if the VLM can't run
                log(f"VLM naming failed ({type(e).__name__}: {e}); using CLIP labels")

        # moved = a gone object that looks like a new one (Hungarian on CLIP similarity)
        G = [o for o in objs if o["kind"] == "gone"]
        N = [o for o in objs if o["kind"] == "new"]
        pairs = []
        if G and N:
            S = (torch.stack([o["feat"] for o in G]) @ torch.stack([o["feat"] for o in N]).T).cpu().numpy()
            log("CLIP similarity, removed (rows) x added (cols):", np.round(S, 2).tolist())
            for r, c in zip(*linear_sum_assignment(-S)):
                mutual = S[r].argmax() == c and S[:, c].argmax() == r  # each is the other's best match
                if same_thing(G[r], N[c], cfg) and mutual and S[r, c] >= cfg.match_sim and np.linalg.norm(G[r]["center"] - N[c]["center"]) > 8 * cfg.voxel:
                    pairs.append((G[r], N[c], float(S[r, c])))
        used = {id(o) for p in pairs for o in p[:2]}
        for g, n, s in pairs:
            d = n["center"] - g["center"]
            changes.append(dict(type="moved", label=g["label"], before=g, after=n, similarity=s,
                                distance_m=float(np.linalg.norm(d)),
                                text=f"The {g['label']} moved {np.linalg.norm(d):.2f} m ({direction(d)})."))
        for o in objs:
            if id(o) in used:
                continue
            if o["kind"] == "gone":
                changes.append(dict(type="removed", label=o["label"], before=o, after=None,
                                    text=f"The {o['label']} {where(o['pts'], cfg.has_floor, 'table' if cfg.voxel < 0.018 else 'floor')} was removed."))
            else:
                changes.append(dict(type="added", label=o["label"], before=None, after=o,
                                    text=f"A {o['label']} appeared {where(o['pts'], cfg.has_floor, 'table' if cfg.voxel < 0.018 else 'floor')}."))
    for c in changes:
        ev = "  ".join(f"{k}: free {o['ev']['free']:.1f} same {o['ev']['same']:.1f} n {o['ev']['n']}" for k in ("before", "after") if (o := c[k]) is not None)
        log(c["text"], "|", ev)

    write_outputs(out, cfg, rec, full, A, B, XA, CA, sA, XB, CB, sB, changes)
    # raw geometry for evaluation / reuse: query cameras and every change's 3D points
    npz = dict(q_w2c=rec["w2c"][Q], q_K=rec["K"][Q], q_depth=rec["depth"][Q], hw=np.array(rec["depth"].shape[1:]))
    for n, c in enumerate(changes, 1):
        for role in ("before", "after"):
            if c[role] is not None:
                npz[f"{n:02d}_{c['type']}_{role}"] = c[role]["pts"]
    np.savez_compressed(out / "scene.npz", **npz)
    log(f"wrote {out}/viewer.html, changes.json, report.md")


def write_outputs(out, cfg, rec, full, A, B, XA, CA, sA, XB, CB, sB, changes):
    COL = dict(removed=(230, 60, 60), added=(60, 200, 90), moved_before=(240, 150, 40), moved_after=(70, 130, 240))
    (out / "crops").mkdir(exist_ok=True)
    for f in (out / "crops").glob("*.jpg"):  # crops from an earlier run would be misleading
        f.unlink()

    # background: everything not in a change, coloured by the photo; state kept for toggles
    in_change = {("gone", int(i)) for c in changes for k in ("before", "after") if c[k] is not None and c[k]["kind"] == "gone" for i in c[k]["idx"]}
    in_change |= {("new", int(i)) for c in changes for k in ("before", "after") if c[k] is not None and c[k]["kind"] == "new" for i in c[k]["idx"]}
    mA = np.array([("gone", i) not in in_change for i in range(len(XA))])
    mB = np.array([("new", i) not in in_change for i in range(len(XB))])
    bgX = np.concatenate([XA[mA], XB[mB]])
    bgC = np.concatenate([CA[mA], CB[mB]])
    bgS = np.concatenate([sA[mA], sB[mB]]).astype(np.uint8)
    bgX, bgC, bgS = voxel_down(bgX, bgC, cfg.voxel / 2, bgS)
    if len(bgX) > 250_000:
        k = np.random.default_rng(0).choice(len(bgX), 250_000, replace=False)
        bgX, bgC, bgS = bgX[k], bgC[k], bgS[k]

    items, js = [], []
    for n, c in enumerate(changes, 1):
        objs = []
        if c["type"] == "moved":
            objs = [("before", c["before"], COL["moved_before"]), ("after", c["after"], COL["moved_after"])]
        else:
            o = c["before"] or c["after"]
            objs = [("before" if c["before"] else "after", o, COL[c["type"]])]
        entry = dict(id=n, type=c["type"], label=c["label"], text=c["text"], objects=[])
        for role, o, col in objs:
            P, _ = voxel_down(o["pts"], np.zeros((len(o["pts"]), 3), np.uint8), cfg.voxel / 3)
            entry["objects"].append(dict(role=role, pts=to_b64(P.astype(np.float32)), n=len(P),
                                         min=P.min(0).tolist(), max=P.max(0).tolist(), color=col))
        # images: before / after, with the object boxed where it is
        if c["type"] == "moved":
            imgs = [("before", c["before"]["crop"], c["before"]["box"]), ("after", c["after"]["crop"], c["after"]["box"])]
        elif c["type"] == "removed":
            o = c["before"]
            imgs = [("before", o["crop"], o["box"])] + ([("after", o["crop2"], o["box2"])] if "crop2" in o else [])
        else:
            o = c["after"]
            imgs = ([("before", o["crop2"], o["box2"])] if "crop2" in o else []) + [("after", o["crop"], o["box"])]
        entry["images"] = [dict(role=r, jpg=jpg_b64(im, bx)) for r, im, bx in imgs]
        for r, im, bx in imgs:
            cv2.imwrite(str(out / "crops" / f"{n:02d}_{c['type']}_{r}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        js.append(entry)
        rec_json = dict(id=n, type=c["type"], label=c["label"], text=c["text"])
        for role, o, _ in objs:
            rec_json[role] = dict(center_m=np.round(o["center"], 3).tolist(), size_m=np.round(o["size"], 3).tolist(),
                                  label_scores=[[w, round(s, 3)] for w, s in o["labels"]])
        if c["type"] == "moved":
            rec_json["distance_m"] = round(c["distance_m"], 3)
            rec_json["clip_similarity"] = round(c["similarity"], 3)
        items.append(rec_json)

    cams = []
    for group, name in [(A, "before"), (B, "after")]:
        cams.append(dict(name=name, pos=[np.linalg.inv(rec["w2c"][i])[:3, 3].tolist() for i in group]))
    data = dict(bg=dict(pts=to_b64(bgX.astype(np.float32)), col=to_b64(bgC.astype(np.uint8)),
                        state=to_b64(bgS), n=len(bgX)),
                changes=js, cams=cams, unit=cfg.voxel / 0.012,  # viewer sizes scale with the scene
                title=f"{Path(cfg.before).name} vs {Path(cfg.after).name}")
    html = (HERE / "viewer.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (out / "viewer.html").write_text(html)

    summary = dict(before=str(cfg.before), after=str(cfg.after), metric_scale=cfg.scale,
                   frames_per_video=len(A), changes=items,
                   settings={k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str))})
    (out / "changes.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# What changed: {data['title']}", ""]
    lines += [f"{n}. **{c['type']}**: {c['text']}" for n, c in enumerate(changes, 1)] or ["No changes found."]
    lines += ["", "Crops: `crops/NN_type_before|after.jpg`. Open `viewer.html` for the 3D view."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
