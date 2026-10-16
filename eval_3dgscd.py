"""Score changes.py on the 3DGS-CD dataset (Lu et al., RA-L 2025).

    python eval_3dgscd.py data/3dgscd/Desk [data/3dgscd/Bench ...]

Per scene: before = rgb/ (sampled), after = even-indexed rgb_new/ images, query = odd-indexed
rgb_new/ images, which have ground-truth masks of the changed objects in their new place.
Query views are reconstructed with the scene but never used for detection. Our added /
moved-to objects are projected into each query view (visible points only) and compared to
the mask with IoU. The paper describes its masks as move-in + move-out, but the released masks
for these views only outline objects in their new place (checked by overlay), so we score
move-in only; projecting our move-out regions too lands on bare table and drops IoU.
"""
import json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).parent


def predicted_mask(z, q, k):
    H, W = z["hw"]
    E, K, D = z["q_w2c"][k], z["q_K"][k], z["q_depth"][k]
    m = np.zeros((H, W), np.uint8)
    for key in z.files:
        if not key.endswith("_after"):  # objects present in the after scene (move-in)
            continue
        P = z[key]
        Xc = np.c_[P, np.ones(len(P))] @ E.T
        zz = Xc[:, 2]
        ok = zz > 0.05
        u = np.round(Xc[ok, 0] / zz[ok] * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(Xc[ok, 1] / zz[ok] * K[1, 1] + K[1, 2]).astype(int)
        zz = zz[ok]
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, zz = u[inb], v[inb], zz[inb]
        vis = zz <= D[v, u] * 1.03  # in front of what this view sees, i.e. not occluded
        m[v[vis], u[vis]] = 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return ndimage.binary_fill_holes(m).astype(np.uint8)


def run(scene, extra):
    scene = Path(scene)
    out = HERE / "results" / "3dgscd" / scene.name
    split = out / "split"
    new = sorted((scene / "rgb_new").iterdir())
    for name, files in [("after", new[0::2]), ("query", new[1::2])]:
        (split / name).mkdir(parents=True, exist_ok=True)
        for f in files:
            link = split / name / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    subprocess.run([sys.executable, str(HERE / "changes.py"), str(scene / "rgb"), str(split / "after"),
                    "--query", str(split / "query"), "--out", str(out)] + extra, check=True)

    z = np.load(out / "scene.npz")
    rows, tiles = [], []
    for k, f in enumerate(new[1::2]):
        gt = cv2.imread(str(scene / "masks_gt" / f.name), cv2.IMREAD_GRAYSCALE) > 127
        pr = cv2.resize(predicted_mask(z, f, k), gt.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        inter, union = (gt & pr).sum(), (gt | pr).sum()
        rows.append(dict(view=f.name, iou=float(inter / union) if union else 1.0,
                         precision=float(inter / pr.sum()) if pr.sum() else 0.0,
                         recall=float(inter / gt.sum()) if gt.sum() else 1.0))
        img = cv2.imread(str(f))
        over = img.copy()
        over[pr] = (0.5 * over[pr] + [0, 0, 128]).astype(np.uint8)  # prediction in red
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cs, -1, (0, 255, 0), 3)  # ground truth outline in green
        tiles.append(cv2.resize(over, (480, round(480 * img.shape[0] / img.shape[1]))))
    cv2.imwrite(str(out / "eval_views.jpg"), np.hstack(tiles))
    res = dict(scene=scene.name, mean_iou=float(np.mean([r["iou"] for r in rows])), views=rows,
               changes=json.loads((out / "changes.json").read_text())["changes"])
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    print(f"{scene.name}: mean IoU {res['mean_iou']:.3f}  " + " ".join(f"{r['iou']:.2f}" for r in rows))
    return res


if __name__ == "__main__":
    scenes = [a for a in sys.argv[1:] if not a.startswith("--")]
    extra = [a for a in sys.argv[1:] if a.startswith("--")]  # pass-through, e.g. --tau-rel=0.03
    res = [run(s, extra) for s in scenes]
    print("mean over scenes:", round(float(np.mean([r["mean_iou"] for r in res])), 3))
