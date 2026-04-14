"""
StitchWise — Live Disaster Map Viewer
======================================
Pipeline per image:
  1. Metric scale  — estimate GSD (m/px) from EXIF or depth fallback
  2. YOLO detect   — tiled inference (Raphael) → bounding boxes in image coords
  3. SAM segment   — refine each box to a pixel mask (Kane)
  4. Stitch        — warp image onto growing mosaic (Zhaochen)
  5. Overlay       — project masks onto mosaic canvas and composite

GUI controls:
  Drag           — pan
  Scroll wheel   — zoom (anchored at cursor)
  Double-click   — fit whole mosaic to window
  Left-click ×2  — measure ground distance (metres)
  Fit / Clear    — buttons in right panel

Usage:
    python live_view.py --image-dir data/rescuenet_test --ext .jpg
    python live_view.py --image-dir data/rescuenet_test --n-frames 10 --ext .jpg --reuse
    python live_view.py --image-dir data/rescuenet_test --no-seg   # skip detection/SAM
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Class definitions (matches Raphael's final rescuenet_v2 model) ─────────
# 4 classes: water, building-damaged, road-blocked, vehicle
CLASS_NAMES: dict[int, str] = {
    0: "Water / Flooding",
    1: "Bldg Damaged",
    2: "Road Blocked",
    3: "Vehicle",
}
# BGR for OpenCV, RGB for PIL/tkinter
CLASS_BGR: dict[int, tuple] = {
    0: ( 20,  90, 210),   # blue
    1: ( 20,  20, 210),   # red
    2: (  0, 140, 255),   # orange
    3: ( 20, 185,  20),   # green
}
CLASS_RGB: dict[int, tuple] = {k: (v[2], v[1], v[0]) for k, v in CLASS_BGR.items()}

OVERLAY_ALPHA          = 0.45
STITCHING_RESIZE_MAX_DIM = 1600
RENDER_MAX_SIDE        = 4000
RENDER_MAX_AREA        = 16_000_000
DISPLAY_W, DISPLAY_H   = 960, 700    # initial canvas size

# Auto-select GPU if available, fall back to CPU
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else \
         "mps"  if _torch.backends.mps.is_available() else "cpu"


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_idx(name: str) -> int:
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else 10 ** 9


def _tiled_detect(img_bgr: np.ndarray, model,
                  conf: float, iou: float,
                  tile_size: int = 640, overlap: int = 64,
                  device: str = DEVICE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Raphael's tiled-inference logic, inlined for use on per-image BGR arrays.
    Returns (boxes[N,4], scores[N], classes[N]) in image-pixel coords.
    """
    h, w   = img_bgr.shape[:2]
    stride = tile_size - overlap
    tiles  = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size); y1 = max(0, y2 - tile_size)
            tiles.append((img_bgr[y1:y2, x1:x2].copy(), x1, y1))

    all_boxes, all_scores, all_classes = [], [], []
    for tile_img, x_off, y_off in tiles:
        res = model.predict(source=tile_img, conf=conf, iou=iou,
                            device=device, imgsz=tile_size, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        for box in res.boxes:
            tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
            all_boxes.append([tx1+x_off, ty1+y_off, tx2+x_off, ty2+y_off])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return (np.zeros((0,4),np.float32),
                np.zeros(0,np.float32),
                np.zeros(0,np.int32))

    import torch
    from torchvision.ops import batched_nms
    b_t = torch.tensor(all_boxes,   dtype=torch.float32)
    s_t = torch.tensor(all_scores,  dtype=torch.float32)
    c_t = torch.tensor(all_classes, dtype=torch.int64)
    keep = batched_nms(b_t, s_t, c_t, iou)
    return (b_t[keep].numpy(), s_t[keep].numpy(),
            c_t[keep].numpy().astype(np.int32))


# ── Background pipeline worker ─────────────────────────────────────────────

def pipeline_worker(
    image_dir: Path,
    images:    list[Path],
    output_dir: Path,
    yolo_weights: str | None,
    conf: float,
    neighbor_offsets: str,
    ext: str,
    q: queue.Queue,
) -> None:
    try:
        from stitchwise.config   import load_config
        from stitchwise.io_utils import load_image, resolve_image_path, resize_by_max_dim
        from metric_scale        import estimate as estimate_gsd

        # ── 1. Metric scale ───────────────────────────────────────────
        q.put(("status", "Estimating metric scale…"))
        gsds: dict[str, float] = {}
        for p in images:
            try:
                gsds[p.stem] = estimate_gsd(p, use_depth_fallback=False).gsd_m_per_px
            except Exception:
                gsds[p.stem] = 0.05
        mean_gsd = float(np.mean(list(gsds.values())))
        q.put(("status", f"Mean GSD: {mean_gsd*100:.2f} cm/px"))

        # ── 2. Stitching: pair graph + global solve ────────────────────
        pair_dir   = output_dir / "pair_graph"
        global_dir = output_dir / "global_no_ba"
        poses_path = global_dir / "global_poses.json"

        if not poses_path.exists():
            q.put(("status", "Building image pair graph…"))
            subprocess.run([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_pair_graph.py"),
                "--data-dir",        str(image_dir),
                "--output-dir",      str(pair_dir),
                "--neighbor-offsets", neighbor_offsets,
                "--ext",             ext,
            ], check=True, capture_output=True)

            q.put(("status", "Solving global poses…"))
            subprocess.run([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "solve_global_no_ba.py"),
                "--pair-graph-dir", str(pair_dir),
                "--output-dir",     str(global_dir),
            ], check=True, capture_output=True)
        else:
            q.put(("status", "Reusing existing global poses…"))

        if not poses_path.exists():
            q.put(("error", "Stitching failed — no global_poses.json produced"))
            return

        with poses_path.open() as f:
            payload = json.load(f)
        nodes: list[dict] = payload.get("nodes", [])
        if not nodes:
            q.put(("error", "No nodes in global poses"))
            return

        # ── 3. Canvas geometry ─────────────────────────────────────────
        cfg = load_config(PROJECT_ROOT / "configs" / "stitching.yaml")
        cfg.data_dir = str(image_dir)

        all_corners: list[np.ndarray] = []
        for n in nodes:
            sh, Hr = n.get("image_processed_shape"), n.get("H_to_anchor")
            if sh and Hr:
                h, w  = int(sh[0]), int(sh[1])
                H     = np.array(Hr, dtype=np.float64)
                c     = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
                all_corners.append(cv2.perspectiveTransform(c, H).reshape(-1,2))

        if not all_corners:
            q.put(("error", "Cannot compute canvas bounds"))
            return

        pts   = np.vstack(all_corners)
        x_min = float(np.floor(pts[:,0].min()))
        y_min = float(np.floor(pts[:,1].min()))
        x_max = float(np.ceil(pts[:,0].max()))
        y_max = float(np.ceil(pts[:,1].max()))
        base_w, base_h = int(x_max - x_min), int(y_max - y_min)

        rs = min(1.0,
                 RENDER_MAX_SIDE / max(base_w, base_h),
                 float(np.sqrt(RENDER_MAX_AREA / max(base_w*base_h, 1))))
        final_w, final_h = int(base_w * rs), int(base_h * rs)

        T = np.array([[rs, 0.0, -x_min*rs],
                      [0.0, rs, -y_min*rs],
                      [0.0, 0.0,      1.0]], dtype=np.float64)

        mosaic_gsd = mean_gsd / rs
        q.put(("canvas", final_w, final_h, mosaic_gsd))

        # ── 4. Load YOLO (Raphael's final model) ──────────────────────
        yolo = sam = None
        if yolo_weights and Path(yolo_weights).exists():
            q.put(("status", f"Loading YOLO model (device: {DEVICE})…"))
            from ultralytics import YOLO, SAM
            yolo = YOLO(str(yolo_weights))
            q.put(("status", f"Loading SAM model (device: {DEVICE})…"))
            try:
                sam = SAM("mobile_sam.pt")   # small + fast
            except Exception:
                try:
                    sam = SAM("sam_b.pt")
                except Exception:
                    sam = None
                    q.put(("status", "SAM not available — using YOLO boxes only"))

        nodes_sorted = sorted(nodes, key=lambda n: _parse_idx(str(n.get("image",""))))

        # Shared canvas buffers
        accum   = np.zeros((final_h, final_w, 3), dtype=np.float32)
        weights = np.zeros((final_h, final_w),    dtype=np.float32)
        overlay = np.zeros((final_h, final_w, 3), dtype=np.uint8)   # BGR colour
        ovr_msk = np.zeros((final_h, final_w),    dtype=np.float32) # alpha

        # ── 5. Incremental render loop ────────────────────────────────
        placed = 0
        for i, node in enumerate(nodes_sorted, 1):
            name = str(node.get("image",""))
            sh   = node.get("image_processed_shape")
            Hr   = node.get("H_to_anchor")
            if not name or sh is None or Hr is None:
                continue

            q.put(("status", f"Frame {i}/{len(nodes_sorted)}  {name}"))
            q.put(("progress", i, len(nodes_sorted)))

            try:
                img_path = resolve_image_path(name, cfg.data_dir)

                # ── 5a. Load + resize to stitching dims ───────────────
                img      = load_image(img_path)
                img_proc, _ = resize_by_max_dim(img, cfg.resize_max_dim)
                th, tw = int(sh[0]), int(sh[1])
                if img_proc.shape[:2] != (th, tw):
                    img_proc = cv2.resize(img_proc, (tw, th),
                                          interpolation=cv2.INTER_AREA)

                H        = np.array(Hr, dtype=np.float64)
                warp_mat = T @ H

                # ── 5b. Warp image onto accumulation canvas ────────────
                warped  = cv2.warpPerspective(
                    img_proc, warp_mat, (final_w, final_h), flags=cv2.INTER_LINEAR)
                src_mask = np.ones((th, tw), dtype=np.uint8) * 255
                wmask    = cv2.warpPerspective(
                    src_mask, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)
                wf = wmask.astype(np.float32) / 255.0

                accum   += warped.astype(np.float32) * wf[..., None]
                weights += wf
                placed  += 1

                denom  = np.maximum(weights, 1e-6)
                mosaic = (accum / denom[..., None]).astype(np.uint8)
                mosaic[weights <= 0] = 0

                # ── 5c. Detect (Raphael tiled YOLO) ───────────────────
                if yolo is not None:
                    orig_bgr = cv2.imread(str(img_path))
                    oh, ow   = orig_bgr.shape[:2]
                    sx, sy   = tw / ow, th / oh

                    boxes, scores, classes = _tiled_detect(
                        orig_bgr, yolo, conf=conf, iou=0.6)

                    if len(boxes) > 0:
                        # ── 5d. Segment (Kane SAM within YOLO boxes) ──
                        det_layer = np.zeros((th, tw, 3), dtype=np.uint8)
                        det_alpha = np.zeros((th, tw),    dtype=np.uint8)

                        if sam is not None:
                            try:
                                # Scale boxes to processed image size
                                scaled_boxes = boxes.copy()
                                scaled_boxes[:, [0,2]] *= sx
                                scaled_boxes[:, [1,3]] *= sy
                                sam_res = sam.predict(
                                    source=str(img_path),
                                    bboxes=scaled_boxes.tolist(),
                                    device=DEVICE,
                                    verbose=False)[0]
                                if sam_res.masks is not None:
                                    mask_data = sam_res.masks.data.cpu().numpy()
                                    for mi, mask in enumerate(mask_data):
                                        cls_id = int(classes[mi]) if mi < len(classes) else 0
                                        color  = CLASS_BGR.get(cls_id % len(CLASS_BGR),
                                                               (128, 128, 128))
                                        # Resize mask to processed image dims
                                        m = (cv2.resize(
                                            mask.astype(np.float32), (tw, th),
                                            interpolation=cv2.INTER_NEAREST) > 0.5)
                                        det_layer[m] = color
                                        det_alpha[m] = 255
                            except Exception as e:
                                # SAM failed — fall back to filled boxes
                                sam = None
                                q.put(("status", f"SAM error, using boxes: {e}"))

                        if sam is None or det_alpha.sum() == 0:
                            # Fallback: draw filled bounding boxes
                            for box, cls_id in zip(boxes, classes):
                                x1, y1, x2, y2 = box
                                color = CLASS_BGR.get(int(cls_id) % len(CLASS_BGR),
                                                      (128, 128, 128))
                                cv2.rectangle(det_layer,
                                              (int(x1*sx), int(y1*sy)),
                                              (int(x2*sx), int(y2*sy)),
                                              color, -1)
                                cv2.rectangle(det_alpha,
                                              (int(x1*sx), int(y1*sy)),
                                              (int(x2*sx), int(y2*sy)),
                                              255, -1)

                        # Warp detection layer to mosaic canvas
                        wd = cv2.warpPerspective(
                            det_layer, warp_mat, (final_w, final_h),
                            flags=cv2.INTER_NEAREST)
                        wa = cv2.warpPerspective(
                            det_alpha, warp_mat, (final_w, final_h),
                            flags=cv2.INTER_NEAREST)
                        dm = wa.astype(np.float32) / 255.0

                        for c in range(3):
                            overlay[:,:,c] = np.where(dm > 0.5, wd[:,:,c], overlay[:,:,c])
                        ovr_msk = np.maximum(ovr_msk, dm)

                # ── 5e. Composite mosaic + disaster overlay ────────────
                frame = mosaic.astype(np.float32)
                for c in range(3):
                    frame[:,:,c] = np.where(
                        ovr_msk > 0.5,
                        frame[:,:,c] * (1 - OVERLAY_ALPHA) + overlay[:,:,c] * OVERLAY_ALPHA,
                        frame[:,:,c])
                frame = np.clip(frame, 0, 255).astype(np.uint8)

                # Green outline — newest frame boundary
                try:
                    bm = cv2.warpPerspective(
                        src_mask, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)
                    ctrs, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, ctrs, -1, (0, 220, 100), 2)
                except Exception:
                    pass

                q.put(("frame", frame))

            except Exception as exc:
                q.put(("status", f"Skipped {name}: {exc}"))

            time.sleep(0.05)

        q.put(("status",
               f"✓ Done — {placed}/{len(nodes_sorted)} frames  |  "
               f"Click two points to measure distance"))
        q.put(("done",))

    except Exception:
        import traceback
        q.put(("error", traceback.format_exc()))


# ── GUI ───────────────────────────────────────────────────────────────────

class LiveViewApp:
    BG     = "#1a1a2e"
    PANEL  = "#16213e"
    ACC    = "#00c896"
    FG     = "#e0e0e0"
    SUBTLE = "#555566"

    ZOOM_MIN      = 0.05
    ZOOM_MAX      = 20.0
    DRAG_THRESHOLD = 4

    def __init__(self, root: tk.Tk, q: queue.Queue) -> None:
        self.root = root
        self.q    = q

        self.mosaic_bgr: np.ndarray | None = None
        self.mosaic_gsd: float | None      = None

        # Viewport
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Drag tracking
        self._drag_start:   tuple[int,int] | None = None
        self._pan_on_press: tuple[float,float]    = (0.0, 0.0)
        self._drag_moved = False

        # Measurement
        self.click_pts: list[tuple[int,int]] = []

        self._tk_img = None
        self._build_ui()
        root.after(80, self._poll)

    # ── Layout ────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        r = self.root
        r.configure(bg=self.BG)
        r.title("StitchWise — Live Disaster Map")

        tk.Label(r, text="🛰   StitchWise  Live Disaster Map",
                 font=("Helvetica", 13, "bold"),
                 bg=self.PANEL, fg=self.ACC, pady=9).pack(fill=tk.X)

        body = tk.Frame(r, bg=self.BG)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4,0))

        self.canvas = tk.Canvas(body, bg="#0d0d1a", cursor="fleur",
                                width=DISPLAY_W, height=DISPLAY_H,
                                highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",  self._on_press)
        self.canvas.bind("<B1-Motion>",      self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",self._on_release)
        self.canvas.bind("<MouseWheel>",     self._on_wheel)
        self.canvas.bind("<Button-4>",       lambda e: self._zoom_at(e,  1.15))
        self.canvas.bind("<Button-5>",       lambda e: self._zoom_at(e, 1/1.15))
        self.canvas.bind("<Double-Button-1>",lambda _: self._fit_to_window())
        self.canvas.bind("<Configure>",      self._on_canvas_resize)

        # ── Right panel ───────────────────────────────────────────────
        rp = tk.Frame(body, bg=self.PANEL, width=200)
        rp.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
        rp.pack_propagate(False)

        tk.Label(rp, text="Detected Classes",
                 font=("Helvetica", 10, "bold"),
                 bg=self.PANEL, fg=self.ACC).pack(pady=(14,6))

        for cls_id, name in CLASS_NAMES.items():
            rx, gx, bx = CLASS_RGB[cls_id]
            row = tk.Frame(rp, bg=self.PANEL)
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text="■", fg=f"#{rx:02x}{gx:02x}{bx:02x}",
                     bg=self.PANEL, font=("Helvetica", 14)).pack(side=tk.LEFT)
            tk.Label(row, text=name, bg=self.PANEL, fg="#aaaaaa",
                     font=("Helvetica", 8), wraplength=150,
                     justify=tk.LEFT).pack(side=tk.LEFT, padx=4)

        tk.Frame(rp, bg="#2e2e50", height=1).pack(fill=tk.X, padx=10, pady=12)

        tk.Label(rp, text="📏  Distance Tool",
                 font=("Helvetica", 10, "bold"),
                 bg=self.PANEL, fg=self.ACC).pack()
        tk.Label(rp, text="Left-click two points on the map",
                 bg=self.PANEL, fg="#666677",
                 font=("Helvetica", 8), wraplength=170).pack(pady=(3,0))

        self.dist_var = tk.StringVar(value="—")
        tk.Label(rp, textvariable=self.dist_var,
                 bg=self.PANEL, fg=self.ACC,
                 font=("Helvetica", 22, "bold")).pack(pady=(8,2))

        self.pt_var = tk.StringVar(value="")
        tk.Label(rp, textvariable=self.pt_var,
                 bg=self.PANEL, fg="#555566",
                 font=("Helvetica", 7), wraplength=185,
                 justify=tk.LEFT).pack()

        btn_row = tk.Frame(rp, bg=self.PANEL)
        btn_row.pack(pady=8)
        for label, cmd in [("Clear pts", self._clear_pts),
                           ("Fit",        self._fit_to_window)]:
            tk.Button(btn_row, text=label, command=cmd,
                      bg="#252545", fg=self.FG, activebackground="#333355",
                      relief=tk.FLAT, padx=7, pady=4, cursor="hand2",
                      ).pack(side=tk.LEFT, padx=2)

        tk.Label(rp, text="Drag · Scroll zoom · Dbl-click fit",
                 bg=self.PANEL, fg="#444455",
                 font=("Helvetica", 7), justify=tk.CENTER).pack()

        self.zoom_var = tk.StringVar(value="")
        tk.Label(rp, textvariable=self.zoom_var,
                 bg=self.PANEL, fg=self.SUBTLE,
                 font=("Helvetica", 8)).pack(pady=(4,0))

        self.gsd_var = tk.StringVar(value="")
        tk.Label(rp, textvariable=self.gsd_var,
                 bg=self.PANEL, fg=self.SUBTLE,
                 font=("Helvetica", 7)).pack(side=tk.BOTTOM, pady=6)

        # ── Bottom bar ────────────────────────────────────────────────
        bot = tk.Frame(r, bg=self.PANEL, pady=5)
        bot.pack(fill=tk.X)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#252545",
                        background=self.ACC, bordercolor=self.PANEL)

        self.progress = ttk.Progressbar(bot, style="green.Horizontal.TProgressbar",
                                        length=420, mode="determinate")
        self.progress.pack(side=tk.LEFT, padx=10, pady=3)

        self.status_var = tk.StringVar(value="Starting…")
        tk.Label(bot, textvariable=self.status_var,
                 bg=self.PANEL, fg="#888899",
                 font=("Helvetica", 9), anchor="w",
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ── Queue polling ──────────────────────────────────────────────────────
    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _handle(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "status":
            self.status_var.set(msg[1])
        elif kind == "progress":
            _, n, total = msg
            self.progress.configure(maximum=total, value=n)
        elif kind == "canvas":
            _, w, h, gsd = msg
            self.mosaic_gsd = gsd
            self.gsd_var.set(f"GSD ≈ {gsd*100:.2f} cm/px")
        elif kind == "frame":
            self.mosaic_bgr = msg[1]
            if self.zoom == 1.0 and self.pan_x == 0.0:
                self._fit_to_window()
            else:
                self._refresh_display()
        elif kind == "done":
            self.progress.configure(value=self.progress["maximum"])
        elif kind == "error":
            self.status_var.set("ERROR — " + str(msg[1])[:120])

    # ── Viewport rendering ────────────────────────────────────────────────
    def _fit_to_window(self) -> None:
        if self.mosaic_bgr is None:
            return
        cw = self.canvas.winfo_width()  or DISPLAY_W
        ch = self.canvas.winfo_height() or DISPLAY_H
        mh, mw = self.mosaic_bgr.shape[:2]
        self.zoom  = min(cw / mw, ch / mh)
        self.pan_x = (cw - mw * self.zoom) / 2
        self.pan_y = (ch - mh * self.zoom) / 2
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self.mosaic_bgr is None:
            return

        cw = self.canvas.winfo_width()  or DISPLAY_W
        ch = self.canvas.winfo_height() or DISPLAY_H
        mh, mw = self.mosaic_bgr.shape[:2]

        margin = 40
        self.pan_x = max(margin - mw*self.zoom, min(cw - margin, self.pan_x))
        self.pan_y = max(margin - mh*self.zoom, min(ch - margin, self.pan_y))

        x0 = max(0, int(-self.pan_x / self.zoom))
        y0 = max(0, int(-self.pan_y / self.zoom))
        x1 = min(mw, int(np.ceil((cw - self.pan_x) / self.zoom)) + 1)
        y1 = min(mh, int(np.ceil((ch - self.pan_y) / self.zoom)) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        crop   = self.mosaic_bgr[y0:y1, x0:x1]
        disp_w = max(1, min(cw, int(round((x1-x0)*self.zoom))))
        disp_h = max(1, min(ch, int(round((y1-y0)*self.zoom))))

        pil = Image.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ).resize((disp_w, disp_h), Image.LANCZOS)

        screen_x = int(self.pan_x + x0 * self.zoom)
        screen_y = int(self.pan_y + y0 * self.zoom)

        if self.click_pts:
            draw   = ImageDraw.Draw(pil)
            colors = ["#00ff80", "#ff4444"]
            dpts   = []
            for mx, my in self.click_pts:
                dpts.append((int((mx-x0)*self.zoom), int((my-y0)*self.zoom)))
            for i, (sx, sy) in enumerate(dpts):
                if -15 <= sx <= disp_w+15 and -15 <= sy <= disp_h+15:
                    rr = 7
                    draw.ellipse([sx-rr, sy-rr, sx+rr, sy+rr],
                                 fill=colors[i], outline="white", width=2)
                    draw.text((sx+11, sy-9), "AB"[i], fill="white")
            if len(dpts) == 2:
                draw.line([dpts[0], dpts[1]], fill="#ffff00", width=2)

        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(screen_x, screen_y, anchor="nw", image=self._tk_img)
        self.zoom_var.set(f"{self.zoom*100:.0f}%")

    # ── Interaction handlers ──────────────────────────────────────────────
    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._refresh_display()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start    = (event.x, event.y)
        self._pan_on_press  = (self.pan_x, self.pan_y)
        self._drag_moved    = False

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if abs(dx) > self.DRAG_THRESHOLD or abs(dy) > self.DRAG_THRESHOLD:
            self._drag_moved = True
        if self._drag_moved:
            self.pan_x = self._pan_on_press[0] + dx
            self.pan_y = self._pan_on_press[1] + dy
            self._refresh_display()

    def _on_release(self, event: tk.Event) -> None:
        if not self._drag_moved:
            self._register_click(event.x, event.y)
        self._drag_start = None
        self._drag_moved = False

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom_at(event, 1.15 if event.delta > 0 else 1/1.15)

    def _zoom_at(self, event: tk.Event, factor: float) -> None:
        old = self.zoom
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old * factor))
        if new == old:
            return
        self.pan_x = event.x - (event.x - self.pan_x) * (new / old)
        self.pan_y = event.y - (event.y - self.pan_y) * (new / old)
        self.zoom  = new
        self._refresh_display()

    # ── Distance measurement ──────────────────────────────────────────────
    def _register_click(self, sx: int, sy: int) -> None:
        if self.mosaic_bgr is None:
            return
        mh, mw = self.mosaic_bgr.shape[:2]
        mx = int((sx - self.pan_x) / self.zoom)
        my = int((sy - self.pan_y) / self.zoom)
        if not (0 <= mx < mw and 0 <= my < mh):
            return

        if len(self.click_pts) >= 2:
            self.click_pts.clear()
        self.click_pts.append((mx, my))

        if len(self.click_pts) == 2:
            dx = self.click_pts[1][0] - self.click_pts[0][0]
            dy = self.click_pts[1][1] - self.click_pts[0][1]
            dist_px = float(np.sqrt(dx**2 + dy**2))
            if self.mosaic_gsd:
                dist_m = dist_px * self.mosaic_gsd
                self.dist_var.set(f"{dist_m:.1f} m")
                self.pt_var.set(
                    f"A ({self.click_pts[0][0]}, {self.click_pts[0][1]})\n"
                    f"B ({self.click_pts[1][0]}, {self.click_pts[1][1]})\n"
                    f"{dist_px:.0f} px  ×  {self.mosaic_gsd*100:.2f} cm/px"
                )
            else:
                self.dist_var.set(f"{dist_px:.0f} px")
        else:
            self.dist_var.set("…")
            self.pt_var.set(f"A ({mx}, {my})\nNow click B")

        self._refresh_display()

    def _clear_pts(self) -> None:
        self.click_pts.clear()
        self.dist_var.set("—")
        self.pt_var.set("")
        self._refresh_display()


# ── Entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StitchWise live disaster map viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image-dir",  required=True,
                   help="Directory containing input images")
    p.add_argument("--n-frames",   type=int, default=None,
                   help="Use only the first N frames (default: all)")
    p.add_argument("--ext",        default=".jpg",
                   help="Image extension e.g. .jpg  .JPG  .png")
    p.add_argument("--output-dir", default="outputs/live_view",
                   help="Cache directory for stitching results")
    p.add_argument("--yolo-weights",
                   default=str(PROJECT_ROOT / "detection" / "model" / "best.pt"),
                   help="Path to YOLOv8 weights (Raphael's final model)")
    p.add_argument("--conf",  type=float, default=0.25,
                   help="YOLO confidence threshold")
    p.add_argument("--offsets", default="1,2,3",
                   help="Stitching neighbour offsets")
    p.add_argument("--no-seg",  action="store_true",
                   help="Skip detection + segmentation (plain stitching)")
    p.add_argument("--reuse",   action="store_true",
                   help="Reuse cached stitching results if present")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_lower = args.ext.lower()
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ext_lower
    )
    if not images:
        sys.exit(f"No {args.ext} images found in {image_dir}")
    if args.n_frames:
        images = images[:args.n_frames]

    if not args.reuse:
        import shutil
        for d in ["pair_graph", "global_no_ba"]:
            t = output_dir / d
            if t.exists():
                shutil.rmtree(t)

    yolo_weights = None if args.no_seg else args.yolo_weights

    print(f"Images  : {len(images)}")
    print(f"Output  : {output_dir}")
    print(f"Segment : {'disabled' if args.no_seg else yolo_weights}")

    q: queue.Queue = queue.Queue()
    root = tk.Tk()
    root.resizable(True, True)
    root.minsize(700, 520)
    app = LiveViewApp(root, q)

    t = threading.Thread(
        target=pipeline_worker,
        kwargs=dict(
            image_dir=image_dir,
            images=images,
            output_dir=output_dir,
            yolo_weights=yolo_weights,
            conf=args.conf,
            neighbor_offsets=args.offsets,
            ext=args.ext,
            q=q,
        ),
        daemon=True,
    )
    t.start()
    root.mainloop()


if __name__ == "__main__":
    main()
