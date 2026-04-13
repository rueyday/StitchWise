"""
StitchWise GUI
==============
Run with:
    streamlit run app.py

Features
--------
* Upload images or point to a local directory
* Run the full pipeline (metric scale → segmentation → stitching → overlay)
* Interactive mosaic viewer with disaster region colour overlay
* Click-to-measure distance tool (two clicks → exact distance in metres)
* Per-image browser showing original + segmentation side-by-side
* GSD statistics table + timeline chart
* Incremental update: add new images and re-run without redoing everything
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO = str(PROJECT_ROOT / "detection" / "model" / "best.pt")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "pipeline")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StitchWise",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "pipeline_result": None,
    "running": False,
    "run_log": [],
    "click_pts": [],       # [(x, y), ...] in preview pixel space
    "gsd_override": None,  # user-calibrated GSD m/px (preview space)
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_summary(output_dir: str) -> dict | None:
    p = Path(output_dir) / "pipeline_summary.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def cv2_read_rgb(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_pipeline_thread(
    image_dir: str,
    output_dir: str,
    yolo_weights: str,
    conf: float,
    sam_model: str,
    neighbor_offsets: str,
    scale_bar_m: float,
    image_ext: str,
    use_depth_fallback: bool,
) -> None:
    """Background thread target — runs pipeline and stores result in session state."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from run_pipeline import run_full_pipeline

    def progress_cb(msg: str, pct: float) -> None:
        st.session_state.run_log.append((pct, msg))

    try:
        result = run_full_pipeline(
            image_dir=image_dir,
            output_dir=output_dir,
            yolo_weights=yolo_weights if Path(yolo_weights).exists() else None,
            conf=conf,
            sam_model=sam_model,
            neighbor_offsets=neighbor_offsets,
            scale_bar_m=scale_bar_m,
            use_depth_fallback=use_depth_fallback,
            image_ext=image_ext,
            progress_callback=progress_cb,
        )
        st.session_state.pipeline_result = result
        st.session_state.gsd_override = result.get("preview_gsd_m_per_px")
    except Exception as exc:
        st.session_state.run_log.append((1.0, f"ERROR: {exc}"))
    finally:
        st.session_state.running = False


# ── Class colour palette (RGB for display) ───────────────────────────────────
CLASS_COLORS_RGB = {
    0: (0, 100, 200),   # Water / Flooding
    1: (255, 140, 0),   # Building Major Damage
    2: (220, 0, 0),     # Building Total Destruction
    3: (200, 0, 150),   # Road Blocked
    4: (0, 200, 200),   # Vehicle
}
CLASS_NAMES = {
    0: "Water / Flooding",
    1: "Building Major Damage",
    2: "Building Total Destruction",
    3: "Road Blocked",
    4: "Vehicle",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ StitchWise")
    st.caption("Disaster region mapping from drone imagery")
    st.divider()

    # ── Input images
    st.subheader("📂 Input")
    input_mode = st.radio("Source", ["Local directory", "Upload files"], horizontal=True)

    image_dir_input = None
    if input_mode == "Local directory":
        image_dir_input = st.text_input("Image directory", placeholder="data/my_images")
        image_ext = st.selectbox("Extension", [".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff"])
    else:
        uploaded = st.file_uploader(
            "Upload images", type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True)
        image_ext = ".jpg"
        if uploaded:
            upload_dir = PROJECT_ROOT / "uploads"
            upload_dir.mkdir(exist_ok=True)
            for uf in uploaded:
                dest = upload_dir / uf.name
                dest.write_bytes(uf.read())
            image_dir_input = str(upload_dir)
            st.success(f"Saved {len(uploaded)} images to uploads/")

    st.divider()

    # ── Model settings
    st.subheader("🤖 Models")
    yolo_weights = st.text_input("YOLO weights", value=DEFAULT_YOLO)
    col_sam, col_conf = st.columns(2)
    sam_model = col_sam.selectbox("SAM model", ["sam_b.pt", "mobile_sam.pt", "sam_l.pt"])
    conf = col_conf.slider("Confidence", 0.1, 0.9, 0.25, 0.05)

    yolo_ok = Path(yolo_weights).exists()
    if yolo_ok:
        st.caption("✅ YOLO weights found")
    else:
        st.caption("⚠️ YOLO weights not found — segmentation will be skipped")

    st.divider()

    # ── Pipeline settings
    st.subheader("⚙️ Settings")
    output_dir = st.text_input("Output directory", value=DEFAULT_OUTPUT)
    neighbor_offsets = st.text_input("Pair offsets", value="1,2,3",
                                     help="Stitching: compare each frame to its ±N neighbours")
    scale_bar_m = st.slider("Scale bar (m)", 1.0, 100.0, 10.0, 1.0)
    use_depth = st.checkbox("Use depth model fallback", value=True,
                            help="Depth Anything V2 for GSD when EXIF is incomplete")

    st.divider()

    # ── Run button
    run_disabled = st.session_state.running or not image_dir_input
    if st.button("▶ Run Pipeline", disabled=run_disabled, use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.run_log = []
        st.session_state.click_pts = []
        t = threading.Thread(
            target=run_pipeline_thread,
            kwargs=dict(
                image_dir=image_dir_input,
                output_dir=output_dir,
                yolo_weights=yolo_weights,
                conf=conf,
                sam_model=sam_model,
                neighbor_offsets=neighbor_offsets,
                scale_bar_m=scale_bar_m,
                image_ext=image_ext,
                use_depth_fallback=use_depth,
            ),
            daemon=True,
        )
        t.start()
        st.rerun()

    # Auto-load existing results
    if not st.session_state.pipeline_result:
        existing = load_summary(output_dir)
        if existing:
            st.session_state.pipeline_result = existing
            st.session_state.gsd_override = existing.get("preview_gsd_m_per_px")

    # ── Progress display
    if st.session_state.running:
        st.info("Pipeline running…")
        for pct, msg in st.session_state.run_log[-8:]:
            st.progress(pct, text=msg)
        st.button("🔄 Refresh", on_click=st.rerun)

    if st.session_state.pipeline_result and not st.session_state.running:
        r = st.session_state.pipeline_result
        st.success(f"✅ {r.get('images_processed', '?')} images processed")
        st.metric("Mean GSD", f"{r.get('mean_gsd_m_per_px', 0):.4f} m/px")


# ── Main content ──────────────────────────────────────────────────────────────
result = st.session_state.pipeline_result

if not result and not st.session_state.running:
    st.title("StitchWise — Disaster Region Mapping")
    st.markdown("""
    **How to use:**
    1. Set an image directory (or upload files) in the sidebar
    2. Confirm YOLO weights path is correct
    3. Click **▶ Run Pipeline**

    The pipeline will:
    - Estimate ground sampling distance (GSD) for each image
    - Detect and segment disaster regions (flood, structural damage, road blockages)
    - Stitch all images into a single georeferenced mosaic
    - Overlay disaster regions with colour coding and a metric scale bar
    """)
    st.stop()

if st.session_state.running:
    st.title("Pipeline in progress…")
    for pct, msg in st.session_state.run_log:
        st.write(f"`{pct*100:.0f}%` {msg}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_mosaic, tab_images, tab_stats = st.tabs(["🗺️ Mosaic", "🖼️ Per Image", "📊 GSD Stats"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Mosaic view + distance tool
# ════════════════════════════════════════════════════════════════════════════
with tab_mosaic:
    st.header("Disaster Mosaic")

    preview_path = result.get("disaster_preview")
    if not preview_path or not Path(preview_path).exists():
        # Fall back to plain mosaic preview
        preview_path = str(Path(result.get("mosaic_tif", "")).parent / "mosaic_no_ba_preview.jpg")

    mosaic_rgb = cv2_read_rgb(preview_path) if preview_path and Path(preview_path).exists() else None

    if mosaic_rgb is None:
        st.warning("Mosaic image not found. Check output directory.")
    else:
        ph, pw = mosaic_rgb.shape[:2]

        # ── Distance tool ──────────────────────────────────────────────
        st.subheader("📏 Distance Measurement")
        col_tool, col_gsd = st.columns([3, 1])

        with col_gsd:
            default_gsd = st.session_state.gsd_override or result.get("preview_gsd_m_per_px", 0.05)
            user_gsd = st.number_input(
                "GSD (m/px) — calibrate if needed",
                value=float(f"{default_gsd:.6f}"),
                format="%.6f",
                step=0.000001,
                help="Metres per mosaic preview pixel. Auto-computed from EXIF + stitching scale.",
            )

        with col_tool:
            pts = st.session_state.click_pts
            n_pts = len(pts)
            if n_pts == 0:
                st.info("Click a point on the mosaic below to start measuring.")
            elif n_pts == 1:
                st.info(f"Point A set at ({pts[0][0]}, {pts[0][1]}).  Click a second point.")
            else:
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                dist_px = float(np.sqrt(dx**2 + dy**2))
                dist_m = dist_px * user_gsd
                st.success(
                    f"**Distance: {dist_m:.2f} m**  "
                    f"({dist_px:.1f} px · GSD {user_gsd:.5f} m/px)"
                )

            if st.button("🔄 Reset points"):
                st.session_state.click_pts = []
                st.rerun()

        # ── Interactive mosaic with Plotly ─────────────────────────────
        # Draw measurement overlay on image copy
        display_img = mosaic_rgb.copy()
        for i, (px_, py_) in enumerate(st.session_state.click_pts):
            cv2.circle(display_img, (px_, py_), 8, (255, 80, 0), -1)
            cv2.circle(display_img, (px_, py_), 10, (255, 255, 255), 2)
            cv2.putText(display_img, "AB"[i], (px_ + 12, py_ - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if len(st.session_state.click_pts) == 2:
            cv2.line(display_img,
                     st.session_state.click_pts[0], st.session_state.click_pts[1],
                     (255, 80, 0), 2)

        fig = px.imshow(display_img, binary_string=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            dragmode="pan",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        fig.update_traces(hovertemplate="x=%{x}  y=%{y}<extra></extra>")

        # Plotly click → capture via plotly_events if available, else use workaround
        try:
            from streamlit_plotly_events import plotly_events  # type: ignore
            clicked = plotly_events(fig, click_event=True, key="mosaic_plot",
                                    override_height=600)
            if clicked:
                cx, cy = int(clicked[0]["x"]), int(clicked[0]["y"])
                cx = max(0, min(cx, pw - 1))
                cy = max(0, min(cy, ph - 1))
                pts = st.session_state.click_pts
                if len(pts) < 2:
                    pts.append((cx, cy))
                else:
                    st.session_state.click_pts = [(cx, cy)]
                st.rerun()
        except ImportError:
            # Fallback: standard Plotly chart + manual coordinate entry
            st.plotly_chart(fig, use_container_width=True, key="mosaic_plot_static")
            st.caption(
                "💡 Install `streamlit-plotly-events` for click-to-measure: "
                "`pip install streamlit-plotly-events`"
            )
            with st.expander("Manual coordinate entry"):
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.write("**Point A**")
                    ax = st.number_input("A x", 0, pw, value=pts[0][0] if pts else pw // 4, key="ax")
                    ay = st.number_input("A y", 0, ph, value=pts[0][1] if pts else ph // 2, key="ay")
                with mc2:
                    st.write("**Point B**")
                    bx = st.number_input("B x", 0, pw, value=pts[1][0] if len(pts) > 1 else 3 * pw // 4, key="bx")
                    by = st.number_input("B y", 0, ph, value=pts[1][1] if len(pts) > 1 else ph // 2, key="by")
                if st.button("Compute distance"):
                    st.session_state.click_pts = [(ax, ay), (bx, by)]
                    st.rerun()

        # ── Legend ─────────────────────────────────────────────────────
        if result.get("has_segmentation"):
            st.subheader("Legend")
            cols = st.columns(len(CLASS_NAMES))
            for col, (cls_id, name) in zip(cols, CLASS_NAMES.items()):
                r_, g, b = CLASS_COLORS_RGB[cls_id]
                hex_color = f"#{r_:02x}{g:02x}{b:02x}"
                col.markdown(
                    f'<div style="background:{hex_color};width:20px;height:20px;'
                    f'display:inline-block;border-radius:3px;margin-right:6px;'
                    f'vertical-align:middle"></div>{name}',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Segmentation was not run — no disaster regions annotated.")

        # ── Download button ─────────────────────────────────────────────
        full_path = result.get("disaster_mosaic")
        if full_path and Path(full_path).exists():
            with open(full_path, "rb") as fh:
                st.download_button(
                    "⬇ Download full-resolution disaster mosaic",
                    data=fh.read(),
                    file_name="disaster_mosaic.jpg",
                    mime="image/jpeg",
                )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Per-image browser
# ════════════════════════════════════════════════════════════════════════════
with tab_images:
    st.header("Per-Image Results")

    image_dir_path = Path(result.get("gsd_csv", "")).parent.parent.parent
    # gsd_csv is at output_dir/metric_scale/gsd_results.csv
    gsd_dict: dict[str, float] = result.get("gsd_dict", {})
    mask_paths: dict[str, str] = result.get("mask_paths", {})

    stems = sorted(gsd_dict.keys())
    if not stems:
        st.info("No per-image data available.")
    else:
        selected = st.selectbox("Select image", stems, key="img_select")
        if selected:
            gsd_val = gsd_dict.get(selected, None)
            if gsd_val:
                st.metric("GSD", f"{gsd_val:.5f} m/px",
                          help="Ground Sampling Distance: metres per original pixel")

            # Find original image
            output_path = Path(result.get("gsd_csv", "")).parent.parent
            # Search for the image in common locations
            img_path = None
            for ext in [".jpg", ".JPG", ".jpeg", ".png", ".tif"]:
                candidate = output_path.parent / f"{selected}{ext}"
                if not candidate.exists():
                    # Try the pipeline input dir
                    manifest = result.get("manifest_path", "")
                    if manifest:
                        candidate = Path(manifest).parent.parent / f"{selected}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            mask_p = mask_paths.get(selected)

            if img_path or mask_p:
                left_col, right_col = st.columns(2)
                if img_path and img_path.exists():
                    orig = cv2_read_rgb(img_path)
                    if orig is not None:
                        left_col.subheader("Original")
                        left_col.image(orig, use_container_width=True)

                if mask_p and Path(mask_p).exists():
                    mask_gray = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                    if mask_gray is not None:
                        colored = np.zeros((*mask_gray.shape, 3), dtype=np.uint8)
                        for cls_id, rgb in CLASS_COLORS_RGB.items():
                            colored[mask_gray == (cls_id + 1)] = rgb

                        right_col.subheader("Disaster Mask")
                        right_col.image(colored, use_container_width=True)
                        # Class presence
                        present = [CLASS_NAMES[c] for c in CLASS_NAMES
                                   if np.any(mask_gray == (c + 1))]
                        if present:
                            right_col.caption("Detected: " + ", ".join(present))
                        else:
                            right_col.caption("No disaster regions detected")

                # Segmentation visualisation
                viz_dir = output_path / "seg_viz"
                for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                    viz_p = viz_dir / f"viz_{selected}{ext}"
                    if viz_p.exists():
                        st.subheader("YOLO + SAM Overlay")
                        st.image(cv2_read_rgb(viz_p), use_container_width=True)
                        break
            else:
                st.info(f"Could not locate source image for '{selected}'.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — GSD Statistics
# ════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.header("GSD Statistics")

    gsd_csv_path = result.get("gsd_csv")
    if gsd_csv_path and Path(gsd_csv_path).exists():
        import csv as csv_mod
        rows = []
        with open(gsd_csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))

        if rows:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(rows)
            df["gsd_m_per_px"] = df["gsd_m_per_px"].astype(float)
            df["gsd_cm_per_px"] = df["gsd_m_per_px"] * 100

            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean GSD", f"{df['gsd_m_per_px'].mean():.4f} m/px")
            c2.metric("Min GSD", f"{df['gsd_m_per_px'].min():.4f} m/px")
            c3.metric("Max GSD", f"{df['gsd_m_per_px'].max():.4f} m/px")

            # Timeline chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df["gsd_cm_per_px"].tolist(),
                mode="lines+markers",
                name="GSD (cm/px)",
                line=dict(color="#1f77b4"),
            ))
            fig.update_layout(
                title="GSD per Frame",
                xaxis_title="Frame index",
                yaxis_title="GSD (cm/px)",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(
                df[["image", "gsd_cm_per_px", "method"]].rename(columns={
                    "image": "Image",
                    "gsd_cm_per_px": "GSD (cm/px)",
                    "method": "Estimation Method",
                }),
                use_container_width=True,
            )

            # Download
            with open(gsd_csv_path, "rb") as fh:
                st.download_button("⬇ Download GSD CSV", fh.read(),
                                   "gsd_results.csv", "text/csv")
    else:
        st.info("GSD data not available yet. Run the pipeline first.")
