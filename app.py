
import io
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

import streamlit as st
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd


# Helpers
def load_image(file) -> Image.Image:
    img = Image.open(file).convert("RGB")
    return img

def downsample(img: Image.Image, max_dim: int, keep_aspect: bool = True) -> Image.Image:
    if keep_aspect:
        img = ImageOps.contain(img, (max_dim, max_dim), method=Image.Resampling.LANCZOS)
    else:
        img = img.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

def to_ndarray(img: Image.Image, colorspace: str = "RGB") -> np.ndarray:
    if colorspace == "RGB":
        return np.array(img)
    elif colorspace == "GRAY":
        arr = np.array(img).astype(np.float32)
        gray = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
        return gray.astype(np.uint8)
    elif colorspace == "HSV":
        return np.array(img.convert("HSV"))
    else:
        return np.array(img)

def array_to_csv_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        df = pd.DataFrame(arr)
        return df.to_csv(index=False).encode("utf-8")
    else:
        h,w,c = arr.shape
        df = pd.DataFrame(arr.reshape(h*w, c), columns=[f"c{i}" for i in range(c)])
        df.insert(0, "row", np.repeat(np.arange(h), w))
        df.insert(1, "col", np.tile(np.arange(w), h))
        return df.to_csv(index=False).encode("utf-8")

def show_mat_grid(arr2d: np.ndarray, title: str, cmap: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr2d, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(arr2d.shape[1]))
    ax.set_yticks(np.arange(arr2d.shape[0]))
    ax.grid(which="both", color="white", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="both", length=0)
    st.pyplot(fig, clear_figure=True)

# basic processing ops (numpy only)

def clamp_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)

def brightness_contrast(channel: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    # brightness in [-100,100], contrast in [0.1, 3.0]; pivot at 128
    ch = channel.astype(np.float32)
    ch = (ch - 128.0) * contrast + 128.0 + brightness
    return clamp_uint8(ch)

def invert(channel: np.ndarray) -> np.ndarray:
    return 255 - channel

def threshold(channel: np.ndarray, t: int) -> np.ndarray:
    return (channel >= t).astype(np.uint8) * 255

def box_blur(channel: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return channel
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ch = channel.astype(np.float32)
    # integral image for fast box blur
    integral = np.pad(ch, ((1,0),(1,0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    H, W = ch.shape
    y0 = np.arange(H) - pad; y1 = y0 + k
    x0 = np.arange(W) - pad; x1 = x0 + k
    y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
    x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
    out = np.empty_like(ch)
    for i in range(H):
        for j in range(W):
            A = integral[y0[i], x0[j]]
            B = integral[y0[i], x1[j]]
            C = integral[y1[i], x0[j]]
            D = integral[y1[i], x1[j]]
            area = (y1[i]-y0[i])*(x1[j]-x0[j])
            out[i, j] = (D - B - C + A) / max(area, 1)
    return clamp_uint8(out)

def sobel_edges(channel: np.ndarray) -> np.ndarray:
    # simple Sobel magnitude
    ch = channel.astype(np.float32)
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    def conv2(a, k):
        pad = k.shape[0]//2
        a_pad = np.pad(a, ((pad,pad),(pad,pad)), mode="edge")
        H,W = a.shape
        out = np.zeros_like(a, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                region = a_pad[i:i+2*pad+1, j:j+2*pad+1]
                out[i,j] = np.sum(region * k)
        return out
    gx = conv2(ch, Kx)
    gy = conv2(ch, Ky)
    mag = np.sqrt(gx*gx + gy*gy)
    # normalize to 0..255
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    return clamp_uint8(mag)

def sharpen(channel: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    blur = box_blur(channel, radius)
    detail = channel.astype(np.float32) - blur.astype(np.float32)
    out = channel.astype(np.float32) + amount * detail
    return clamp_uint8(out)

def apply_ops(arr: np.ndarray,
              ops: dict,
              channels_to_apply: List[int]) -> np.ndarray:
    """
    Apply selected operations to specified channels.
    Supports both 2D (gray) and 3D arrays.
    """
    if arr.ndim == 2:
        arr_proc = arr.copy()
        if 0 in channels_to_apply or channels_to_apply == [-1]:  # -1 means 'all'
            ch = arr_proc
            if ops["invert"]:
                ch = invert(ch)
            ch = brightness_contrast(ch, ops["brightness"], ops["contrast"])
            if ops["threshold_enable"]:
                ch = threshold(ch, ops["threshold_t"])
            if ops["blur_k"] > 1:
                ch = box_blur(ch, ops["blur_k"])
            if ops["edges"]:
                ch = sobel_edges(ch)
            if ops["sharpen_enable"]:
                ch = sharpen(ch, amount=ops["sharpen_amt"], radius=ops["sharpen_radius"])
            arr_proc = ch
        return arr_proc

    # color / multi-channel
    H, W, C = arr.shape
    arr_proc = arr.copy()
    for k in range(C):
        if k not in channels_to_apply and channels_to_apply != [-1]:
            continue
        ch = arr_proc[:, :, k]
        if ops["invert"]:
            ch = invert(ch)
        ch = brightness_contrast(ch, ops["brightness"], ops["contrast"])
        if ops["threshold_enable"]:
            ch = threshold(ch, ops["threshold_t"])
        if ops["blur_k"] > 1:
            ch = box_blur(ch, ops["blur_k"])
        if ops["edges"]:
            ch = sobel_edges(ch)
        if ops["sharpen_enable"]:
            ch = sharpen(ch, amount=ops["sharpen_amt"], radius=ops["sharpen_radius"])
        arr_proc[:, :, k] = ch
    return arr_proc

#3D Array Cube 

def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    d = a.max() - a.min()
    if d == 0:
        return np.zeros_like(a, dtype=np.float32)
    return (a - a.min()) / d

def build_array_cube(arr: np.ndarray,
                     colorspace: str,
                     point_size: int = 3,
                     opacity: float = 0.85,
                     color_mode: str = "per_channel") -> go.Figure:
    """
    Visualize a 3D array-like cube for image arrays.
    For RGB/HSV (H,W,3): z = channel index.
    For GRAY (H,W): treated as (H,W,1).
    """
    if arr.ndim == 2:
        arr = arr[:, :, None]
    H, W, C = arr.shape

    yy, xx, zz = np.mgrid[0:H, 0:W, 0:C]
    xx = xx.ravel(); yy = yy.ravel(); zz = zz.ravel()

    vals = arr.astype(np.float32).ravel()
    vals01 = _norm01(vals)

    # colors
    if color_mode == "pixel" and C >= 3:
        base_rgb = arr[:, :, :3].astype(np.float32) / 255.0
        rgb_expanded = np.repeat(base_rgb[:, :, None, :], C, axis=2).reshape(-1, 3)
        colors = [mcolors.to_hex(rgb_expanded[i]) for i in range(rgb_expanded.shape[0])]
    else:
        cmap_map = {"RGB": [cm.Reds, cm.Greens, cm.Blues],
                    "HSV": [cm.hsv, cm.viridis, cm.Greys],
                    "GRAY": [cm.Greys]}
        cmaps = cmap_map.get(colorspace, [cm.viridis]*C)
        if len(cmaps) < C:
            cmaps = (cmaps * ((C + len(cmaps) - 1) // len(cmaps)))[:C]
        ch_idx_full = np.repeat(np.arange(C), H*W)
        colors = []
        for ch in range(C):
            mask = (ch_idx_full == ch)
            v = vals01[mask]
            rgba = cmaps[ch](v)
            colors.extend([mcolors.to_hex(r) for r in rgba])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xx, y=yy, z=zz,
        mode='markers',
        marker=dict(size=point_size, opacity=opacity, color=colors),
        name='voxels'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, W-1, W-1, 0, 0, 0, 0, 0, W-1, W-1, W-1, W-1],
        y=[0, 0, H-1, H-1, 0, 0, H-1, H-1, H-1, 0, 0, H-1],
        z=[0, 0, 0, 0, 0, C-1, C-1, 0, 0, 0, C-1, C-1],
        mode='lines',
        line=dict(width=2, color='black'),
        name='bounds'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Column (x)",
            yaxis_title="Row (y)",
            zaxis_title="Slice (channel)",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(orientation='h')
    )
    return fig


# UI

st.set_page_config(page_title="Image Inspector", layout="wide")

st.title("Image Inspector")

with st.sidebar:
    st.subheader("Input")
    use_sample = st.checkbox("Use sample image")
    file = st.file_uploader("Image", type=["png","jpg","jpeg","webp"], disabled=use_sample)

    st.subheader("Downsample")
    max_dim = st.slider("Max dimension", 16, 256, 96)
    keep_aspect = st.toggle("Keep aspect ratio", value=True)

    st.subheader("Color space")
    colorspace = st.selectbox("Color space", ["RGB", "GRAY", "HSV"], index=0)

    # st.subheader("3D view")
    # point_size = st.slider("Point size", 1, 6, 3)
    # opacity = st.slider("Point opacity", 0.1, 1.0, 0.85)
    # color_mode = st.selectbox("Cube color mode", ["per_channel", "pixel"], index=0)

    st.subheader("Processing")
    apply_ops_enable = st.toggle("Enable processing", value=False)
    target = st.selectbox("Apply to", ["All channels", "Red", "Green", "Blue"], index=0)
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", 10, 300, 100) / 100.0
    invert_en = st.toggle("Invert", value=False)
    th_en = st.toggle("Threshold", value=False)
    t_val = st.slider("Threshold value", 0, 255, 128)
    blur_k = st.slider("Box blur kernel", 1, 15, 1, step=2)
    edges = st.toggle("Sobel edges", value=False)
    sharp_en = st.toggle("Sharpen", value=False)
    sharp_amt = st.slider("Sharpen amount", 0.0, 3.0, 1.0)
    sharp_rad = st.slider("Sharpen radius", 1, 9, 3, step=2)


# Pick uploaded file or the built-in sample
if use_sample:
    sample_path = Path(__file__).parent / "images" / "melaniethecat.JPG"
    if not sample_path.exists():
        st.error(f"Sample image not found at: {sample_path}")
        st.stop()
    # Open as a file-like object so the rest of the pipeline is unchanged
    file = open(sample_path, "rb")
    st.caption("Using sample image: images/melaniethecat.JPG")

if not file:
    st.info("Upload an image or enable sample image to begin.")
    st.stop()


# Load and prepare
orig_img = load_image(file)
down_img = downsample(orig_img, max_dim=max_dim, keep_aspect=keep_aspect)
arr = to_ndarray(down_img, colorspace=colorspace)

# Build processing config
ops = {
    "brightness": float(brightness),
    "contrast": float(contrast),
    "invert": bool(invert_en),
    "threshold_enable": bool(th_en),
    "threshold_t": int(t_val),
    "blur_k": int(blur_k),
    "edges": bool(edges),
    "sharpen_enable": bool(sharp_en),
    "sharpen_amt": float(sharp_amt),
    "sharpen_radius": int(sharp_rad),
}

if colorspace == "RGB":
    target_map = {"All channels": [-1], "Red": [0], "Green": [1], "Blue": [2]}
elif colorspace == "HSV":
    target_map = {"All channels": [-1], "Red": [0], "Green": [1], "Blue": [2]}  # labels reused; values are H,S,V
else:
    target_map = {"All channels": [-1], "Red": [0], "Green": [1], "Blue": [2]}  # ignored for gray

channels_to_apply = target_map.get(target, [-1])
arr_proc = apply_ops(arr, ops, channels_to_apply) if apply_ops_enable else arr

# Summary header
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(orig_img, use_container_width=True)
    st.text(f"{orig_img.mode} | {orig_img.size[0]} × {orig_img.size[1]}")
with col2:
    st.subheader("Processed")
    st.image(arr_proc if arr_proc.ndim == 2 else Image.fromarray(arr_proc), use_container_width=True)
    shape_text = f"{arr_proc.shape} (H, W)" if arr_proc.ndim == 2 else f"{arr_proc.shape} (H, W, C)"
    st.text(f"{colorspace} array | shape: {shape_text}")

st.divider()

# Channels (larger previews)
if arr_proc.ndim == 3:
    H, W, C = arr_proc.shape
    st.subheader("Channels")
    labels = {"RGB": ["Red", "Green", "Blue"], "HSV": ["Hue", "Saturation", "Value"]}.get(colorspace, [f"C{i}" for i in range(C)])
    cols = st.columns(C)
    for i in range(C):
        with cols[i]:
            st.markdown(f"**{labels[i]}**")
            st.image(arr_proc[:, :, i], clamp=True, caption=f"Slice [:,:,{i}]", width=360)
else:
    st.subheader("Grayscale")
    st.image(arr_proc, clamp=True, caption="2D array [H, W]", width=360)

st.divider()

# # 3D visualization 
# st.subheader("3D array visualization")
# H, W = arr_proc.shape[:2]
# C = arr_proc.shape[2] if arr_proc.ndim == 3 else 1
# total_points = H * W * C
# if total_points > 80000:
#     st.warning(f"Consider downsampling for responsiveness (points: {total_points:,} > 80,000).")
# fig = build_array_cube(arr_proc, colorspace=colorspace,
#                        point_size=point_size, opacity=opacity, color_mode=color_mode)
# st.plotly_chart(fig, use_container_width=True)

# Grid overlay at the bottom
st.divider()
st.subheader("Grid overlay")
if arr_proc.ndim == 3:
    labels = {"RGB": ["Red", "Green", "Blue"], "HSV": ["Hue", "Saturation", "Value"]}.get(colorspace, [f"C{i}" for i in range(arr_proc.shape[2])])
    idx_channel = st.selectbox("Channel for grid", options=list(range(arr_proc.shape[2])), format_func=lambda i: labels[i])
    show_mat_grid(arr_proc[:, :, idx_channel], f"{labels[idx_channel]} channel ({arr_proc.shape[0]} × {arr_proc.shape[1]})")
else:
    show_mat_grid(arr_proc, f"Grayscale ({arr_proc.shape[0]} × {arr_proc.shape[1]})", cmap=None)

# Downloads
st.divider()
st.subheader("Downloads")
c1, c2, c3 = st.columns(3)
with c1:
    npy_bytes = io.BytesIO()
    np.save(npy_bytes, arr_proc)
    st.download_button("Array (.npy)", data=npy_bytes.getvalue(), file_name="image_array.npy", mime="application/octet-stream")
with c2:
    st.download_button("Array (.csv)", data=array_to_csv_bytes(arr_proc), file_name="image_array.csv", mime="text/csv")
with c3:
    img_bytes = io.BytesIO()
    if arr_proc.ndim == 2:
        Image.fromarray(arr_proc).save(img_bytes, format="PNG")
    else:
        Image.fromarray(arr_proc).save(img_bytes, format="PNG")
    st.download_button("Processed image (PNG)", data=img_bytes.getvalue(), file_name="processed.png", mime="image/png")
