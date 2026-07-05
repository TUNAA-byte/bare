#!/usr/bin/env python3
"""
Bare™ — Desktop App
Removes Instagram/social media filters, restores natural colour,
and smooths skin using professional frequency-separation retouching.

Requirements:  pip install Pillow numpy
Run:           python photo_defilter.py
"""

import sys, os, threading
import tkinter as tk
from tkinter import ttk, filedialog

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
try:    from PIL import Image, ImageFilter, ImageTk, ImageOps
except: _missing.append("Pillow")
try:    import numpy as np
except: _missing.append("numpy")

if _missing:
    root = tk.Tk(); root.withdraw()
    from tkinter import messagebox
    messagebox.showerror(
        "Missing packages",
        f"Please install the required packages then restart:\n\n"
        f"  pip install {' '.join(_missing)}")
    sys.exit(1)

# -- Drag-and-drop support (optional, works when tkinterdnd2 is installed) ----
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _APP_BASE = TkinterDnD.Tk
    _DND_OK   = True
except ImportError:
    _APP_BASE = tk.Tk
    _DND_OK   = False

# -- HEIC/HEIF support (iPhone photos) — optional, active when installed ------
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_OK = True
except ImportError:
    _HEIC_OK = False

# -- Version / auto-update ------------------------------------------------------
VERSION = "1.1.0"
# To enable update checks: create a GitHub repo, upload photo_defilter.py and a
# file called VERSION containing just the version number (e.g. 1.1.1), then set
# these to your repo's raw URLs:
UPDATE_VERSION_URL = "https://raw.githubusercontent.com/TUNAA-byte/bare/main/VERSION"
UPDATE_FILE_URL    = "https://raw.githubusercontent.com/TUNAA-byte/bare/main/photo_defilter.py"

def _parse_ver(v):
    try:    return tuple(int(x) for x in v.strip().split('.'))
    except: return (0,)

# -- AI upscaling helpers ------------------------------------------------------
_AI_MODEL     = 'RealESRGAN_x4plus.pth'
_AI_MODEL_URL = ('https://github.com/xinntao/Real-ESRGAN/releases/'
                 'download/v0.1.0/RealESRGAN_x4plus.pth')

def _ai_available():
    """True when the Real-ESRGAN inference stack is importable."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa
        from realesrgan import RealESRGANer              # noqa
        import torch                                     # noqa
        return True
    except Exception:
        return False

def ai_upscale(img, factor):
    """
    Upscale using Real-ESRGAN x4plus (official CPU inference).
    Returns a PIL Image, or None if the packages are not installed.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, _AI_MODEL)
        if not os.path.exists(model_path):
            model_path = _AI_MODEL_URL   # RealESRGANer auto-downloads

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        ups = RealESRGANer(
            scale=4, model_path=model_path, model=model,
            tile=256, tile_pad=10, pre_pad=0, half=False,
            device=torch.device('cpu'))

        bgr = np.array(img)[:, :, ::-1].copy()
        out, _ = ups.enhance(bgr, outscale=factor)
        return Image.fromarray(out[:, :, ::-1])
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def _rgb_to_hsl(r, g, b):
    """Vectorised RGB (0-255) → HSL (0-1)."""
    r, g, b = r / 255., g / 255., b / 255.
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l  = (mx + mn) / 2.
    d  = mx - mn
    ep = 1e-9
    s  = np.where(d < ep, 0.,
          np.where(l > .5, d / (2 - mx - mn + ep), d / (mx + mn + ep)))
    h = np.zeros_like(r)
    h = np.where((mx == r) & (d > ep), ((g - b) / (d + ep)) % 6 / 6, h)
    h = np.where((mx == g) & (d > ep), ((b - r) / (d + ep) + 2) / 6,  h)
    h = np.where((mx == b) & (d > ep), ((r - g) / (d + ep) + 4) / 6,  h)
    return h, s, l


def _hsl_to_rgb_arr(h, s, l):
    """Vectorised HSL (0-1) → uint8 array H×W×3."""
    q = np.where(l < .5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    def hue2rgb(t):
        t = t % 1.
        return np.select(
            [t < 1/6,  t < .5,   t < 2/3],
            [p+(q-p)*6*t, q, p+(q-p)*(2/3-t)*6],
            default=p)
    r = np.where(s == 0, l, hue2rgb(h + 1/3))
    g = np.where(s == 0, l, hue2rgb(h))
    b = np.where(s == 0, l, hue2rgb(h - 1/3))
    return np.clip(np.stack([r, g, b], axis=2) * 255, 0, 255).astype(np.uint8)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(arr):
    """Return image statistics for auto-parameter detection."""
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    _, sat, _ = _rgb_to_hsl(r, g, b)
    aR, aG, aB = r.mean(), g.mean(), b.mean()
    neu  = (aR + aG + aB) / 3.
    rLo, rHi = np.percentile(r, 1.5), np.percentile(r, 98.5)
    gLo, gHi = np.percentile(g, 1.5), np.percentile(g, 98.5)
    bLo, bHi = np.percentile(b, 1.5), np.percentile(b, 98.5)
    avgLo = (rLo + gLo + bLo) / 3.
    avgHi = (rHi + gHi + bHi) / 3.
    H, W  = arr.shape[:2]
    yy, xx = np.mgrid[:H, :W]
    edge   = (xx < W*.2) | (xx > W*.8) | (yy < H*.2) | (yy > H*.8)
    br     = (r + g + b) / 3.
    vig    = float(br[edge].mean()) / max(float(br[~edge].mean()), 1.)
    return dict(
        castM  = max(abs(aR-neu), abs(aG-neu), abs(aB-neu)),
        vig    = vig,
        avgSat = float(sat.mean()),
        avgLo  = float(avgLo),  avgHi  = float(avgHi),
        rLo    = float(rLo),    rHi    = float(rHi),
        gLo    = float(gLo),    gHi    = float(gHi),
        bLo    = float(bLo),    bHi    = float(bHi),
    )


def auto_params(a):
    return dict(
        cast       = 90 if a['castM']  > 20  else 55 if a['castM']  > 9  else 15,
        vignette   = 80 if a['vig']    < .75 else 45 if a['vig']    < .88 else 0,
        saturation = 80 if a['avgSat'] > .42 else 40 if a['avgSat'] > .30 else 0,
        matte      = 85 if a['avgLo']  > 24  else 38 if a['avgLo']  > 13  else 0,
        skin       = 65,
    )


# ── Skin mask ─────────────────────────────────────────────────────────────────

def _skin_mask(arr):
    """Return a smooth float32 mask (0–1) highlighting skin-toned pixels."""
    h, s, l = _rgb_to_hsl(arr[:,:,0], arr[:,:,1], arr[:,:,2])
    raw = ((h < .15) & (s > .05) & (s < .82) & (l > .10) & (l < .94)).astype(np.float32)
    # Smooth the mask edges so blending doesn't leave hard borders
    mimg = Image.fromarray((raw * 255).astype(np.uint8))
    mimg = mimg.filter(ImageFilter.GaussianBlur(5))
    return np.array(mimg, dtype=np.float32) / 255.


# ── Frequency-separation skin retouching ─────────────────────────────────────

def freq_sep_skin(img, strength):
    """
    Professional-grade skin retouching via frequency separation.

    The image is split into two layers:
      • Low-frequency  — overall tone and colour (large-radius Gaussian)
      • High-frequency — fine texture: pores, fine lines, detail
                         (original minus low-freq, centred at 128)

    Only the LOW-FREQ layer is smoothed in skin areas.
    The HIGH-FREQ layer is added back unchanged.
    Result: blemishes/uneven tone removed, natural texture preserved.
    """
    arr  = np.array(img, dtype=np.float32)
    mask = _skin_mask(arr)
    if mask.mean() < .004:
        return img, False          # no skin found

    W    = img.size[0]
    r1   = max(8,  int(W * .018))  # radius to extract low-freq
    r2   = max(22, int(W * .045))  # radius to smooth low-freq

    low     = np.array(img.filter(ImageFilter.GaussianBlur(r1)), dtype=np.float32)
    high    = arr - low + 128.     # texture layer  (neutral = 128)
    low_sm  = np.array(img.filter(ImageFilter.GaussianBlur(r2)), dtype=np.float32)

    # Blend low → low_sm  only in skin areas, controlled by strength
    blend        = mask[:, :, np.newaxis] * min(strength * .92, .92)
    smoothed_low = low + (low_sm - low) * blend

    # Recompose: smoothed tone + original texture
    result = np.clip(smoothed_low + high - 128., 0, 255).astype(np.uint8)
    return Image.fromarray(result), True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process(img, params):
    """Apply all enabled corrections and return (result_image, list_of_corrections)."""
    arr     = np.array(img, dtype=np.float32)
    H, W    = arr.shape[:2]
    a       = analyze(arr)
    out     = arr.copy()
    done    = []

    cast_s = params.get('cast',       0) / 100.
    vig_s  = params.get('vignette',   0) / 100.
    sat_s  = params.get('saturation', 0) / 100.
    mat_s  = params.get('matte',      0) / 100.
    skin_s = params.get('skin',       0) / 100.

    # 1. Colour cast: gray-world white balance
    #    Computes the per-channel factor needed to bring the average toward neutral,
    #    then applies cast_s% of that correction. Stable on all real images.
    if cast_s > 0:
        done.append('cast')
        aR = float(out[:,:,0].mean())
        aG = float(out[:,:,1].mean())
        aB = float(out[:,:,2].mean())
        neu = (aR + aG + aB) / 3.
        for ci, avg in enumerate([aR, aG, aB]):
            full = neu / max(avg, 1.)          # factor to reach full neutrality
            mult = 1. + (full - 1.) * cast_s * .92
            out[:,:,ci] = np.clip(out[:,:,ci] * mult, 0, 255)

    # 2. Matte / lifted-blacks correction
    if mat_s > 0 and a['avgLo'] > 6:
        done.append('matte')
        mLo   = a['avgLo'] - 2.
        scale = 255. / (255. - mLo)
        fixed = np.clip((out - mLo) * scale, 0, 255)
        out   = out + (fixed - out) * mat_s

    # 3. Vignette removal — radial brightness boost at edges
    if vig_s > 0:
        done.append('vignette')
        yy, xx  = np.mgrid[:H, :W]
        nx = (xx / W - .5) * 2;  ny = (yy / H - .5) * 2
        dist    = np.minimum(1., np.sqrt(nx**2 + ny**2) / np.sqrt(2))
        max_b   = 1 + (1 - a['vig']) * np.power(dist, 1.5) * .88
        boost   = (1 + (max_b - 1) * vig_s)[:, :, np.newaxis]
        out    *= boost

    # 4. Saturation reduction (up to 60 % at full slider)
    if sat_s > 0:
        done.append('saturation')
        tmp     = np.clip(out, 0, 255).astype(np.uint8)
        h_ch, s_ch, l_ch = _rgb_to_hsl(tmp[:,:,0], tmp[:,:,1], tmp[:,:,2])
        new_s   = np.maximum(0., s_ch * (1 - sat_s * .60))
        out     = _hsl_to_rgb_arr(h_ch, new_s, l_ch).astype(np.float32)

    # Brightness guard: undo accidental over-brightening
    orig_mean = arr.mean();  new_mean = out.mean()
    if new_mean > orig_mean * 1.06:
        out *= orig_mean / new_mean

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # 5. Frequency-separation skin retouching
    if skin_s > 0:
        result, detected = freq_sep_skin(result, skin_s)
        if detected:
            done.append('skin')

    return result, done


def _fit(img, max_w, max_h):
    w, h = img.size
    s = min(max_w / w, max_h / h, 1.0)
    return img.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)


def upscale_image(img, factor):
    """
    Upscale using LANCZOS with pre- and post-sharpening for crisp results.

    Steps:
    1. Mild pre-sharpen — accentuates edges before interpolation
    2. LANCZOS resample  — best-quality PIL filter for upscaling
    3. Unsharp mask      — recovers perceived detail lost in interpolation
    """
    if factor <= 1.0:
        return img

    arr = np.array(img, dtype=np.float32)

    # Pre-sharpen: subtract a tiny blur to enhance edges before upscaling
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)
    pre = np.clip(arr + (arr - blurred) * 0.35, 0, 255).astype(np.uint8)

    # Upscale with high-quality LANCZOS
    w, h   = img.size
    new_w  = round(w * factor)
    new_h  = round(h * factor)
    result = Image.fromarray(pre).resize((new_w, new_h), Image.LANCZOS)

    # Post-sharpen to recover interpolation softness
    # More aggressive for larger jumps
    radius  = round(0.6 * factor, 1)
    percent = min(110, int(35 * factor))
    result  = result.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=percent, threshold=2))

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════

C = dict(
    bg     = '#0d0d0d',
    panel  = '#111111',
    border = '#1c1c1c',
    imgbg  = '#080808',
    text   = '#e0e0e0',
    muted  = '#555555',
    hint   = '#2a2a2a',
    accent = '#1D9E75',
    acc_h  = '#0f7a5a',
    btn    = '#1c1c1c',
    btn_h  = '#272727',
)

CORRECTIONS = {
    'cast':       'Colour cast fixed',
    'vignette':   'Vignette removed',
    'saturation': 'Saturation reduced',
    'matte':      'Matte filter cleared',
    'skin':       'Skin smoothed',
}


class App(_APP_BASE):

    def __init__(self):
        super().__init__()
        self.title(f"Bare™  v{VERSION}")
        self.geometry("1240x780")
        self.minsize(900, 600)
        self.configure(bg=C['bg'])

        # Window icon (Start Menu / taskbar) — looks for bare.ico next to this script
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bare.ico')
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception:
            pass

        # State
        self.orig_img    = None
        self.proc_img    = None
        self.orig_tk     = None
        self.proc_tk     = None
        self.filepath    = None
        self._after_id   = None
        self._busy       = False
        self.upscale_var = tk.StringVar(value='off')
        self.ai_var      = tk.BooleanVar(value=False)

        # Undo history: list of param dicts, most recent last
        self._history     = []
        self._history_max = 20

        # Zoom / pan state for the canvas
        self._zoom     = 1.0     # 1.0 = fit-to-window
        self._pan_x    = 0
        self._pan_y    = 0
        self._drag_last = None

        # Presets file lives next to the script
        self._presets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'presets.json')

        self._configure_ttk()
        self._build_header()
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x')
        self._build_body()
        self._center()

        # Keyboard shortcuts
        self.bind_all('<Control-o>', lambda e: self.open_file())
        self.bind_all('<Command-o>', lambda e: self.open_file())
        self.bind_all('<Control-s>', lambda e: self.save_file())
        self.bind_all('<Command-s>', lambda e: self.save_file())
        self.bind_all('<Control-z>', lambda e: self.undo())
        self.bind_all('<Command-z>', lambda e: self.undo())
        self.bind_all('<Control-w>', lambda e: self.destroy())
        self.bind_all('<Command-w>', lambda e: self.destroy())

        # Zoom / pan bindings on the image canvas
        self.canvas.bind('<MouseWheel>',   self._on_wheel)          # Windows / Mac
        self.canvas.bind('<Button-4>',     lambda e: self._on_wheel(e,  1))  # Linux up
        self.canvas.bind('<Button-5>',     lambda e: self._on_wheel(e, -1))  # Linux down
        self.canvas.bind('<ButtonPress-1>',  self._pan_start)
        self.canvas.bind('<B1-Motion>',      self._pan_move)
        self.canvas.bind('<ButtonRelease-1>',lambda e: setattr(self, '_drag_last', None))
        self.canvas.bind('<Double-Button-1>',lambda e: self._zoom_reset())

        # Drag-and-drop (only active when tkinterdnd2 is installed)
        if _DND_OK:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._on_drop)

        # Silent update check in the background (only when URLs are configured)
        if UPDATE_VERSION_URL:
            threading.Thread(target=self._check_updates, daemon=True).start()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _configure_ttk(self):
        s = ttk.Style(self)
        try:    s.theme_use('clam')
        except: pass
        s.configure('D.Horizontal.TScale',
            background=C['panel'], troughcolor='#252525',
            darkcolor='#1D9E75', lightcolor='#1D9E75',
            bordercolor=C['panel'], arrowcolor=C['panel'])
        s.map('D.Horizontal.TScale', background=[('active', C['panel'])])

    def _center(self):
        self.update_idletasks()
        W, H = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  - W) // 2
        y = (self.winfo_screenheight() - H) // 2
        self.geometry(f"+{x}+{y}")

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=C['bg'], padx=20, pady=14)
        hdr.pack(fill='x')
        tk.Label(hdr, text="Bare™", bg=C['bg'], fg=C['text'],
                 font=('Arial', 17, 'bold')).pack(side='left')
        tk.Label(hdr, text="  Remove filters · restore colour · smooth skin",
                 bg=C['bg'], fg=C['muted'], font=('Arial', 12)).pack(side='left')
        bf = tk.Frame(hdr, bg=C['bg'])
        bf.pack(side='right')
        self._btn(bf, "💾  Save", self.save_file, accent=True).pack(side='right', padx=(8,0))
        self._btn(bf, "📂  Open Photo", self.open_file).pack(side='right')

    # ── Body ──────────────────────────────────────────────────────────────────

    def _build_body(self):
        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True)

        # ── Left: image display ───────────────────────────────────────────────
        img_area = tk.Frame(body, bg=C['imgbg'])
        img_area.pack(side='left', fill='both', expand=True)

        label_bar = tk.Frame(img_area, bg=C['imgbg'], padx=16, pady=7)
        label_bar.pack(fill='x')
        tk.Label(label_bar, text="ORIGINAL",   bg=C['imgbg'], fg='#2e2e2e',
                 font=('Arial', 9, 'bold')).pack(side='left')
        tk.Label(label_bar, text="ENHANCED", bg=C['imgbg'], fg=C['accent'],
                 font=('Arial', 9, 'bold')).pack(side='right')

        self.canvas = tk.Canvas(img_area, bg=C['imgbg'], highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=14, pady=(0, 14))
        self.canvas.bind('<Configure>', lambda e: self.redraw())

        self.placeholder = tk.Label(self.canvas,
            text="📂   Open a photo to get started\n\nCtrl+O  /  ⌘O",
            bg=C['imgbg'], fg='#252525', font=('Arial', 15), justify='center')
        self.placeholder.place(relx=.5, rely=.5, anchor='center')

        # ── Right: controls ───────────────────────────────────────────────────
        tk.Frame(body, bg=C['border'], width=1).pack(side='right', fill='y')

        panel = tk.Frame(body, bg=C['panel'], width=268)
        panel.pack(side='right', fill='y')
        panel.pack_propagate(False)

        inner = tk.Frame(panel, bg=C['panel'], padx=18, pady=18)
        inner.pack(fill='both', expand=True)

        tk.Label(inner, text="ADJUSTMENTS", bg=C['panel'], fg='#333',
                 font=('Arial', 9, 'bold')).pack(anchor='w', pady=(0, 14))

        self.svars = {}
        for key, label, default in [
            ('cast',       'Colour cast',    85),
            ('vignette',   'Vignette',        65),
            ('saturation', 'Saturation',      50),
            ('matte',      'Matte / fade',    40),
            ('skin',       'Skin smoothing',  65),
        ]:
            self._mk_slider(inner, key, label, default)

        tk.Frame(inner, bg=C['panel'], height=10).pack()
        btn_row = tk.Frame(inner, bg=C['panel'])
        btn_row.pack(fill='x', pady=(0, 6))
        self._btn(btn_row, "↺  Undo", self.undo).pack(side='left', fill='x', expand=True, padx=(0, 4))
        self._btn(btn_row, "⟲  Auto", self.reset_auto).pack(side='left', fill='x', expand=True, padx=(4, 0))

        # ── Presets section ───────────────────────────────────────────────────
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', pady=(14, 12))
        preset_hdr = tk.Frame(inner, bg=C['panel'])
        preset_hdr.pack(fill='x', pady=(0, 8))
        tk.Label(preset_hdr, text="PRESETS", bg=C['panel'], fg='#333',
                 font=('Arial', 9, 'bold')).pack(side='left')
        save_lbl = tk.Label(preset_hdr, text="+ save current", bg=C['panel'],
                            fg=C['accent'], font=('Arial', 9), cursor='hand2')
        save_lbl.pack(side='right')
        save_lbl.bind('<Button-1>', lambda e: self.save_preset())

        self.preset_var = tk.StringVar(value='')
        self.preset_menu = ttk.Combobox(inner, textvariable=self.preset_var,
            state='readonly', font=('Arial', 10))
        self.preset_menu.pack(fill='x')
        self.preset_menu.bind('<<ComboboxSelected>>', lambda e: self.apply_preset())
        del_lbl = tk.Label(inner, text="delete selected preset", bg=C['panel'],
                           fg='#3a3a3a', font=('Arial', 9), cursor='hand2')
        del_lbl.pack(anchor='w', pady=(4, 0))
        del_lbl.bind('<Button-1>', lambda e: self.delete_preset())
        self._reload_presets()

        # ── Upscale section ───────────────────────────────────────────────────
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', pady=(14, 12))
        tk.Label(inner, text="UPSCALE OUTPUT", bg=C['panel'], fg='#333',
                 font=('Arial', 9, 'bold')).pack(anchor='w', pady=(0, 10))

        scales = [('Off', 'off'), ('4K', '4k'), ('8K', '8k')]
        rb_frame = tk.Frame(inner, bg=C['panel'])
        rb_frame.pack(fill='x')
        for i, (lbl, val) in enumerate(scales):
            rb = tk.Radiobutton(rb_frame, text=lbl, variable=self.upscale_var,
                value=val, bg=C['panel'], fg='#888', selectcolor=C['bg'],
                activebackground=C['panel'], activeforeground=C['accent'],
                font=('Arial', 11), cursor='hand2',
                indicatoron=True, command=self._on_upscale_change)
            rb.grid(row=0, column=i, sticky='w', padx=(0, 14), pady=2)

        # AI upscaling toggle
        ai_row = tk.Frame(inner, bg=C['panel'])
        ai_row.pack(fill='x', pady=(6, 0))
        tk.Checkbutton(ai_row, text="AI upscaling", variable=self.ai_var,
            bg=C['panel'], fg='#888', selectcolor=C['bg'],
            activebackground=C['panel'], activeforeground=C['accent'],
            font=('Arial', 11), cursor='hand2',
            command=self._on_ai_toggle).pack(side='left')
        self.ai_hint = tk.Label(ai_row, bg=C['panel'], fg='#3a3a3a',
            font=('Arial', 9))
        self.ai_hint.pack(side='left', padx=(5, 0))
        self._refresh_ai_hint()

        self.dims_var = tk.StringVar(value='')
        tk.Label(inner, textvariable=self.dims_var, bg=C['panel'], fg='#444',
                 font=('Arial', 9), anchor='w').pack(anchor='w', pady=(6, 0))

        # ── Status / corrections ──────────────────────────────────────────────
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', pady=(14, 12))

        self.status_var = tk.StringVar(value="Open a photo to begin")
        self.status_lbl = tk.Label(inner, textvariable=self.status_var,
            bg=C['panel'], fg='#3a3a3a', font=('Arial', 10),
            justify='left', wraplength=224, anchor='nw')
        self.status_lbl.pack(fill='x', anchor='w')

    # ── Widgets ───────────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, accent=False):
        bg  = C['accent']  if accent else C['btn']
        fg  = '#ffffff'    if accent else C['muted']
        abg = C['acc_h']   if accent else C['btn_h']
        b = tk.Label(parent, text=text, bg=bg, fg=fg,
                     font=('Arial', 11), padx=14, pady=8, cursor='hand2', bd=0)
        b.bind('<Button-1>', lambda e: cmd())
        b.bind('<Enter>',    lambda e: b.configure(bg=abg))
        b.bind('<Leave>',    lambda e: b.configure(bg=bg))
        return b

    def _mk_slider(self, parent, key, label, default):
        frame = tk.Frame(parent, bg=C['panel'])
        frame.pack(fill='x', pady=5)
        top = tk.Frame(frame, bg=C['panel'])
        top.pack(fill='x')
        tk.Label(top, text=label, bg=C['panel'], fg='#777',
                 font=('Arial', 11), anchor='w').pack(side='left')
        sv = tk.IntVar(value=default)
        tk.Label(top, textvariable=sv, bg=C['panel'], fg='#444',
                 font=('Arial', 10), width=3, anchor='e').pack(side='right')
        sc = ttk.Scale(frame, from_=0, to=100, orient='horizontal',
                       variable=sv, style='D.Horizontal.TScale',
                       command=lambda v, k=key: self._on_slide(k))
        sc.pack(fill='x', pady=(3, 0))
        self.svars[key] = sv

    # ── Logic ─────────────────────────────────────────────────────────────────

    # 4K = 3840 long edge, 8K = 7680 long edge
    _RES = {'4k': 3840, '8k': 7680}

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _push_history(self):
        """Snapshot current slider params before a change."""
        p = self.params()
        if self._history and self._history[-1] == p:
            return
        self._history.append(p)
        if len(self._history) > self._history_max:
            self._history.pop(0)

    def undo(self):
        if not self._history:
            self.status_var.set("Nothing to undo")
            return
        p = self._history.pop()
        self._set_sliders(p)
        self.schedule()
        self.status_var.set("Undone")

    def _set_sliders(self, p):
        for k, v in p.items():
            if k in self.svars:
                self.svars[k].set(int(v))

    # ── Presets ───────────────────────────────────────────────────────────────

    def _load_presets(self):
        import json
        try:
            with open(self._presets_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_presets(self, presets):
        import json
        try:
            with open(self._presets_path, 'w', encoding='utf-8') as f:
                json.dump(presets, f, indent=2)
            return True
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Presets", f"Could not save presets:\n{e}")
            return False

    def _reload_presets(self):
        names = sorted(self._load_presets().keys())
        self.preset_menu['values'] = names
        if self.preset_var.get() not in names:
            self.preset_var.set('')

    def save_preset(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Save preset",
            "Name this preset:", parent=self)
        if not name:
            return
        name = name.strip()[:40]
        presets = self._load_presets()
        presets[name] = self.params()
        if self._write_presets(presets):
            self._reload_presets()
            self.preset_var.set(name)
            self.status_var.set(f'Preset "{name}" saved')

    def apply_preset(self):
        name = self.preset_var.get()
        if not name:
            return
        presets = self._load_presets()
        if name in presets:
            self._push_history()
            self._set_sliders(presets[name])
            self.schedule()
            self.status_var.set(f'Preset "{name}" applied')

    def delete_preset(self):
        name = self.preset_var.get()
        if not name:
            return
        from tkinter import messagebox
        if not messagebox.askyesno("Delete preset", f'Delete "{name}"?'):
            return
        presets = self._load_presets()
        presets.pop(name, None)
        if self._write_presets(presets):
            self._reload_presets()
            self.status_var.set(f'Preset "{name}" deleted')

    # ── Zoom / pan ────────────────────────────────────────────────────────────

    def _on_wheel(self, event, direction=None):
        if not self.orig_img:
            return
        if direction is None:
            direction = 1 if event.delta > 0 else -1
        old = self._zoom
        self._zoom = max(1.0, min(8.0, self._zoom * (1.15 if direction > 0 else 1/1.15)))
        if self._zoom == 1.0:
            self._pan_x = self._pan_y = 0
        if self._zoom != old:
            self.redraw()

    def _pan_start(self, event):
        if self._zoom > 1.0:
            self._drag_last = (event.x, event.y)

    def _pan_move(self, event):
        if self._drag_last and self._zoom > 1.0:
            dx = event.x - self._drag_last[0]
            dy = event.y - self._drag_last[1]
            self._pan_x += dx
            self._pan_y += dy
            self._drag_last = (event.x, event.y)
            self.redraw()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0
        self.redraw()

    # ── Auto-update ───────────────────────────────────────────────────────────

    def _check_updates(self):
        """Runs in a background thread; shows a banner if a newer version exists."""
        try:
            import urllib.request
            with urllib.request.urlopen(UPDATE_VERSION_URL, timeout=8) as r:
                latest = r.read().decode('utf-8').strip()
            if _parse_ver(latest) > _parse_ver(VERSION):
                self.after(0, lambda: self._show_update_banner(latest))
        except Exception:
            pass   # offline or URL not reachable — stay silent

    def _show_update_banner(self, latest):
        banner = tk.Frame(self, bg='#123d2e')
        banner.pack(fill='x', before=self.winfo_children()[0])
        tk.Label(banner,
            text=f"  Update available: v{latest}  (you have v{VERSION})",
            bg='#123d2e', fg='#7fd6b5', font=('Arial', 10)).pack(side='left', pady=6)
        btn = tk.Label(banner, text="Install and restart", bg=C['accent'],
                       fg='white', font=('Arial', 10), padx=12, pady=4, cursor='hand2')
        btn.pack(side='right', padx=8, pady=4)
        btn.bind('<Button-1>', lambda e: self._install_update(banner))
        dismiss = tk.Label(banner, text="x", bg='#123d2e', fg='#7fd6b5',
                           font=('Arial', 11), padx=8, cursor='hand2')
        dismiss.pack(side='right')
        dismiss.bind('<Button-1>', lambda e: banner.destroy())

    def _install_update(self, banner):
        banner.destroy()
        def do_update():
            try:
                import urllib.request
                script = os.path.abspath(__file__)
                with urllib.request.urlopen(UPDATE_FILE_URL, timeout=30) as r:
                    new_code = r.read()
                # Sanity: must be plausible Python and bigger than 10 KB
                if len(new_code) < 10000 or not new_code.lstrip().startswith(b'#!'):
                    raise ValueError("Downloaded file failed validation")
                with open(script, 'wb') as f:
                    f.write(new_code)
                self.after(0, self._restart)
            except Exception as e:
                from tkinter import messagebox
                self.after(0, lambda: messagebox.showerror(
                    "Update failed", f"Could not install the update:\n{e}"))
        threading.Thread(target=do_update, daemon=True).start()

    def _restart(self):
        import subprocess
        script = os.path.abspath(__file__)
        subprocess.Popen([sys.executable, script])
        self.destroy()

    def _on_drop(self, event):
        """Handle a file dropped onto the app window."""
        paths = self.tk.splitlist(event.data)
        if paths:
            self.open_file(paths[0])

    def _on_ai_toggle(self):
        if self.ai_var.get() and not _ai_available():
            self._prompt_ai_install()
        self._refresh_ai_hint()
        self._update_dims()

    def _refresh_ai_hint(self):
        if _ai_available():
            self.ai_hint.configure(text='ready', fg=C['accent'])
        else:
            self.ai_hint.configure(text='~255 MB one-time download', fg='#3a3a3a')

    def _prompt_ai_install(self):
        from tkinter import messagebox
        ok = messagebox.askyesno("Install AI packages",
            "AI upscaling needs these packages (one-time download):\n\n"
            "  PyTorch (CPU only)  ~180 MB\n"
            "  basicsr              ~  5 MB\n"
            "  Real-ESRGAN          ~  5 MB\n"
            "  Model weights        ~ 65 MB (downloaded on first use)\n\n"
            "Total: ~255 MB\n\nInstall now?")
        if ok:
            self._install_ai_deps()
        else:
            self.ai_var.set(False)

    def _install_ai_deps(self):
        """Install AI packages in a background thread with a progress window."""
        win = tk.Toplevel(self)
        win.title("Installing AI packages")
        win.geometry("440x150")
        win.resizable(False, False)
        win.configure(bg=C['bg'])
        win.grab_set()

        tk.Label(win, text="Installing AI upscaling packages...",
                 bg=C['bg'], fg=C['text'], font=('Arial', 12)).pack(pady=(18, 6))
        sv = tk.StringVar(value="Starting...")
        tk.Label(win, textvariable=sv, bg=C['bg'], fg=C['muted'],
                 font=('Arial', 10)).pack()
        tk.Label(win, text="This may take several minutes.",
                 bg=C['bg'], fg='#3a3a3a', font=('Arial', 9)).pack(pady=(4, 0))

        def do_install():
            import subprocess
            steps = [
                ("Downloading PyTorch (CPU)...",
                 [sys.executable, '-m', 'pip', 'install', 'torch',
                  '--index-url', 'https://download.pytorch.org/whl/cpu',
                  '--user', '-q']),
                ("Installing basicsr and Real-ESRGAN...",
                 [sys.executable, '-m', 'pip', 'install',
                  'basicsr', 'realesrgan', '--user', '-q']),
            ]
            for msg, cmd in steps:
                win.after(0, lambda m=msg: sv.set(m))
                r = subprocess.run(cmd, capture_output=True)
                if r.returncode != 0:
                    err = r.stderr.decode(errors='replace')[-600:]
                    win.after(0, lambda e=err: (
                        win.destroy(),
                        messagebox.showerror("Install failed",
                            f"A package failed to install.\n\n{e}")))
                    return
            win.after(0, lambda: (
                win.destroy(),
                self._refresh_ai_hint(),
                messagebox.showinfo("Done",
                    "AI packages installed!\n\n"
                    "The model (~65 MB) will download automatically on first use.")))

        threading.Thread(target=do_install, daemon=True).start()

    def _get_upscale_factor(self, img):
        """Return the scale factor needed to bring the long edge to the target."""
        val = self.upscale_var.get()
        if val == 'off':
            return 1.0
        long_edge = max(img.size)
        target    = self._RES[val]
        return max(1.0, target / long_edge)   # never downscale

    def _on_upscale_change(self):
        self._update_dims()

    def _update_dims(self):
        """Refresh the output-dimensions label under the upscale buttons."""
        if not self.proc_img:
            self.dims_var.set('')
            return
        val = self.upscale_var.get()
        w, h = self.proc_img.size
        if val == 'off':
            self.dims_var.set(f'{w} x {h} px')
            return
        factor = self._get_upscale_factor(self.proc_img)
        if factor == 1.0:
            label = '4K' if val == '4k' else '8K'
            self.dims_var.set(f'Already {label}+  ({w} x {h} px)')
        else:
            nw = round(w * factor);  nh = round(h * factor)
            label = '4K' if val == '4k' else '8K'
            ai = '  AI' if (self.ai_var.get() and _ai_available()) else ''
            self.dims_var.set(f'{w}x{h}  ->  {nw}x{nh} px  ({label}{ai})')

    def _on_slide(self, key):
        if self.orig_img:
            # Push a history snapshot at the start of a drag session:
            # if no reprocess is pending, this is a fresh interaction.
            if self._after_id is None:
                self._push_history()
            self.schedule()

    def schedule(self):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(90, self._run)

    def params(self):
        return {k: v.get() for k, v in self.svars.items()}

    def open_file(self, path=None):
        if path is None:
            img_exts = "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"
            if _HEIC_OK:
                img_exts += " *.heic *.heif"
            path = filedialog.askopenfilename(
                title="Open Photo",
                filetypes=[
                    ("Images", img_exts),
                    ("All files", "*.*"),
                ])
        if not path:
            return
        # Friendly message if a HEIC arrives but support isn't installed
        if path.lower().endswith(('.heic', '.heif')) and not _HEIC_OK:
            from tkinter import messagebox
            if messagebox.askyesno("HEIC support",
                    "This is an iPhone HEIC photo. Bare needs one extra\n"
                    "package to open it (a quick install).\n\nInstall now?"):
                import subprocess
                r = subprocess.run([sys.executable, '-m', 'pip', 'install',
                                    'pillow-heif', '--user', '-q'],
                                   capture_output=True)
                if r.returncode == 0:
                    messagebox.showinfo("Done",
                        "Installed! Please restart Bare, then open the photo again.")
                else:
                    messagebox.showerror("Install failed",
                        r.stderr.decode(errors='replace')[-400:])
            return
        # Reset zoom/undo for the new photo
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0
        self._history.clear()
        self.filepath = path
        try:
            img = Image.open(path).convert('RGB')
            try:    img = ImageOps.exif_transpose(img)
            except: pass
            img.thumbnail((2400, 2400), Image.LANCZOS)
            self.orig_img = img
            self.proc_img = None
            self.placeholder.place_forget()
            self._update_auto()
            self._run()
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Could not open image:\n{e}")

    def save_file(self):
        if not self.proc_img:
            return
        base = os.path.splitext(os.path.basename(self.filepath or 'photo'))[0]

        # Determine upscale factor
        val    = self.upscale_var.get()
        factor = self._get_upscale_factor(self.proc_img)
        use_ai = self.ai_var.get() and _ai_available() and factor > 1.0
        suffix = '_bare' if val == 'off' else f'_bare_{val.upper()}{"_AI" if use_ai else ""}'

        path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            initialfile=f"{base}{suffix}",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return

        # Apply upscale in a thread so the UI doesn't freeze on large factors
        self.status_var.set("Saving…")
        self.status_lbl.configure(fg=C['muted'])

        def do_save():
            img = self.proc_img
            if factor > 1.0:
                if use_ai:
                    self.after(0, lambda: self.status_var.set(
                        "AI upscaling... may take 1-2 min"))
                    result = ai_upscale(img, factor)
                    img = result if result is not None else upscale_image(img, factor)
                else:
                    img = upscale_image(img, factor)
            fmt = 'PNG' if path.lower().endswith('.png') else 'JPEG'
            kw  = {'quality': 95, 'subsampling': 0} if fmt == 'JPEG' else {}
            img.save(path, fmt, **kw)
            w, h = img.size
            name = os.path.basename(path)
            self.after(0, lambda: (
                self.status_var.set(f"✓  Saved  {name}\n   {w} × {h} px"),
                self.status_lbl.configure(fg=C['accent']),
                self.after(4000, lambda: self.status_lbl.configure(fg='#3a3a3a'))
            ))

        threading.Thread(target=do_save, daemon=True).start()

    def _update_auto(self):
        if not self.orig_img:
            return
        a = analyze(np.array(self.orig_img, dtype=np.float32))
        p = auto_params(a)
        for k, v in p.items():
            if k in self.svars:
                self.svars[k].set(v)

    def reset_auto(self):
        self._update_auto()
        self._run()

    def _run(self):
        self._after_id = None   # scheduled run has fired
        if not self.orig_img or self._busy:
            return
        self._busy = True
        self.status_var.set("Processing…")
        self.status_lbl.configure(fg=C['muted'])
        p = self.params()

        def worker():
            try:
                result, done = process(self.orig_img, p)
                self.proc_img = result
                self.after(0, lambda: self._done(done))
            except Exception as e:
                self.after(0, lambda: self.status_var.set(f"Error: {e}"))
                self._busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, done):
        self._busy = False
        self.redraw()
        self._update_dims()
        if done:
            lines = '\n'.join(f"✓  {CORRECTIONS[k]}" for k in done if k in CORRECTIONS)
            self.status_var.set(lines)
        else:
            self.status_var.set("No corrections applied")
        self.status_lbl.configure(fg='#3a3a3a')

    # ── Canvas rendering ──────────────────────────────────────────────────────

    def redraw(self):
        if not self.orig_img:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        pad  = 14
        half = (cw - pad * 3) // 2   # width for each image pane
        avail_h = ch - pad * 2

        self.canvas.delete('all')

        def render(img, pane_w, pane_h):
            """Fit or zoom+crop an image into a pane."""
            if self._zoom <= 1.0:
                return _fit(img, pane_w, pane_h)
            iw, ih = img.size
            base = min(pane_w / iw, pane_h / ih, 1.0)
            scale = base * self._zoom
            # visible source region
            vis_w = min(iw, int(pane_w / scale))
            vis_h = min(ih, int(pane_h / scale))
            cx = iw / 2 - self._pan_x / scale
            cy = ih / 2 - self._pan_y / scale
            x0 = max(0, min(iw - vis_w, int(cx - vis_w / 2)))
            y0 = max(0, min(ih - vis_h, int(cy - vis_h / 2)))
            crop = img.crop((x0, y0, x0 + vis_w, y0 + vis_h))
            return crop.resize(
                (max(1, int(vis_w * scale)), max(1, int(vis_h * scale))),
                Image.LANCZOS)

        # Left: original
        orig_fit = render(self.orig_img, half, avail_h)
        self.orig_tk = ImageTk.PhotoImage(orig_fit)
        x_orig = pad + half // 2
        self.canvas.create_image(x_orig, ch // 2, image=self.orig_tk, anchor='center')

        # Divider
        div_x = pad * 2 + half
        self.canvas.create_line(div_x, 0, div_x, ch, fill=C['accent'], width=1, dash=(4, 4))

        # Right: processed
        if self.proc_img:
            proc_fit = render(self.proc_img, half, avail_h)
            self.proc_tk = ImageTk.PhotoImage(proc_fit)
            x_proc = div_x + half // 2
            self.canvas.create_image(x_proc, ch // 2, image=self.proc_tk, anchor='center')

        # Corner labels
        self.canvas.create_text(pad + 6, pad, text="ORIGINAL", fill='#2a2a2a',
                                font=('Arial', 8, 'bold'), anchor='nw')
        self.canvas.create_text(div_x + 8, pad, text="ENHANCED", fill=C['accent'],
                                font=('Arial', 8, 'bold'), anchor='nw')

        # Zoom indicator
        if self._zoom > 1.0:
            self.canvas.create_text(cw - pad, ch - pad,
                text=f"{self._zoom:.1f}x  (double-click to reset)",
                fill='#3a3a3a', font=('Arial', 9), anchor='se')


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import traceback

    def _show_crash(message):
        """Best-effort error dialog — works even if the main app failed before
        any window existed. Falls back to stdout if no GUI is available."""
        try:
            r = tk.Tk(); r.withdraw()
            from tkinter import messagebox
            messagebox.showerror("Bare — failed to start", message)
            r.destroy()
        except Exception:
            print(message)

    try:
        app = App()
        app.mainloop()
    except Exception:
        tb = traceback.format_exc()
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'error_log.txt')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(tb)
        except Exception:
            log_path = None

        msg = "Bare ran into a problem and couldn't start.\n\n" + tb[-1500:]
        if log_path:
            msg += f"\n\nFull details saved to:\n{log_path}"
        _show_crash(msg)
