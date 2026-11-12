"""Generates the artwork for this entry: a lighthouse on a dark cliff
whose lamp beam is a spiral galaxy.

Deterministic (fixed seed 37): running `python3 draw.py` always produces
the same art.png. Needs Pillow + numpy (only for generating; the entry
itself, main.py, is standard library only and ships the finished image
as base64 in art.b64).
"""
import base64
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2  # supersampling factor: draw at 2048, downscale to 1024
W = H = 1024
HERE = Path(__file__).resolve().parent


def vgrad(w, h, stops):
    """Vertical gradient. stops = [(pos0to1, (r,g,b)), ...]."""
    ys = np.linspace(0, 1, h * SS)
    cols = np.zeros((h * SS, 3), dtype=np.float32)
    pos = np.array([s[0] for s in stops])
    arr = np.array([s[1] for s in stops], dtype=np.float32)
    for ch in range(3):
        cols[:, ch] = np.interp(ys, pos, arr[:, ch])
    img = np.repeat(cols[:, None, :], w * SS, axis=1)
    return Image.fromarray(img.astype(np.uint8), "RGB")


def radial_glow(w, h, cx, cy, radius, color, peak=255):
    yy, xx = np.mgrid[0:h * SS, 0:w * SS]
    d = np.sqrt((xx - cx * SS) ** 2 + (yy - cy * SS) ** 2) / (radius * SS)
    a = np.clip(1 - d, 0, 1) ** 2 * peak
    layer = np.zeros((h * SS, w * SS, 4), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = color
    layer[..., 3] = a.astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def blob(w, h, cx, cy, rx, ry, color, alpha, blur=60):
    layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx * SS - rx * SS, cy * SS - ry * SS, cx * SS + rx * SS, cy * SS + ry * SS],
              fill=color + (alpha,))
    return layer.filter(ImageFilter.GaussianBlur(blur * SS))


def stars(draw, rng, w, h, n, y0=0, y1=1024, bright=255, colors=None):
    colors = colors or [(255, 255, 255)]
    for _ in range(n):
        x = rng.uniform(0, w)
        y = rng.uniform(y0, y1)
        r = rng.choice([0.7, 1.0, 1.0, 1.4, 1.8, 2.4])
        c = rng.choice(colors)
        a = rng.randint(140, bright)
        draw.ellipse([x * SS - r * SS, y * SS - r * SS, x * SS + r * SS, y * SS + r * SS],
                     fill=c + (a,))
        if r >= 2.0 and rng.random() < 0.4:
            L = r * 3.2
            draw.line([x * SS - L * SS, y * SS, x * SS + L * SS, y * SS], fill=c + (a // 2,), width=max(1, SS // 2))
            draw.line([x * SS, y * SS - L * SS, x * SS, y * SS + L * SS], fill=c + (a // 2,), width=max(1, SS // 2))


def poly(draw, points, **kw):
    draw.polygon([(x * SS, y * SS) for x, y in points], **kw)


def ell(draw, cx, cy, rx, ry, **kw):
    draw.ellipse([cx * SS - rx * SS, cy * SS - ry * SS, cx * SS + rx * SS, cy * SS + ry * SS], **kw)


def vignette(img, strength=70):
    yy, xx = np.mgrid[0:H * SS, 0:W * SS]
    dx = (xx - W * SS / 2) / (W * SS / 2)
    dy = (yy - H * SS / 2) / (H * SS / 2)
    d = np.clip(np.sqrt(dx ** 2 + dy ** 2) - 0.62, 0, 1) ** 2 * strength
    layer = np.zeros((H * SS, W * SS, 4), dtype=np.uint8)
    layer[..., 3] = d.astype(np.uint8)
    img.alpha_composite(Image.fromarray(layer, "RGBA"))
    return img


def base_sky(stops, nebulae, nstars=320, seed=7):
    img = vgrad(W, H, stops).convert("RGBA")
    rng = random.Random(seed)
    for (cx, cy, rx, ry, col, a, blur) in nebulae:
        img.alpha_composite(blob(W, H, cx, cy, rx, ry, col, a, blur))
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    stars(ImageDraw.Draw(layer), rng, W, H, nstars)
    img.alpha_composite(layer)
    return img


def draw_art():
    # Seed 37: the starfield, spiral dots and sea shimmer all derive from it.
    rng = random.Random(37)
    img = base_sky([(0.0, (8, 10, 34)), (0.5, (22, 18, 60)), (1.0, (56, 26, 88))],
                   [(700, 300, 320, 120, (150, 110, 230), 52, 70),
                    (250, 180, 240, 110, (90, 140, 220), 46, 80),
                    (850, 750, 260, 140, (200, 70, 160), 40, 90)],
                   nstars=320, seed=37)
    # milky way band (diagonal soft cloud)
    mw = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    md = ImageDraw.Draw(mw)
    for (x, y, r, a) in [(200, 200, 90, 34), (350, 300, 110, 30), (500, 400, 120, 28),
                         (650, 500, 100, 26), (800, 620, 90, 24), (420, 340, 70, 30)]:
        ell(md, x, y, r, r * 0.55, fill=(200, 190, 255, a))
    mw = mw.rotate(-24, resample=Image.BICUBIC, center=(512 * SS, 512 * SS))
    img.alpha_composite(mw.filter(ImageFilter.GaussianBlur(30 * SS)))
    # shooting star
    sh = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.line([820 * SS, 120 * SS, 700 * SS, 190 * SS], fill=(255, 255, 255, 230), width=3 * SS)
    sd.line([820 * SS, 120 * SS, 760 * SS, 156 * SS], fill=(255, 255, 255, 130), width=6 * SS)
    ell(sd, 820, 120, 4, 4, fill=(255, 255, 255, 255))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1 * SS)))
    # cliff
    cliff = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cliff)
    poly(cd, [(0, 1024), (0, 900), (140, 860), (300, 880), (430, 845), (560, 875),
              (700, 850), (860, 885), (1024, 860), (1024, 1024)], fill=(12, 14, 32, 255))
    poly(cd, [(140, 860), (300, 880), (430, 845), (430, 862), (300, 892), (140, 872)],
         fill=(40, 46, 84, 255))
    img.alpha_composite(cliff)
    # lighthouse tower (center x 430, base y 880, top y 430)
    tw = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    td = ImageDraw.Draw(tw)
    tower = [(385, 880), (405, 470), (455, 470), (475, 880)]
    poly(td, tower, fill=(238, 230, 214, 255))
    mask = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(mask).polygon([(x * SS, y * SS) for x, y in tower], fill=255)
    bands = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bands)
    for y0 in (560, 660, 760):
        bd.rectangle([380 * SS, y0 * SS, 480 * SS, (y0 + 46) * SS], fill=(172, 52, 62, 255))
    bands.putalpha(Image.composite(bands.split()[3], Image.new("L", bands.size, 0), mask))
    tw.alpha_composite(bands)
    td.rectangle([392 * SS, 448 * SS, 468 * SS, 472 * SS], fill=(30, 32, 56, 255))
    for x in range(398, 466, 12):
        td.line([x * SS, 448 * SS, x * SS, 470 * SS], fill=(120, 125, 160, 255), width=2 * SS)
    td.polygon([(398 * SS, 448 * SS), (462 * SS, 448 * SS), (446 * SS, 408 * SS), (414 * SS, 408 * SS)],
               fill=(40, 42, 72, 255))
    img.alpha_composite(radial_glow(W, H, 430, 428, 170, (255, 205, 110), 210))
    ell(td, 430, 428, 26, 26, fill=(255, 226, 140, 255))
    ell(td, 430, 428, 16, 16, fill=(255, 248, 210, 255))
    img.alpha_composite(tw)
    # galaxy beam: stronger cone + two-arm spiral with bright core
    beam = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    bdd = ImageDraw.Draw(beam)
    poly(bdd, [(452, 412), (452, 445), (1024, 570), (1024, 310)], fill=(255, 210, 130, 44))
    poly(bdd, [(452, 420), (452, 437), (1024, 520), (1024, 350)], fill=(255, 225, 160, 30))
    img.alpha_composite(beam)
    sp = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sp)
    ell(sd, 640, 428, 26, 16, fill=(255, 240, 200, 255))
    ell(sd, 640, 428, 14, 9, fill=(255, 255, 255, 255))
    for arm in (0, math.pi):
        for i in range(170):
            t = i / 170
            ang = arm + t * 4.6
            rr = 26 + t * 185
            cx = 640 + math.cos(ang) * rr * 1.25
            cy = 428 + math.sin(ang) * rr * 0.45
            r = max(1.2, 5.0 - t * 3.6)
            col = rng.choice([(255, 240, 200), (255, 210, 140), (210, 190, 255), (255, 255, 255)])
            ell(sd, cx, cy, r, r, fill=col + (rng.randint(180, 255),))
    sp = sp.filter(ImageFilter.GaussianBlur(1.1 * SS))
    img.alpha_composite(sp)
    # faint beam to the left
    beam2 = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    poly(ImageDraw.Draw(beam2), [(408, 415), (408, 442), (0, 520), (0, 350)], fill=(255, 210, 130, 18))
    img.alpha_composite(beam2)
    # sea shimmer
    sea = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    sdd = ImageDraw.Draw(sea)
    for y in range(920, 1010, 16):
        x0 = rng.uniform(60, 700)
        sdd.line([x0 * SS, y * SS, (x0 + rng.uniform(60, 200)) * SS, y * SS],
                 fill=(255, 210, 130, rng.randint(26, 60)), width=2 * SS)
    img.alpha_composite(sea)
    vignette(img)
    return img.convert("RGB").resize((1024, 1024), Image.LANCZOS)


if __name__ == "__main__":
    art = draw_art()
    art.save(HERE / "art.png", format="PNG", optimize=True)
    (HERE / "art.b64").write_text(
        base64.b64encode((HERE / "art.png").read_bytes()).decode() + "\n")
    print(f"wrote {HERE / 'art.png'} ({(HERE / 'art.png').stat().st_size} bytes)")
