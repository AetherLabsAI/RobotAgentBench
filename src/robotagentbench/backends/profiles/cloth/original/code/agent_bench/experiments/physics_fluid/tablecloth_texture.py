"""Procedural old/new cloth appearance; dirt follows the cloth's material UVs."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def cloth_texture(*, dirty=False, seed=8, resolution=1024):
    rng = np.random.default_rng(seed)
    pixels = np.clip(np.array([231, 226, 210])[None, None, :]
                     + rng.normal(0, 1.2, (resolution, resolution, 1)), 0, 255)
    if dirty:
        yy, xx = np.mgrid[0:resolution, 0:resolution] / resolution
        mask = np.zeros((resolution, resolution))
        # An irregular central food/grease smear and a smaller faded spill.
        for cx, cy, sx, sy, strength in [(.51, .55, .063, .085, .62),
                                         (.57, .50, .045, .060, .52),
                                         (.35, .33, .046, .029, .25)]:
            for _ in range(13):
                dx, dy = rng.normal(0, [.027, .031])
                blob = np.exp(-(((xx - cx - dx) / (sx * rng.uniform(.25, .7))) ** 2
                                + ((yy - cy - dy) / (sy * rng.uniform(.25, .7))) ** 2) * 2)
                mask = np.maximum(mask, blob * strength * rng.uniform(.55, 1.))
        coarse = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        coarse = np.asarray(coarse.resize((resolution, resolution)).filter(ImageFilter.GaussianBlur(2))) / 255.
        mask *= .55 + .9 * coarse
        # Narrow wipe streaks and specks make the stain recognizable at head-view scale.
        for _ in range(38):
            cx, cy = rng.normal([.53, .53], [.045, .056])
            mask = np.maximum(mask, np.exp(-(((xx - cx) / .003) ** 2
                                             + ((yy - cy) / rng.uniform(.006, .024)) ** 2) * 2)
                              * rng.uniform(.25, .8))
        mask = np.clip(mask, 0, .85)[..., None]
        pixels = pixels * (1 - mask) + np.array([65, 49, 33])[None, None, :] * mask
    image = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for inset, width in ((74, 3), (85, 2)):
        inset = round(inset * resolution / 1024)
        draw.rectangle((inset, inset, resolution - 1 - inset, resolution - 1 - inset),
                       outline=(136, 125, 111), width=width)
    return image
