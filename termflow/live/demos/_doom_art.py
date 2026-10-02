"""Procedural art for the raycaster demo: wall textures, sprites, shading.

Everything is generated at startup from fixed seeds -- no asset files,
no licensing questions, same pixels every run. Textures are stored
column-major and pre-shaded per distance level so the renderer's inner
loop is a single list index.
"""

from __future__ import annotations

import math
import random

TEX = 32  #: texture edge length (power of two: masks instead of modulo)
SHADES = 16  #: distance fog levels
TRANSPARENT = -1

Texture = list[list[int]]  # [y][x] -> 0xRRGGBB (or TRANSPARENT)


def rgb(r: float, g: float, b: float) -> int:
    clamp = lambda v: max(0, min(255, int(v)))  # noqa: E731
    return clamp(r) << 16 | clamp(g) << 8 | clamp(b)


def scale(color: int, f: float) -> int:
    """Darken/brighten a packed color by factor ``f``."""
    return rgb((color >> 16 & 255) * f, (color >> 8 & 255) * f, (color & 255) * f)


def _brick(rng: random.Random) -> Texture:
    tex = []
    for y in range(TEX):
        row = []
        for x in range(TEX):
            offset = 8 if (y // 8) % 2 else 0
            mortar = y % 8 == 7 or (x + offset) % 16 == 15
            if mortar:
                row.append(rgb(90, 85, 80))
            else:
                n = rng.uniform(0.8, 1.15)
                row.append(rgb(140 * n, 52 * n, 34 * n))
        tex.append(row)
    return tex


def _tech(rng: random.Random) -> Texture:
    tex = []
    for y in range(TEX):
        row = []
        for x in range(TEX):
            n = rng.uniform(0.92, 1.05)
            c = rgb(98 * n, 100 * n, 110 * n)
            if x % 16 in (0, 15) or y % 16 in (0, 15):
                c = rgb(55, 56, 64)
            if x % 16 in (2, 13) and y % 16 in (2, 13):
                c = rgb(170, 170, 180)  # rivets
            if 14 <= y <= 17 and 4 <= x <= 27:
                glow = 0.7 + 0.3 * math.sin(x * 0.6)
                c = rgb(40 * glow, 140 * glow, 255 * glow)
            row.append(c)
        tex.append(row)
    return tex


def _stone(rng: random.Random) -> Texture:
    blocks = {(bx, by): rng.uniform(0.75, 1.1) for bx in range(8) for by in range(8)}
    tex = []
    for y in range(TEX):
        row = []
        for x in range(TEX):
            n = blocks[(x // 4, y // 4)] * rng.uniform(0.9, 1.08)
            row.append(rgb(78 * n, 96 * n, 72 * n))
        tex.append(row)
    return tex


def _hell(rng: random.Random) -> Texture:
    tex = []
    for y in range(TEX):
        row = []
        for x in range(TEX):
            vein = abs(math.sin(x * 0.45 + math.sin(y * 0.3) * 2.2))
            hot = max(0.0, 1 - vein * 4)
            n = rng.uniform(0.85, 1.1)
            row.append(rgb((70 + 185 * hot) * n, (12 + 110 * hot) * n, 10 * n))
        tex.append(row)
    return tex


def wall_textures() -> dict[str, Texture]:
    """Map character -> texture."""
    return {
        "1": _brick(random.Random(1)),
        "2": _tech(random.Random(2)),
        "3": _stone(random.Random(3)),
        "4": _hell(random.Random(4)),
    }


ShadedColumns = list[list[list[int]]]  # [level][x] -> column of TEX colors


def shade_columns(tex: Texture, side_factor: float) -> ShadedColumns:
    """Pre-shade a texture for every fog level, stored column-major."""
    levels = []
    for lvl in range(SHADES):
        f = max(0.1, 1 - lvl / SHADES) * side_factor
        levels.append([[scale(tex[y][x], f) for y in range(TEX)] for x in range(TEX)])
    return levels


def _ellipse(tex: Texture, cx: float, cy: float, rx: float, ry: float, color: int) -> None:
    for y in range(TEX):
        for x in range(TEX):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                # Cheap volume: lighter towards the upper left.
                light = 1.15 - 0.35 * ((x - cx + y - cy) / (rx + ry) + 0.5)
                tex[y][x] = scale(color, light)


def imp_sprites() -> dict[str, Texture]:
    """The demon: ``alive``, ``hurt`` (white flash), and ``dead`` frames."""
    alive: Texture = [[TRANSPARENT] * TEX for _ in range(TEX)]
    skin = rgb(150, 70, 40)
    _ellipse(alive, 8, 18, 3, 7, skin)  # arms
    _ellipse(alive, 24, 18, 3, 7, skin)
    _ellipse(alive, 16, 20, 8, 10, rgb(130, 55, 30))  # body
    _ellipse(alive, 16, 9, 6, 6, skin)  # head
    for hx in (11, 21):  # horns
        for dy in range(4):
            alive[3 - dy][hx + (dy if hx > 16 else -dy) // 2] = rgb(230, 220, 190)
    for ex in (13, 19):  # glowing eyes
        alive[8][ex] = alive[8][ex + 1] = rgb(255, 230, 40)
    for mx in range(13, 20):  # mouth
        alive[12][mx] = rgb(40, 0, 0)
    hurt = [[TRANSPARENT if c == TRANSPARENT else rgb(255, 230, 220) for c in row] for row in alive]
    dead: Texture = [[TRANSPARENT] * TEX for _ in range(TEX)]
    _ellipse(dead, 16, 29, 12, 3, rgb(120, 10, 10))
    _ellipse(dead, 14, 28, 5, 2, rgb(150, 70, 40))
    return {"alive": alive, "hurt": hurt, "dead": dead}
