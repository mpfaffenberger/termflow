"""termflow DOOM-ish: a raycaster in a terminal window, next to live markdown.

Run it::

    tf --doom
    python -m termflow.live.demos.doom

Pure Python, no assets, no dependencies: a DDA raycaster with textured
walls and distance fog, billboard demons with a z-buffer, hitscan
shooting, and a minimap -- rendered as half-block truecolor pixels in
one window while termflow streams the mission briefing into another.

The point is the engine, not the game: anything that can fill an RGB
framebuffer (see :class:`termflow.live.FramebufferView` for engines
that produce their own frames, e.g. a doomgeneric port) gets the same
windowing, diffed rendering, and true key-up input on Windows Terminal.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from termflow.live.app import LiveApp, Window, hsplit, vsplit
from termflow.live.demos._doom_art import (
    SHADES,
    TEX,
    TRANSPARENT,
    imp_sprites,
    rgb,
    scale,
    shade_columns,
    wall_textures,
)
from termflow.live.input import ALT, CTRL, SHIFT
from termflow.live.widgets import MarkdownView, PixelWidget, TextLog
from termflow.render.document import render_markdown_lines
from termflow.tui.keys import Key

if TYPE_CHECKING:
    from collections.abc import Callable

    from termflow.live.buffer import Rect
    from termflow.live.input import KeyEvent, KeyState
    from termflow.live.pixels import PixelSurface

LEVEL = (
    "1111111111111111111111",
    "1P.....1.......2.....1",
    "1......1..D....2..D..1",
    "1..33..1.......2.....1",
    "1..33..........222.222",
    "1......1.............2",
    "1111.111...D.........2",
    "1..........4444..2...2",
    "1..D.......4..4..2.D.2",
    "1..........4..4..2...2",
    "1...3......4D.4......2",
    "1...3..........D.....2",
    "1111111111111112222222",
)

FORWARD, BACK = ("w", Key.UP), ("s", Key.DOWN)
TURN_LEFT, TURN_RIGHT = ("a", Key.LEFT), ("d", Key.RIGHT)
STRAFE_LEFT, STRAFE_RIGHT = ("q", ","), ("e", ".")
FIRE = (" ", CTRL, "f")
RUN = (SHIFT, ALT)

MOVE_SPEED = 3.0  # cells / second
TURN_SPEED = 2.4  # radians / second
FIRE_COOLDOWN = 0.32
DEMON_SPEED = 1.1
DEMON_HP = 2
PLAYER_HP = 100
RADIUS = 0.25  # collision radius

CEILING = rgb(58, 52, 50)
FLOOR = rgb(74, 62, 48)


@dataclass
class Demon:
    x: float
    y: float
    hp: int = DEMON_HP
    hurt: float = 0.0  # seconds of hit flash left
    cooldown: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Hit:
    perp: float  # perpendicular distance (no fisheye)
    side: int  # 0: hit an x-side, 1: a y-side
    cell: str
    wall_x: float  # where along the wall face, 0..1


class RaycasterGame(PixelWidget):
    """The game: state, simulation, and rendering into a pixel surface."""

    def __init__(self, on_event: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.on_event = on_event or (lambda _msg: None)
        self.grid = [list(row) for row in LEVEL]
        textures = wall_textures()
        self._columns = {
            (cell, side): shade_columns(tex, 1.0 if side == 0 else 0.72)
            for cell, tex in textures.items()
            for side in (0, 1)
        }
        self._sprites = imp_sprites()
        self._background: tuple[int, int, list[int]] | None = None
        self.show_map = True
        self.reset()

    # -- state ---------------------------------------------------------

    def reset(self) -> None:
        self.demons: list[Demon] = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "P":
                    self.px, self.py = x + 0.5, y + 0.5
                elif cell == "D":
                    self.demons.append(Demon(x + 0.5, y + 0.5))
        self.angle = 0.0
        self.hp = PLAYER_HP
        self.kills = 0
        self.fire_cooldown = 0.0
        self.flash = 0.0  # muzzle flash timer
        self.pain = 0.0  # damage vignette timer
        self.on_event("**Mission start.** Clear the demons.")

    @property
    def dead(self) -> bool:
        return self.hp <= 0

    @property
    def cleared(self) -> bool:
        return not any(d.alive for d in self.demons)

    def solid(self, x: float, y: float) -> bool:
        gx, gy = int(x), int(y)
        if not (0 <= gy < len(self.grid) and 0 <= gx < len(self.grid[gy])):
            return True
        return self.grid[gy][gx] in "1234"

    def _move(self, dx: float, dy: float) -> None:
        """Slide along walls: resolve each axis separately."""
        if not self.solid(self.px + dx + math.copysign(RADIUS, dx), self.py):
            self.px += dx
        if not self.solid(self.px, self.py + dy + math.copysign(RADIUS, dy)):
            self.py += dy

    # -- simulation ----------------------------------------------------

    def on_key(self, event: KeyEvent) -> bool:
        if event.key == "m":
            self.show_map = not self.show_map
            return True
        if event.key == "r" and (self.dead or self.cleared):
            self.reset()
            return True
        return False

    def update(self, dt: float, keys: KeyState) -> None:
        self.flash = max(0.0, self.flash - dt)
        self.pain = max(0.0, self.pain - dt)
        self.fire_cooldown = max(0.0, self.fire_cooldown - dt)
        if self.dead or self.cleared:
            return
        speed = MOVE_SPEED * (1.8 if keys.is_down(*RUN) else 1.0) * dt
        self.angle += keys.axis(TURN_LEFT, TURN_RIGHT) * TURN_SPEED * dt
        dx, dy = math.cos(self.angle), math.sin(self.angle)
        fwd = keys.axis(BACK, FORWARD)
        strafe = keys.axis(STRAFE_LEFT, STRAFE_RIGHT)
        self._move((dx * fwd - dy * strafe) * speed, (dy * fwd + dx * strafe) * speed)
        if keys.is_down(*FIRE) and self.fire_cooldown == 0:
            self._fire()
        for demon in self.demons:
            self._think(demon, dt)

    def _fire(self) -> None:
        self.fire_cooldown = FIRE_COOLDOWN
        self.flash = 0.08
        wall = self.cast(math.cos(self.angle), math.sin(self.angle)).perp
        target = None
        for demon in self.demons:
            if not demon.alive:
                continue
            depth, lateral = self._to_camera(demon.x - self.px, demon.y - self.py)
            if 0 < depth < wall and abs(lateral) < 0.35 and (target is None or depth < target[0]):
                target = (depth, demon)
        if target is None:
            return
        demon = target[1]
        demon.hp -= 1
        demon.hurt = 0.12
        if not demon.alive:
            self.kills += 1
            self.on_event(f"Demon down! **{self.kills}/{len(self.demons)}**")
            if self.cleared:
                self.on_event("**Level clear.** Press `r` to replay.")

    def _think(self, demon: Demon, dt: float) -> None:
        demon.hurt = max(0.0, demon.hurt - dt)
        demon.cooldown = max(0.0, demon.cooldown - dt)
        if not demon.alive:
            return
        vx, vy = self.px - demon.x, self.py - demon.y
        dist = math.hypot(vx, vy)
        if dist < 0.9:
            if demon.cooldown == 0:
                demon.cooldown = 0.9
                self.hp = max(0, self.hp - 9)
                self.pain = 0.25
                if self.dead:
                    self.on_event("**You died.** Press `r` to restart.")
        elif dist < 9:
            step = DEMON_SPEED * dt / dist
            nx, ny = demon.x + vx * step, demon.y + vy * step
            if not self.solid(nx, demon.y):
                demon.x = nx
            if not self.solid(demon.x, ny):
                demon.y = ny

    # -- rendering -----------------------------------------------------

    def cast(self, rdx: float, rdy: float) -> Hit:
        """DDA through the grid from the player along (rdx, rdy)."""
        mx, my = int(self.px), int(self.py)
        ddx = abs(1 / rdx) if rdx else 1e30
        ddy = abs(1 / rdy) if rdy else 1e30
        sx, sdx = (-1, (self.px - mx) * ddx) if rdx < 0 else (1, (mx + 1 - self.px) * ddx)
        sy, sdy = (-1, (self.py - my) * ddy) if rdy < 0 else (1, (my + 1 - self.py) * ddy)
        grid = self.grid
        for _ in range(64):
            if sdx < sdy:
                sdx += ddx
                mx += sx
                side = 0
            else:
                sdy += ddy
                my += sy
                side = 1
            cell = grid[my][mx] if 0 <= my < len(grid) and 0 <= mx < len(grid[my]) else "1"
            if cell in "1234":
                break
        perp = max(1e-4, (sdx - ddx) if side == 0 else (sdy - ddy))
        wall_x = (self.py + perp * rdy) if side == 0 else (self.px + perp * rdx)
        return Hit(perp, side, cell, wall_x - math.floor(wall_x))

    def _camera(self, width: int, height: int) -> tuple[float, float, float, float, float]:
        """Direction, camera plane, and projection scale for this viewport."""
        plane_len = max(0.66, 0.42 * width / max(1, height))  # widen FOV for wide panes
        dx, dy = math.cos(self.angle), math.sin(self.angle)
        return dx, dy, -dy * plane_len, dx * plane_len, width / (2 * plane_len)

    def _to_camera(self, rx: float, ry: float) -> tuple[float, float]:
        """World offset -> (depth, lateral offset) relative to the view ray."""
        dx, dy = math.cos(self.angle), math.sin(self.angle)
        return rx * dx + ry * dy, -rx * dy + ry * dx

    def _sky_and_floor(self, w: int, h: int, proj: float) -> list[int]:
        """Fogged ceiling/floor gradient, cached per resolution."""
        if self._background is None or self._background[:2] != (w, h):
            rows = []
            for y in range(h):
                dist = proj / max(1.0, abs(2 * y - h + 1))
                fog = max(0.1, 1 - min(SHADES - 1, dist * 1.6) / SHADES)
                rows += [scale(CEILING if y < h / 2 else FLOOR, fog)] * w
            self._background = (w, h, rows)
        return self._background[2]

    def render(self, surface: PixelSurface) -> None:
        w, h = surface.width, surface.height
        px = surface.pixels
        dx, dy, plx, ply, proj = self._camera(w, h)
        px[:] = self._sky_and_floor(w, h, proj)
        zbuf = [0.0] * w
        half = h // 2
        for x in range(w):
            cam = 2 * x / w - 1
            hit = self.cast(dx + plx * cam, dy + ply * cam)
            zbuf[x] = hit.perp
            line = max(1, int(proj / hit.perp))
            top = half - line // 2
            start, end = max(0, top), min(h, top + line)
            level = min(SHADES - 1, int(hit.perp * 1.6))
            tx = int(hit.wall_x * TEX) & (TEX - 1)
            column = self._columns[(hit.cell, hit.side)][level][tx]
            px[start * w + x : end * w + x : w] = [
                column[((y - top) * TEX) // line] for y in range(start, end)
            ]
        self._draw_demons(surface, zbuf, proj)
        self._draw_hud(surface)

    def _draw_demons(self, surface: PixelSurface, zbuf: list[float], proj: float) -> None:
        w, h = surface.width, surface.height
        visible = []
        for demon in self.demons:
            depth, lateral = self._to_camera(demon.x - self.px, demon.y - self.py)
            if depth > 0.2:
                visible.append((depth, lateral, demon))
        plane_len = w / (2 * proj)
        for depth, lateral, demon in sorted(visible, key=lambda v: v[0], reverse=True):
            size = int(proj / depth)
            if size < 2:
                continue
            cx = int(w / 2 * (1 + lateral / (depth * plane_len)))
            bottom = h // 2 + size // 2
            top, left = bottom - size, cx - size // 2
            frame = self._sprites["dead" if not demon.alive else "hurt" if demon.hurt else "alive"]
            level = min(SHADES - 1, int(depth * 1.6))
            fog = max(0.1, 1 - level / SHADES)
            for x in range(max(0, left), min(w, left + size)):
                if depth >= zbuf[x]:
                    continue
                tx = ((x - left) * TEX) // size
                for y in range(max(0, top), min(h, bottom)):
                    color = frame[((y - top) * TEX) // size][tx]
                    if color != TRANSPARENT:
                        surface.pixels[y * w + x] = scale(color, fog)

    def _draw_hud(self, surface: PixelSurface) -> None:
        w, h = surface.width, surface.height
        cx, cy = w // 2, h // 2
        for d in (-2, -1, 1, 2):  # crosshair
            surface.set(cx + d, cy, rgb(255, 255, 255))
            surface.set(cx, cy + d, rgb(255, 255, 255))
        # The gun: a stubby barrel rising from the bottom edge.
        gw, gh = max(4, w // 14), max(6, h // 4)
        kick = gh // 6 if self.flash else 0
        gx, gy = cx - gw // 2, h - gh + kick
        surface.fill_rect(gx - gw // 2, gy + gh // 2, gw * 2, gh, rgb(70, 50, 40))  # hands
        surface.fill_rect(gx, gy, gw, gh, rgb(95, 95, 105))
        surface.fill_rect(gx + gw // 3, gy, max(1, gw // 3), gh, rgb(45, 45, 52))
        if self.flash:
            r = max(2, gw // 2)
            surface.fill_rect(cx - r, gy - r * 2, r * 2, r * 2, rgb(255, 210, 90))
            surface.fill_rect(cx - r // 2, gy - r * 2 - r // 2, r, r * 2, rgb(255, 250, 200))
        if self.pain or self.dead:  # red border vignette
            t = max(2, h // 12)
            red = rgb(200, 10, 10)
            for rect in ((0, 0, w, t), (0, h - t, w, t), (0, 0, t, h), (w - t, 0, t, h)):
                surface.fill_rect(*rect, red)
        if self.show_map:
            self._draw_minimap(surface)

    def _draw_minimap(self, surface: PixelSurface) -> None:
        cell = 2 if surface.width >= 100 else 1
        colors = {
            "1": rgb(160, 70, 50),
            "2": rgb(90, 110, 170),
            "3": rgb(90, 120, 80),
            "4": rgb(220, 90, 20),
        }
        for y, row in enumerate(self.grid):
            for x, c in enumerate(row):
                color = colors.get(c, rgb(20, 20, 24))
                surface.fill_rect(1 + x * cell, 1 + y * cell, cell, cell, color)
        for demon in self.demons:
            if demon.alive:
                surface.fill_rect(
                    1 + int(demon.x * cell), 1 + int(demon.y * cell), cell, cell, rgb(255, 60, 60)
                )
        ppx, ppy = 1 + int(self.px * cell), 1 + int(self.py * cell)
        surface.fill_rect(ppx, ppy, cell, cell, rgb(80, 255, 80))
        tip = (
            ppx + int(math.cos(self.angle) * 3 * cell),
            ppy + int(math.sin(self.angle) * 3 * cell),
        )
        surface.set(*tip, rgb(255, 255, 255))


BRIEFING = """\
# Mission briefing

You are a **terminal**. Somewhere in this grid, *demons* are rendering
themselves at 30 frames per second, and frankly that is too many.

## Controls

| Key | Action |
| --- | --- |
| `W` `S` / arrows | move |
| `A` `D` / arrows | turn |
| `Q` `E` | strafe |
| `Space` / `Ctrl` / `F` | fire |
| `Shift` | run |
| `M` | toggle map |
| `Tab` | focus next window |
| `Ctrl+Q` | quit |

## How this works

> Every pixel is half of a `▀` cell with truecolor fg/bg. Only cells
> that changed since the last frame are sent to the terminal.

```python
app = LiveApp(windows, layout=...)
app.run()
```

Good luck. Don't let them reach the **status bar**.
"""


def build_app(fps: float = 30.0) -> LiveApp:
    """The demo app: game, streaming briefing, and an event log."""
    log = TextLog()

    def announce(markdown: str) -> None:
        for line in render_markdown_lines(markdown, 200):
            log.write(line + "\n")

    game = RaycasterGame(on_event=announce)
    windows = [
        Window("DOOM-ish", game),
        Window("Briefing", MarkdownView(BRIEFING, reveal_rate=240)),
        Window("Log", log),
    ]

    def layout(area: Rect) -> list[Rect]:
        if area.width < 100:
            return [area]  # small terminal: the game gets everything
        left, right = hsplit(area, 5, 2)
        return [left, *vsplit(right, 3, 1)]

    app = LiveApp(windows, layout, title="termflow DOOM-ish", fps=fps)

    def status() -> str:
        held = "native key-up" if app.console.reports_release else "autorepeat"
        return f"HP {game.hp} · Kills {game.kills}/{len(game.demons)} · input: {held}"

    app.status = status
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="termflow live-mode raycaster demo")
    parser.add_argument("--fps", type=float, default=30.0, help="target frame rate")
    parser.add_argument("--frames", type=int, default=None, help="exit after N frames (benchmark)")
    args = parser.parse_args(argv)
    stats = build_app(fps=args.fps).run(max_frames=args.frames)
    print(
        f"{stats.frames} frames in {stats.seconds:.1f}s ({stats.fps:.1f} fps), "
        f"{stats.bytes_written / max(1, stats.frames) / 1024:.1f} KiB/frame"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
