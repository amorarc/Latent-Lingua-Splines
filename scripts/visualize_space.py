#!/usr/bin/env python3
"""
Interactive 3-D visualisation of the UMAP embedding manifold.

Loads the fitted projection from models/projection.pkl, extracts vocab
embeddings from the model, projects them to 3-D, and shows an interactive
scatter with hover labels, colour-mode controls, and regex search.

Projected coordinates are cached to outputs/coords_cache.npz so subsequent
runs skip the slow model-extraction step.

Usage:
    python scripts/visualize_space.py
    python scripts/visualize_space.py --config config/config.yaml
    python scripts/visualize_space.py --recompute   # ignore cache
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from mpl_toolkits.mplot3d import proj3d  # noqa: F401 – side-effect: registers 3-D projection
from scipy.spatial import cKDTree

from src import EmbeddingExtractor, LatentProjection, load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_PATH = Path("outputs/coords_cache.npz")
_LATIN_OR_DIGIT = re.compile(r"[a-zA-Z0-9]")
_WORD_START = re.compile(r"^[▁Ġ]")          # SentencePiece / BPE word-start markers


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _token_type(token: str) -> int:
    """Classify: 0 = subword fragment, 1 = full word, 2 = digit, 3 = symbol."""
    core = token.lstrip("▁Ġ")
    if not core:
        return 3
    if core.isdigit():
        return 2
    if core.isalpha():
        return 0 if _WORD_START.match(token) else 1
    return 3


# ---------------------------------------------------------------------------
# Data: load from cache or compute from model + projection
# ---------------------------------------------------------------------------

def _compute_and_cache(cfg: dict, projection_path: str) -> tuple[np.ndarray, list[str]]:
    extractor = EmbeddingExtractor(cfg["model"]["name"])
    all_vecs = extractor.full_vocab_embeddings()
    all_tokens = extractor.tokenizer.convert_ids_to_tokens(
        list(range(all_vecs.shape[0]))
    )

    mask = [bool(_LATIN_OR_DIGIT.search(t or "")) for t in all_tokens]
    vecs   = all_vecs[[i for i, keep in enumerate(mask) if keep]]
    tokens = [t for t, keep in zip(all_tokens, mask) if keep]
    print(f"  {len(tokens):,} tokens after Latin/digit filter")

    projection = LatentProjection.load(projection_path)
    print("Projecting embeddings …")
    coords = projection.transform(vecs)          # (N, 3)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, coords=coords, tokens=np.array(tokens))
    print(f"Cache saved → {CACHE_PATH}")
    return coords, tokens


def _load_data(
    cfg: dict, projection_path: str, recompute: bool
) -> tuple[np.ndarray, list[str]]:
    if not recompute and CACHE_PATH.exists():
        print(f"Loading cache from {CACHE_PATH} …")
        data = np.load(CACHE_PATH)
        coords = data["coords"]
        tokens = data["tokens"].tolist()
        print(f"  {len(tokens):,} tokens")
        return coords, tokens
    return _compute_and_cache(cfg, projection_path)


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------

def _build_color_arrays(coords: np.ndarray, tokens: list[str]) -> dict:
    az = np.arctan2(coords[:, 1], coords[:, 0])
    el = np.arctan2(coords[:, 2], np.hypot(coords[:, 0], coords[:, 1]))
    lengths = np.array([len(t.lstrip("▁Ġ")) for t in tokens], dtype=float)
    types   = np.array([_token_type(t) for t in tokens], dtype=float)

    return {
        "Azimuth":   ((az + np.pi) / (2 * np.pi),   "hsv",     (-180, 180,  "Azimuthal angle [°]")),
        "Elevation": ((el + np.pi / 2) / np.pi,      "plasma",  (-90,  90,   "Elevation angle [°]")),
        "Length":    (lengths / max(lengths.max(), 1), "viridis", (0, lengths.max(), "Token length (chars)")),
        "Type":      (types / 3.0,                    "tab10",   (0, 3, "0=frag  1=word  2=digit  3=sym")),
    }


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class EmbeddingViewer:
    _HOVER_PX  = 25      # pixel radius for hover hit-test
    _HOVER_DT  = 0.04    # seconds between hover updates

    def __init__(self, coords: np.ndarray, tokens: list[str]) -> None:
        self.coords     = coords
        self.tokens     = tokens
        self.color_data = _build_color_arrays(coords, tokens)
        self.color_mode = "Azimuth"

        self._search_mask: np.ndarray | None = None
        self._sc_hl = None
        self._last_hover = 0.0
        self._hover_idx  = -1

        # Cached 2-D screen projection — rebuilt only when the view changes.
        self._proj_key:   bytes | None          = None
        self._proj_cache: np.ndarray | None     = None
        self._kd_cache:   cKDTree | None        = None

        self._build_figure()
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor("#0f0f1e")

        # --- 3-D axes ---
        self.ax = self.fig.add_axes([0.01, 0.02, 0.67, 0.94], projection="3d")
        self.ax.set_facecolor("#0a0a14")
        self.ax.set_title(
            f"Embedding Manifold — {len(self.tokens):,} tokens",
            color="white", fontsize=12, pad=10,
        )
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.fill  = False
            axis.pane.set_edgecolor("#333355")
        self.ax.set_xlabel("UMAP 1", fontsize=7, color="#888899")
        self.ax.set_ylabel("UMAP 2", fontsize=7, color="#888899")
        self.ax.set_zlabel("UMAP 3", fontsize=7, color="#888899")
        self.ax.tick_params(colors="#555566", labelsize=5)

        # initial scatter
        c_vals, cmap_name, _ = self.color_data[self.color_mode]
        self._sc = self.ax.scatter(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
            c=c_vals, cmap=cmap_name, vmin=0, vmax=1,
            s=2, alpha=0.5, linewidths=0, depthshade=True,
        )

        # hover annotation
        self._annot = self.ax.text2D(
            0.02, 0.02, "", transform=self.ax.transAxes,
            fontsize=8, color="white", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="#0f0f2a", ec="#4fc3f7", alpha=0.9),
            visible=False, zorder=10,
        )

        # --- Right-side panel ---

        # Token info box (top)
        ax_info = self.fig.add_axes([0.70, 0.76, 0.29, 0.20])
        ax_info.set_facecolor("#12122a")
        ax_info.axis("off")
        self._info_title = ax_info.text(
            0.04, 0.95, "Hover over a point", va="top", ha="left",
            fontsize=9, color="#aaaacc", transform=ax_info.transAxes,
            fontfamily="monospace",
        )

        # Color mode radio
        ax_radio = self.fig.add_axes([0.70, 0.47, 0.29, 0.27], facecolor="#12122a")
        self._radio = mwidgets.RadioButtons(
            ax_radio, list(self.color_data.keys()), activecolor="#4fc3f7",
        )
        for lbl in self._radio.labels:
            lbl.set_color("#ddddee")
            lbl.set_fontsize(9)
        ax_radio.set_title("Colour by", color="#888899", fontsize=8, loc="left", pad=6)
        self._radio.on_clicked(self._on_color_change)

        # Regex search label
        ax_slbl = self.fig.add_axes([0.70, 0.41, 0.29, 0.04])
        ax_slbl.axis("off")
        ax_slbl.text(0.02, 0.5, "Highlight tokens (regex):", va="center",
                     color="#888899", fontsize=8, transform=ax_slbl.transAxes)

        # TextBox
        ax_tb = self.fig.add_axes([0.70, 0.34, 0.29, 0.06], facecolor="#1a1a32")
        self._textbox = mwidgets.TextBox(
            ax_tb, "", initial="",
            color="#1a1a32", hovercolor="#222244", label_pad=0.01,
        )
        self._textbox.text_disp.set_color("white")
        self._textbox.text_disp.set_fontsize(9)
        self._textbox.on_submit(self._on_search)

        # Match count label
        ax_cnt = self.fig.add_axes([0.70, 0.29, 0.29, 0.04])
        ax_cnt.axis("off")
        self._count_text = ax_cnt.text(
            0.02, 0.5, "", va="center", color="#4fc3f7",
            fontsize=8, transform=ax_cnt.transAxes,
        )

        # Clear button
        ax_btn = self.fig.add_axes([0.70, 0.22, 0.29, 0.06])
        self._btn = mwidgets.Button(
            ax_btn, "Clear highlight",
            color="#1a1a32", hovercolor="#2a2a55",
        )
        self._btn.label.set_color("#ddddee")
        self._btn.label.set_fontsize(8)
        self._btn.on_clicked(self._on_clear)

        # Usage hint
        ax_hint = self.fig.add_axes([0.70, 0.02, 0.29, 0.18])
        ax_hint.axis("off")
        hint = (
            "Controls\n"
            "─────────────────\n"
            "Rotate : left-drag\n"
            "Zoom   : scroll\n"
            "Hover  : nearest token\n"
            "Search : Enter to apply\n"
        )
        ax_hint.text(
            0.04, 0.95, hint, va="top", ha="left",
            fontsize=7.5, color="#555577", transform=ax_hint.transAxes,
            fontfamily="monospace", linespacing=1.6,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_color_change(self, label: str) -> None:
        self.color_mode = label
        c_vals, cmap_name, _ = self.color_data[label]
        self._sc.set_array(c_vals)
        self._sc.set_cmap(plt.get_cmap(cmap_name))
        self._sc.set_clim(0, 1)
        self.fig.canvas.draw_idle()

    def _on_search(self, text: str) -> None:
        text = text.strip()
        if not text:
            self._clear_highlight()
            return
        try:
            pattern = re.compile(text, re.IGNORECASE)
            self._search_mask = np.array(
                [bool(pattern.search(t)) for t in self.tokens], dtype=bool
            )
            n = int(self._search_mask.sum())
            self._count_text.set_text(f"{n:,} match{'es' if n != 1 else ''}")
        except re.error as exc:
            self._count_text.set_text(f"bad regex: {exc}")
            self._search_mask = None
        self._redraw_highlight()

    def _on_clear(self, _event) -> None:
        self._textbox.set_val("")
        self._clear_highlight()

    def _clear_highlight(self) -> None:
        self._search_mask = None
        self._count_text.set_text("")
        self._redraw_highlight()

    def _redraw_highlight(self) -> None:
        if self._sc_hl is not None:
            self._sc_hl.remove()
            self._sc_hl = None

        if self._search_mask is not None and self._search_mask.any():
            hl = self.coords[self._search_mask]
            self._sc_hl = self.ax.scatter(
                hl[:, 0], hl[:, 1], hl[:, 2],
                c="tomato", s=14, alpha=0.95, linewidths=0,
                depthshade=False, zorder=5,
            )
        self.fig.canvas.draw_idle()

    def _screen_proj(self) -> tuple[np.ndarray, cKDTree]:
        """Return (pts_px, KDTree) in display-pixel space.

        Recomputes only when the 3-D view or figure size changes; all hover
        events between rotations reuse the cached tree → O(log N) per tick.
        """
        M      = self.ax.get_proj()
        fig_sz = (self.fig.get_figwidth(), self.fig.get_figheight())
        key    = M.tobytes() + bytes(str(fig_sz), "ascii")

        if key != self._proj_key:
            x2, y2, _ = proj3d.proj_transform(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], M
            )
            pts = self.ax.transData.transform(np.column_stack([x2, y2]))
            self._proj_cache = pts
            self._kd_cache   = cKDTree(pts)
            self._proj_key   = key

        return self._proj_cache, self._kd_cache

    def _on_hover(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        now = time.monotonic()
        if now - self._last_hover < self._HOVER_DT:
            return
        self._last_hover = now

        try:
            _, kd = self._screen_proj()
            dist, idx = kd.query([event.x, event.y])

            if dist > self._HOVER_PX:
                if self._hover_idx != -1:
                    self._annot.set_visible(False)
                    self._hover_idx = -1
                    self.fig.canvas.draw_idle()
                return

            if idx == self._hover_idx:
                return
            self._hover_idx = idx

            token = self.tokens[idx]
            c     = self.coords[idx]
            ttype = ("frag", "word", "digit", "sym")[_token_type(token)]

            self._info_title.set_text(
                f'"{token}"\n'
                f"type : {ttype}\n"
                f"len  : {len(token.lstrip('▁Ġ'))}\n"
                f"xyz  : ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
            )
            self._info_title.set_color("white")

            pos = self.ax.get_position()
            fw  = self.fig.get_figwidth()  * self.fig.dpi
            fh  = self.fig.get_figheight() * self.fig.dpi
            xf  = float(np.clip((event.x / fw - pos.x0) / pos.width  + 0.03, 0.01, 0.82))
            yf  = float(np.clip((event.y / fh - pos.y0) / pos.height + 0.03, 0.01, 0.96))
            self._annot.set_position((xf, yf))
            self._annot.set_text(token)
            self._annot.set_visible(True)
            self.fig.canvas.draw_idle()

        except Exception:
            pass

    # ------------------------------------------------------------------

    def show(self) -> None:
        plt.tight_layout(rect=[0, 0, 0.69, 1])
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive UMAP embedding viewer")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--recompute", action="store_true",
                        help="Re-extract embeddings even if coords_cache.npz exists")
    args = parser.parse_args()

    cfg             = load_config(args.config)
    projection_path = cfg.get("projection", {}).get("output_path", "models/projection.pkl")

    coords, tokens = _load_data(cfg, projection_path, args.recompute)
    print(f"Launching viewer ({len(tokens):,} tokens) …")

    viewer = EmbeddingViewer(coords, tokens)
    viewer.show()


if __name__ == "__main__":
    main()
