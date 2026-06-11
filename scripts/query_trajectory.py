#!/usr/bin/env python3
"""
Visualise an LLM response as a trajectory through the latent embedding space.

Usage:
    python scripts/query_trajectory.py "What is gravity?"
    python scripts/query_trajectory.py --auto "What is gravity?"         # auto-play
    python scripts/query_trajectory.py --add-hidden-states "What is gravity?"
    python scripts/query_trajectory.py --follow "Describe the ocean"
    python scripts/query_trajectory.py --ms-per-token 200 "What is love?"
    python scripts/query_trajectory.py               # interactive prompt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import splprep, splev
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src import LatentProjection, load_config


_STEPS_PER_SEG  = 162  # spline sub-steps between consecutive token positions
_N_TRAIL_CHUNKS = 8   # number of gradient colour bands in the trail
_TEXT_MAX_LINES = 20  # visible lines in the response panel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_embedding_layer(model: torch.nn.Module) -> torch.nn.Embedding:
    candidates = [
        lambda m: m.model.embed_tokens,
        lambda m: m.transformer.wte,
        lambda m: m.embed_tokens,
    ]
    for fn in candidates:
        try:
            layer = fn(model)
            if isinstance(layer, torch.nn.Embedding):
                return layer
        except AttributeError:
            continue
    all_emb = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Embedding)
    ]
    if all_emb:
        _, layer = max(all_emb, key=lambda x: x[1].weight.shape[0])
        return layer
    raise RuntimeError("Could not locate the token embedding layer.")


def load_causal_model(
    model_name: str, device: str
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.nn.Embedding]:
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    model.to(device)
    embed = _find_embedding_layer(model)
    print(
        f"  Embedding layer: {embed.weight.shape[0]:,} tokens "
        f"× {embed.weight.shape[1]} dims"
    )
    return tokenizer, model, embed


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_inputs(
    tokenizer: AutoTokenizer, question: str, device: str
) -> torch.Tensor:
    """Tokenise the question, disabling Qwen3 thinking mode when supported."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer([text], return_tensors="pt").to(device)
    except TypeError:
        return tokenizer(question, return_tensors="pt").to(device)


def generate_response(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    question: str,
    max_new_tokens: int,
    device: str,
    *,
    capture_hidden_states: bool = False,
) -> tuple[torch.Tensor, list[str], str, tuple | None]:
    inputs    = _build_inputs(tokenizer, question, device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if capture_hidden_states:
        gen_kwargs["output_hidden_states"]    = True
        gen_kwargs["return_dict_in_generate"] = True

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    if capture_hidden_states:
        new_ids       = out.sequences[0, input_len:]
        hidden_states = out.hidden_states
    else:
        new_ids       = out[0, input_len:]
        hidden_states = None

    tokens = tokenizer.convert_ids_to_tokens(new_ids.tolist())
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_ids, tokens, answer, hidden_states


def get_token_embeddings(
    embed_layer: torch.nn.Embedding,
    token_ids: torch.Tensor,
    device: str,
) -> np.ndarray:
    with torch.no_grad():
        vecs = embed_layer(token_ids.to(device))
    return vecs.cpu().float().numpy()


def compute_top_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hidden_vecs: np.ndarray,
    device: str,
) -> list[str]:
    """Apply the LM head to each hidden vector and return the top-1 decoded token."""
    lm_head  = model.lm_head
    lm_dtype = next(lm_head.parameters()).dtype
    with torch.no_grad():
        h      = torch.tensor(hidden_vecs, device=device).to(lm_dtype)
        logits = lm_head(h)                                    # (n_pts, vocab)
        top_id = logits.argmax(dim=-1).cpu().numpy()           # (n_pts,)
    raw = tokenizer.convert_ids_to_tokens(top_id.tolist())
    return [tokenizer.convert_tokens_to_string([t]).strip() for t in raw]


def extract_layer_coords(hidden_states: tuple, projection) -> np.ndarray:
    """
    Project all per-layer hidden states to 3-D.

    hidden_states: tuple[n_tokens] of tuple[n_layers] of tensor(1, seq, hidden)
    Returns: ndarray of shape (n_tokens, n_layers, 3).
             Layer 0 is the token embedding; layer -1 is the final block output.
    """
    n_tokens = len(hidden_states)
    n_layers = len(hidden_states[0])
    vecs = np.stack([
        hidden_states[t][l][0, -1, :].float().cpu().numpy()
        for t in range(n_tokens)
        for l in range(n_layers)
    ])                                          # (n_tokens * n_layers, hidden_dim)
    coords_flat = projection.transform(vecs)   # (n_tokens * n_layers, 3)
    return coords_flat.reshape(n_tokens, n_layers, 3)


# ---------------------------------------------------------------------------
# Spline
# ---------------------------------------------------------------------------

def build_full_spline(coords: np.ndarray, n_pts: int | None = None) -> np.ndarray:
    """
    Fit a global interpolating spline through coords, sampled at n_pts points.
    Defaults to (n-1)*_STEPS_PER_SEG + 1 when n_pts is None.
    """
    n = len(coords)
    if n < 2:
        return coords.copy()
    k = min(3, n - 1)
    jitter = np.random.default_rng(0).normal(0, 1e-9, coords.shape)
    pts = coords + jitter
    tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0, k=k)
    if n_pts is None:
        n_pts = (n - 1) * _STEPS_PER_SEG + 1
    x, y, z = splev(np.linspace(0, 1, n_pts), tck)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Arrowhead geometry
# ---------------------------------------------------------------------------

def _arrowhead(
    head: np.ndarray, tangent: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wing_a, tip, wing_b) forming a '>' chevron in 3-D."""
    t = tangent / (np.linalg.norm(tangent) + 1e-10)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    perp  = np.cross(t, ref)
    perp /= np.linalg.norm(perp)
    wing_a = head - 0.65 * scale * t + 0.45 * scale * perp
    wing_b = head - 0.65 * scale * t - 0.45 * scale * perp
    return wing_a, head.copy(), wing_b


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_trajectory(
    tokens: list[str],
    coords_all: np.ndarray,
    question: str,
    partial_texts: list[str],
    ms_per_token: int = 300,
    follow: bool = False,
    auto: bool = False,
    top_tokens_per_pt: list[str] | None = None,
    actual_decoded: list[str] | None = None,
) -> None:
    n              = len(coords_all)
    n_layers_total = coords_all.shape[1]
    coords         = coords_all[:, -1, :]       # anchor = final layer output per token
    all_pts        = coords_all.reshape(-1, 3)  # all control points for the spline
    cmap           = cm.autumn
    pt_color       = cmap(np.linspace(0, 1, n))

    # Double the sub-steps when hidden states are active: more control points
    # means the spline has more curvature to resolve, so 2× sampling keeps it smooth.
    steps_per_seg = _STEPS_PER_SEG * 2 if n_layers_total > 1 else _STEPS_PER_SEG
    n_frames      = (n - 1) * steps_per_seg + 1
    full_spline   = build_full_spline(all_pts, n_pts=n_frames)
    frame_ms      = max(50, ms_per_token // steps_per_seg)

    # Frame at which each flat control point j becomes visible
    n_total_pts   = n * n_layers_total
    reveal_frames = [
        round(j * (n_frames - 1) / max(n_total_pts - 1, 1))
        for j in range(n_total_pts)
    ]

    data_range    = float(np.ptp(all_pts, axis=0).max())
    arrow_scale   = data_range * 0.07
    follow_radius = data_range * 0.40

    # ------------------------------------------------------------------ figure
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Ubuntu", "Noto Sans", "DejaVu Sans", "Arial", "Helvetica"],
    })
    _win_title = question[:20] + ("..." if len(question) > 20 else "")
    fig = plt.figure(figsize=(16, 9), num=_win_title)
    fig.patch.set_facecolor("#0a0a14")
    try:
        fig.canvas.manager.set_window_title(_win_title)
    except AttributeError:
        pass

    ax = fig.add_axes([0.0, 0.0, 0.60, 1.0], projection="3d")
    ax.set_facecolor("#0a0a14")
    ax.set_axis_off()
    title_q = question[:68] + ("…" if len(question) > 68 else "")
    ax.set_title(f'"{title_q}"', color="#aaaacc", fontsize=14, pad=6)

    if not follow:
        pad = arrow_scale
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    # Layout constants for right-side panels
    # Response panel sits above the top-tokens panel.
    # In manual mode, both panels are shifted up to leave room for Prev/Next.
    _top_tok_h   = 0.22 if top_tokens_per_pt is not None else 0.0
    _btn_reserve = 0.0  if auto              else 0.14
    _panel_bot   = _btn_reserve + _top_tok_h
    _txt_rect    = [0.61, _panel_bot, 0.37, 0.96 - _panel_bot]
    ax_txt = fig.add_axes(_txt_rect)
    ax_txt.set_facecolor("#0d0d22")
    ax_txt.axis("off")
    ax_txt.text(0.05, 0.97, "Response", va="top", ha="left",
                fontsize=14, color="#555577", transform=ax_txt.transAxes,
                fontweight="bold")
    ax_txt.axvline(x=0, color="#222244", linewidth=1.5)

    status_text = ax_txt.text(
        0.05, 0.02, "", va="bottom", ha="left",
        fontsize=11, color="#444466", transform=ax_txt.transAxes,
    )

    # Top-tokens panel (only when top_tokens_per_pt is provided)
    if top_tokens_per_pt is not None:
        ax_top = fig.add_axes([0.61, _btn_reserve, 0.37, _top_tok_h])
        ax_top.set_facecolor("#0a0a1a")
        ax_top.axis("off")
        ax_top.axvline(x=0, color="#222244", linewidth=1.5)
        ax_top.text(0.05, 0.97, "Top predicted tokens", va="top", ha="left",
                    fontsize=10, color="#445566", transform=ax_top.transAxes,
                    fontweight="bold")
        _top_hint = ax_top.text(
            0.05, 0.72, "hover a token ->", va="top", ha="left",
            fontsize=10, color="#2a3a44", transform=ax_top.transAxes,
            fontstyle="italic",
        )
        _commit_text = ax_top.text(
            0.05, 0.88, "", va="top", ha="left",
            fontsize=9, color="#5577aa", transform=ax_top.transAxes,
        )
        _col_kw = dict(va="top", ha="left", fontsize=10, color="#88bbcc",
                       transform=ax_top.transAxes, fontfamily="monospace", linespacing=1.6)
        _top_col1 = ax_top.text(0.04, 0.72, "", **_col_kw)
        _top_col2 = ax_top.text(0.53, 0.72, "", **_col_kw)
        _uncert_label = ax_top.text(
            0.95, 0.97, "", va="top", ha="right",
            fontsize=9, color="#335566", transform=ax_top.transAxes,
        )
    else:
        ax_top = None
        _top_hint = None
        _commit_text = None
        _top_col1 = None
        _top_col2 = None
        _uncert_label = None

    # Precompute first layer at which the actual generated token was the argmax,
    # and derive an avg "commit layer" across all tokens.
    if top_tokens_per_pt is not None and actual_decoded is not None:
        first_correct: list[int | None] = []
        for i in range(n):
            ps = i * n_layers_total
            pe = min((i + 1) * n_layers_total, len(top_tokens_per_pt))
            actual = actual_decoded[i]
            fl = next((l for l in range(pe - ps)
                       if top_tokens_per_pt[ps + l] == actual), None)
            first_correct.append(fl)
        valid = [fl for fl in first_correct if fl is not None]
        avg_commit = sum(valid) / len(valid) if valid else None
    else:
        first_correct = [None] * n
        avg_commit = None

    # Big global stat — shown in the 3-D axes area
    if avg_commit is not None:
        ax.text2D(
            0.98, 0.06,
            f"avg. commit layer  {avg_commit:.1f}/{n_layers_total} = {avg_commit / n_layers_total:.2f}",
            transform=ax.transAxes, fontsize=13, color="#7799bb",
            fontweight="bold", va="bottom", ha="right",
        )

    # Incremental decoded piece each token adds (handles subwords & spaces)
    token_pieces: list[str] = [partial_texts[0]]
    for i in range(1, n):
        token_pieces.append(partial_texts[i][len(partial_texts[i - 1]):])

    # One pre-coloured text artist per token — positioned below after first draw
    token_text_arts: list = []
    for i in range(n):
        art = ax_txt.text(
            0.05, 0.91, "",
            va="top", ha="left",
            fontsize=13, color=cmap(i / max(n - 1, 1)), alpha=0,
            transform=ax_txt.transAxes, fontfamily="monospace",
        )
        token_text_arts.append(art)

    # --------------------------------- trail: _N_TRAIL_CHUNKS coloured lines
    frames_per_chunk = max(1, n_frames // _N_TRAIL_CHUNKS)
    chunk_lines: list = []
    for i in range(_N_TRAIL_CHUNKS):
        t = (i + 0.5) / _N_TRAIL_CHUNKS
        (ln,) = ax.plot([], [], [], color=cmap(t), linewidth=2.0, alpha=0.80,
                        solid_capstyle="round")
        chunk_lines.append(ln)

    # --------------------------------- arrowhead: two wings, no shaft
    wing1, = ax.plot([], [], [], linewidth=2.5, alpha=0.95, solid_capstyle="round")
    wing2, = ax.plot([], [], [], linewidth=2.5, alpha=0.95, solid_capstyle="round")

    # --------------------------------- scatter for intermediate layer dots (small)
    inter_xs: list = []
    inter_ys: list = []
    inter_zs: list = []
    inter_cs: list = []
    inter_scatter = ax.scatter(
        [], [], [], c=[], cmap=cmap, vmin=0, vmax=1,
        s=9, alpha=0.70, linewidths=0, depthshade=False, zorder=4,
    )

    # --------------------------------- single scatter for all token anchor dots
    dot_xs: list = []
    dot_ys: list = []
    dot_zs: list = []
    dot_cs: list = []
    main_scatter = ax.scatter(
        [], [], [], c=[], cmap=cmap, vmin=0, vmax=1,
        s=60, alpha=0.95, linewidths=0, depthshade=False, zorder=5,
    )

    # Glow layers rendered on top of main scatter when a token is hovered
    hl_halo = ax.scatter([], [], [], s=420, alpha=0.18, color="white",
                         linewidths=0, depthshade=False, zorder=6)
    hl_dot  = ax.scatter([], [], [], s=100, alpha=1.00, color="white",
                         linewidths=0, depthshade=False, zorder=7)

    # ---------------------------------------------------------------- state
    state: dict = {"last_chunk": -1, "last_seg": -1, "n_done": 0, "cam_azim": -60.0}

    # ---------------------------------------------------------------- update
    def update(frame: int):
        current_chunk = min(frame // frames_per_chunk, _N_TRAIL_CHUNKS - 1)

        # Finalise all completed trail chunks
        for i in range(state["last_chunk"] + 1, current_chunk):
            s = i * frames_per_chunk
            e = (i + 1) * frames_per_chunk + 1
            pts = full_spline[s:e]
            chunk_lines[i].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        state["last_chunk"] = max(state["last_chunk"], current_chunk - 1)

        # Grow the current chunk up to this frame
        s   = current_chunk * frames_per_chunk
        pts = full_spline[s : frame + 1]
        if len(pts) > 0:
            chunk_lines[current_chunk].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

        # Arrowhead '>' — hidden on the last frame
        hp      = full_spline[frame]
        f_next  = min(frame + 2, n_frames - 1)
        f_prev  = max(frame - 2, 0)
        tangent = full_spline[f_next] - full_spline[f_prev]

        if frame == n_frames - 1:
            wing1.set_data_3d([], [], [])
            wing2.set_data_3d([], [], [])
        else:
            t_color = frame / max(n_frames - 1, 1)
            color   = cmap(t_color)
            wa, tip, wb = _arrowhead(hp, tangent, arrow_scale)
            wing1.set_data_3d([wa[0], tip[0]], [wa[1], tip[1]], [wa[2], tip[2]])
            wing2.set_data_3d([wb[0], tip[0]], [wb[1], tip[1]], [wb[2], tip[2]])
            wing1.set_color(color)
            wing2.set_color(color)

        # Reveal dots (anchor + intermediate) as the head passes each control point
        cur = state["last_seg"]
        while cur + 1 < n_total_pts and reveal_frames[cur + 1] <= frame:
            cur += 1
            tok_i = cur // n_layers_total
            lay_i = cur % n_layers_total
            x, y, z = all_pts[cur]
            c_val   = tok_i / max(n - 1, 1)
            if lay_i == n_layers_total - 1:
                dot_xs.append(x); dot_ys.append(y); dot_zs.append(z)
                dot_cs.append(c_val)
                token_text_arts[tok_i].set_alpha(0.95)
                state["n_done"] += 1
            else:
                inter_xs.append(x); inter_ys.append(y); inter_zs.append(z)
                inter_cs.append(c_val)

        if cur != state["last_seg"]:
            state["last_seg"] = cur
            main_scatter._offsets3d = (dot_xs[:], dot_ys[:], dot_zs[:])
            if dot_cs:
                main_scatter.set_array(np.array(dot_cs))
            inter_scatter._offsets3d = (inter_xs[:], inter_ys[:], inter_zs[:])
            if inter_cs:
                inter_scatter.set_array(np.array(inter_cs))

        status_text.set_text(f"{state['n_done']} / {n} tokens")

        # Camera follow mode
        if follow:
            r = follow_radius
            ax.set_xlim(hp[0] - r, hp[0] + r)
            ax.set_ylim(hp[1] - r, hp[1] + r)
            ax.set_zlim(hp[2] - r, hp[2] + r)
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                t_unit      = tangent / norm
                azim_target = float(np.degrees(np.arctan2(t_unit[1], t_unit[0]))) + 180
                diff        = (azim_target - state["cam_azim"] + 180) % 360 - 180
                state["cam_azim"] += 0.15 * diff
                ax.view_init(elev=18, azim=state["cam_azim"])

        return [*chunk_lines, wing1, wing2, inter_scatter, main_scatter, *token_text_arts, status_text]

    # One-shot render so the renderer is ready for text measurement
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb    = ax_txt.get_window_extent(renderer)
    probe    = ax_txt.text(0, 0, "M", fontsize=13, fontfamily="monospace",
                           transform=ax_txt.transAxes, alpha=0)
    pb       = probe.get_window_extent(renderer)
    probe.remove()

    # Character dimensions in axes-fraction units
    cw = pb.width  / max(ax_bb.width,  1)
    lh = pb.height / max(ax_bb.height, 1) * 1.55

    # Flow token text artists like a paragraph (no newline per token)
    x0, x1 = 0.05, 0.93
    cx, cy  = x0, 0.90
    for i in range(n):
        piece = token_pieces[i]
        if not piece:
            continue
        disp = piece
        # strip leading space when cursor is at line start
        if cx <= x0 and disp.startswith(" "):
            disp = disp[1:]
        pw = len(disp) * cw
        # wrap if this piece would overflow the line
        if cx > x0 and cx + pw > x1:
            cy -= lh
            cx  = x0
            if disp.startswith(" "):
                disp = disp[1:]
                pw   = len(disp) * cw
        if cy < 0.06 or not disp:
            continue
        token_text_arts[i].set_position((cx, cy))
        token_text_arts[i].set_text(disp)
        cx += pw

    # -------------------------------------------------------------- hover link
    _empty = (np.array([]), np.array([]), np.array([]))
    _hovered: list = [None]

    def _on_hover(event):
        if event.inaxes is not ax_txt:
            if _hovered[0] is not None:
                _hovered[0] = None
                hl_halo._offsets3d = _empty
                hl_dot._offsets3d  = _empty
                fig.canvas.draw_idle()
            return

        rdr   = fig.canvas.get_renderer()
        found = None
        for i, art in enumerate(token_text_arts):
            alpha = art.get_alpha()
            if alpha is None or alpha < 0.5 or not art.get_text():
                continue
            if art.get_window_extent(rdr).contains(event.x, event.y):
                found = i
                break

        if found == _hovered[0]:
            return

        # Un-highlight the previously hovered text
        prev = _hovered[0]
        if prev is not None and prev < len(token_text_arts):
            token_text_arts[prev].set_alpha(0.95)
            token_text_arts[prev].set_fontweight("normal")

        _hovered[0] = found

        if found is not None and found < len(coords):
            col = cmap(found / max(n - 1, 1))
            x, y, z = coords[found]
            hl_halo._offsets3d = ([x], [y], [z])
            hl_halo.set_facecolor((*col[:3], 0.28))
            hl_dot._offsets3d  = ([x], [y], [z])
            hl_dot.set_facecolor(col)
            token_text_arts[found].set_alpha(1.0)
            token_text_arts[found].set_fontweight("bold")
            # Top-tokens panel
            if top_tokens_per_pt is not None:
                pt_start    = found * n_layers_total
                pt_end      = min((found + 1) * n_layers_total, len(top_tokens_per_pt))
                freq: Counter = Counter()
                n_lay = max(pt_end - pt_start, 1)
                for layer_i in range(pt_end - pt_start):
                    tok = top_tokens_per_pt[pt_start + layer_i]
                    freq[tok] += 1
                top10 = [(t, c) for t, c in freq.most_common(30)
                         if t.isprintable() and t.isascii() and t.strip()][:10]

                def _fmt(tok, cnt):
                    return f"{tok[:12]:<13}{cnt/n_lay:.2f}"

                col1 = "\n".join(_fmt(t, c) for t, c in top10[:5])
                col2 = "\n".join(_fmt(t, c) for t, c in top10[5:])
                _top_hint.set_visible(False)
                _top_col1.set_text(col1)
                _top_col2.set_text(col2)
                _uncert_label.set_text("cert.")

                fl = first_correct[found]
                if fl is not None:
                    _commit_text.set_text(
                        f'"{actual_decoded[found]}"  ->  layer {fl}/{n_lay - 1} = {fl / max(n_lay - 1, 1):.2f}'
                    )
                else:
                    _commit_text.set_text(
                        f'"{actual_decoded[found]}"  ->  never top-1'
                    )
        else:
            hl_halo._offsets3d = _empty
            hl_dot._offsets3d  = _empty
            if top_tokens_per_pt is not None:
                _top_hint.set_visible(True)
                _commit_text.set_text("")
                _top_col1.set_text("")
                _top_col2.set_text("")
                _uncert_label.set_text("")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_hover)

    if auto:
        anim = FuncAnimation(   # noqa: F841
            fig, update,
            frames=n_frames,
            interval=frame_ms,
            blit=False,
            repeat=False,
        )
    else:
        # ---------------------------------------------------------------- manual step mode

        # Thin progress bar just above the button row
        ax_prog = fig.add_axes([0.620, 0.154, 0.355, 0.009])
        ax_prog.set_xlim(0, 1)
        ax_prog.set_ylim(0, 1)
        ax_prog.axis("off")
        ax_prog.barh(0.5, 1.0, height=1.0, color="#111e2e", align="center")
        prog_bar = ax_prog.barh(0.5, 1.0 / max(n, 1), height=1.0,
                                color="#4fc3f7", align="center")[0]

        # ◀ Prev button
        ax_prev = fig.add_axes([0.620, 0.040, 0.107, 0.100])
        btn_prev = mwidgets.Button(ax_prev, "<< Prev", color="#0d1b2a", hovercolor="#0f3555")
        btn_prev.label.set_color("#4fc3f7")
        btn_prev.label.set_fontsize(14)
        btn_prev.label.set_fontweight("bold")
        for _sp in ax_prev.spines.values():
            _sp.set_edgecolor("#2a5a7a"); _sp.set_linewidth(0.8)

        # Token counter label
        ax_lbl = fig.add_axes([0.737, 0.040, 0.130, 0.100])
        ax_lbl.set_facecolor("#0d1b2a")
        ax_lbl.axis("off")
        for _sp in ax_lbl.spines.values():
            _sp.set_edgecolor("#1a3a55"); _sp.set_linewidth(0.5)
        tok_label = ax_lbl.text(
            0.5, 0.5, f"1 / {n}",
            va="center", ha="center",
            fontsize=15, color="#e0e8ff", fontweight="bold",
            transform=ax_lbl.transAxes,
        )

        # Next ▶ button
        ax_next = fig.add_axes([0.877, 0.040, 0.098, 0.100])
        btn_next = mwidgets.Button(ax_next, "Next >>", color="#0d1b2a", hovercolor="#0f3555")
        btn_next.label.set_color("#4fc3f7")
        btn_next.label.set_fontsize(14)
        btn_next.label.set_fontweight("bold")
        for _sp in ax_next.spines.values():
            _sp.set_edgecolor("#2a5a7a"); _sp.set_linewidth(0.8)

        cur_token = [0]

        def _set_to_token(k: int) -> None:
            """Directly render all artists to match state at token k."""
            k = max(0, min(n - 1, k))
            anchor_j     = (k + 1) * n_layers_total - 1
            target_frame = reveal_frames[anchor_j]

            # Spline trail chunks
            for i in range(_N_TRAIL_CHUNKS):
                s   = i * frames_per_chunk
                end = min((i + 1) * frames_per_chunk + 1, target_frame + 1)
                if s > target_frame:
                    chunk_lines[i].set_data_3d([], [], [])
                else:
                    pts = full_spline[s:end]
                    chunk_lines[i].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

            # Arrowhead
            if target_frame >= n_frames - 1:
                wing1.set_data_3d([], [], [])
                wing2.set_data_3d([], [], [])
            else:
                hp      = full_spline[target_frame]
                tangent = (full_spline[min(target_frame + 2, n_frames - 1)]
                           - full_spline[max(target_frame - 2, 0)])
                t_color = target_frame / max(n_frames - 1, 1)
                wa, tip, wb = _arrowhead(hp, tangent, arrow_scale)
                wing1.set_data_3d([wa[0], tip[0]], [wa[1], tip[1]], [wa[2], tip[2]])
                wing2.set_data_3d([wb[0], tip[0]], [wb[1], tip[1]], [wb[2], tip[2]])
                wing1.set_color(cmap(t_color))
                wing2.set_color(cmap(t_color))

            # Dots
            d_xs, d_ys, d_zs, d_cs = [], [], [], []
            ix, iy, iz, ic          = [], [], [], []
            for j in range(n_total_pts):
                if reveal_frames[j] > target_frame:
                    break
                tok_i = j // n_layers_total
                lay_i = j % n_layers_total
                x, y, z = all_pts[j]
                c_val   = tok_i / max(n - 1, 1)
                if lay_i == n_layers_total - 1:
                    d_xs.append(x); d_ys.append(y); d_zs.append(z); d_cs.append(c_val)
                else:
                    ix.append(x); iy.append(y); iz.append(z); ic.append(c_val)

            main_scatter._offsets3d = (d_xs, d_ys, d_zs)
            if d_cs:
                main_scatter.set_array(np.array(d_cs))
            inter_scatter._offsets3d = (ix, iy, iz)
            if ic:
                inter_scatter.set_array(np.array(ic))

            # Text visibility
            for i in range(n):
                token_text_arts[i].set_alpha(0.95 if i <= k else 0)

            # Update navigation widgets
            prog_bar.set_width((k + 1) / n)
            tok_label.set_text(f"{k + 1} / {n}")
            status_text.set_text(f"Token {k + 1} / {n}")

            if follow and target_frame > 0:
                hp      = full_spline[target_frame]
                tangent = (full_spline[min(target_frame + 2, n_frames - 1)]
                           - full_spline[max(target_frame - 2, 0)])
                r = follow_radius
                ax.set_xlim(hp[0] - r, hp[0] + r)
                ax.set_ylim(hp[1] - r, hp[1] + r)
                ax.set_zlim(hp[2] - r, hp[2] + r)
                norm = np.linalg.norm(tangent)
                if norm > 1e-10:
                    t_unit = tangent / norm
                    ax.view_init(elev=18,
                                 azim=float(np.degrees(np.arctan2(t_unit[1], t_unit[0]))) + 180)

            fig.canvas.draw_idle()

        def _step(delta, _event=None):
            new_k = max(0, min(n - 1, cur_token[0] + delta))
            if new_k != cur_token[0]:
                cur_token[0] = new_k
                _set_to_token(new_k)

        btn_prev.on_clicked(lambda e: _step(-1))
        btn_next.on_clicked(lambda e: _step(+1))

        def _on_key(event):
            if event.key in ("right", "d"):
                _step(+1)
            elif event.key in ("left", "a"):
                _step(-1)

        fig.canvas.mpl_connect("key_press_event", _on_key)
        _set_to_token(0)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate an LLM answer as a spline trajectory in latent space"
    )
    parser.add_argument("question",       nargs="?", default=None)
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--max-tokens",   type=int, default=80)
    parser.add_argument("--ms-per-token", type=int, default=300,
                        help="Animation time per token in ms (default: 300)")
    parser.add_argument("--follow",       action="store_true",
                        help="Camera follows the arrowhead")
    parser.add_argument("--auto",              action="store_true",
                        help="Auto-play animation (default: step manually with ←/→ or Prev/Next buttons)")
    parser.add_argument("--add-hidden-states", action="store_true",
                        help="Project hidden states from all layers (default: token embeddings only)")
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)

    cfg             = load_config(args.config)
    model_name      = cfg["model"]["name"]
    projection_path = cfg.get("projection", {}).get("output_path", "models/projection.pkl")
    device          = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, embed = load_causal_model(model_name, device)

    print(f"\nGenerating answer (max {args.max_tokens} tokens) …")
    new_ids, tokens, answer, hidden_states = generate_response(
        tokenizer, model, question, args.max_tokens, device,
        capture_hidden_states=args.add_hidden_states,
    )

    print(f"\nAnswer:\n{answer}\n")
    print(f"{len(tokens)} tokens generated")

    partial_texts = [
        tokenizer.decode(new_ids[: i + 1], skip_special_tokens=True)
        for i in range(len(tokens))
    ]

    projection = LatentProjection.load(projection_path)

    if args.add_hidden_states:
        print(f"Projecting {len(hidden_states[0])} layers × {len(tokens)} tokens to 3-D …")
        coords_all  = extract_layer_coords(hidden_states, projection)
        hidden_vecs = np.stack([
            hidden_states[t][l][0, -1, :].float().cpu().numpy()
            for t in range(len(tokens))
            for l in range(len(hidden_states[0]))
        ])
    else:
        print(f"Projecting {len(tokens)} token embeddings to 3-D …")
        hidden_vecs = get_token_embeddings(embed, new_ids, device)
        coords_3d   = projection.transform(hidden_vecs)   # (n, 3)
        coords_all  = coords_3d[:, np.newaxis, :]          # (n, 1, 3)

    print("Computing top predicted tokens …")
    top_tokens_per_pt = compute_top_tokens(model, tokenizer, hidden_vecs, device)
    actual_decoded = [
        tokenizer.convert_tokens_to_string([t]).strip() for t in tokens
    ]

    print("Launching animation …")
    animate_trajectory(
        tokens, coords_all, question, partial_texts,
        ms_per_token=args.ms_per_token,
        follow=args.follow,
        auto=args.auto,
        top_tokens_per_pt=top_tokens_per_pt,
        actual_decoded=actual_decoded,
    )


if __name__ == "__main__":
    main()
