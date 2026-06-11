#!/usr/bin/env python3
"""
Fit a PCA → UMAP projection on the model's full vocabulary embeddings and
plot every token as a point in 3-D space, coloured by azimuthal angle.

Usage:
    python scripts/train_projection.py
    python scripts/train_projection.py --config config/config.yaml
    python scripts/train_projection.py --config config/config.yaml --no-viz
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src import EmbeddingExtractor, LatentProjection, load_config


_LATIN_OR_DIGIT = re.compile(r"[a-zA-Z0-9]")


# ---------------------------------------------------------------------------
# Vocab extraction + filter
# ---------------------------------------------------------------------------

def _vocab_embeddings(
    extractor: EmbeddingExtractor,
    n_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Return embeddings and token strings for vocab entries that contain at least
    one Latin letter or digit.  If *n_samples* is set, a random subset of that
    size is drawn from the filtered pool and used to fit the projection.
    """
    print("Extracting full vocabulary embeddings …")
    all_vecs = extractor.full_vocab_embeddings()          # (vocab_size, hidden)
    all_tokens = extractor.tokenizer.convert_ids_to_tokens(
        list(range(all_vecs.shape[0]))
    )

    mask = [bool(_LATIN_OR_DIGIT.search(t or "")) for t in all_tokens]
    vecs   = all_vecs[[i for i, keep in enumerate(mask) if keep]]
    tokens = [t for t, keep in zip(all_tokens, mask) if keep]

    print(
        f"  vocab size: {len(all_tokens)} → "
        f"{len(tokens)} kept after Latin/digit filter "
        f"({len(all_tokens) - len(tokens)} dropped)"
    )

    if n_samples is not None and n_samples != -1 and n_samples < len(tokens):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tokens), size=n_samples, replace=False)
        idx.sort()
        vecs   = vecs[idx]
        tokens = [tokens[i] for i in idx]
        print(f"  sampled {n_samples:,} tokens for projection fitting")

    return vecs, tokens


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_embedding_space(
    coords: np.ndarray,
    tokens: list[str],
    output_path: str,
    dpi: int = 150,
    point_size: float = 2,
    alpha: float = 0.6,
) -> None:
    # azimuthal angle in XY plane: atan2(y, x) → [-π, π] → [0, 1]
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    colors = (angles + np.pi) / (2 * np.pi)   # normalise to [0, 1]

    cmap = plt.cm.hsv
    fig = plt.figure(figsize=(26, 20))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=colors, cmap=cmap, s=point_size, alpha=alpha,
        linewidths=0,
    )

    # colour bar showing angle scale
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=-180, vmax=180))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Azimuthal angle (°)", fontsize=9)

    ax.set_title(f"Token Embedding Space — {len(tokens):,} tokens", fontsize=13)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_zlabel("UMAP 3", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PCA+UMAP projection")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip the 3-D scatter plot")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1 — Load model and extract filtered vocab embeddings ------------------
    extractor = EmbeddingExtractor(cfg["model"]["name"])
    vecs, tokens = _vocab_embeddings(
        extractor,
        n_samples=cfg.get("projection_samples"),
    )

    # 2 — Fit PCA + UMAP ----------------------------------------------------
    pca_cfg  = cfg.get("pca",  {})
    umap_cfg = cfg.get("umap", {})

    projection = LatentProjection(
        pca_components=pca_cfg.get("n_components", 64),
        umap_components=umap_cfg.get("n_components", 3),
        umap_neighbors=umap_cfg.get("n_neighbors", 15),
        umap_min_dist=umap_cfg.get("min_dist", 0.1),
        umap_metric=umap_cfg.get("metric", "cosine"),
        umap_random_state=umap_cfg.get("random_state", 42),
        umap_n_jobs=umap_cfg.get("n_jobs", 1),
    )
    coords = projection.fit_transform(vecs)   # (n_tokens, 3)

    # 3 — Save fitted projection --------------------------------------------
    projection.save(cfg.get("projection", {}).get("output_path", "models/projection.pkl"))

    # 4 — Visualise ---------------------------------------------------------
    if not args.no_viz:
        viz = cfg.get("visualization", {})
        _plot_embedding_space(
            coords, tokens,
            output_path=viz.get("output_path", "outputs/embedding_space.png"),
            dpi=viz.get("dpi", 150),
            point_size=viz.get("point_size", 2),
            alpha=viz.get("alpha", 0.6),
        )


if __name__ == "__main__":
    main()
