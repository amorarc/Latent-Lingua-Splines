#!/usr/bin/env python3
"""
Train the PCA → UMAP projection on a corpus and save phrase trajectories.

Usage:
    python scripts/train_projection.py
    python scripts/train_projection.py --config config/config.yaml
    python scripts/train_projection.py --config config/config.yaml --no-viz
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from src import EmbeddingExtractor, LatentProjection, load_config, load_corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_corpus_embeddings(
    extractor: EmbeddingExtractor,
    phrases: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Collect all token embeddings and their decoded tokens."""
    print("Extracting embeddings from corpus …")
    all_vecs, all_tokens = extractor.corpus_embeddings(phrases)
    print(f"  {len(all_tokens)} tokens from {len(phrases)} phrases")
    return all_vecs, all_tokens


def _phrase_trajectories_in_2d(
    extractor: EmbeddingExtractor,
    projection: LatentProjection,
    phrases: list[str],
) -> list[dict]:
    """Project each phrase's token sequence through PCA+UMAP."""
    trajectories = []
    for phrase in tqdm(phrases, desc="Projecting trajectories"):
        traj = extractor.phrase_trajectory(phrase)
        coords = projection.transform(traj.embeddings)  # (n_tokens, 2)
        trajectories.append({
            "phrase": phrase,
            "tokens": traj.tokens,
            "coords": coords,
        })
    return trajectories


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_trajectories(
    trajectories: list[dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    n = len(trajectories)
    colors = cm.tab20.colors if n <= 20 else cm.turbo(np.linspace(0, 1, n))

    # detect dimensionality from the first trajectory
    is_3d = trajectories[0]["coords"].shape[1] == 3

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    for i, traj in enumerate(trajectories):
        coords = traj["coords"]
        color = colors[i % len(colors)]
        phrase_label = traj["phrase"][:40]

        if is_3d:
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                    "-o", color=color, linewidth=1.5, markersize=5,
                    alpha=0.8, label=phrase_label)
            ax.scatter(*coords[0],  s=80, color=color, marker="^", zorder=5)
            ax.scatter(*coords[-1], s=80, color=color, marker="s", zorder=5)
            for j, token in enumerate(traj["tokens"]):
                ax.text(coords[j, 0], coords[j, 1], coords[j, 2],
                        token, fontsize=6, alpha=0.7)
        else:
            ax.plot(coords[:, 0], coords[:, 1], "-o", color=color,
                    linewidth=1.5, markersize=5, alpha=0.8, label=phrase_label)
            ax.scatter(*coords[0],  s=80, color=color, marker="^", zorder=5)
            ax.scatter(*coords[-1], s=80, color=color, marker="s", zorder=5)
            for j, token in enumerate(traj["tokens"]):
                ax.annotate(token, coords[j], fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

    ax.set_title("Phrase Trajectories in Latent Space (PCA → UMAP)", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    if is_3d:
        ax.set_zlabel("UMAP dim 3")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
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
                        help="Skip trajectory visualisation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1 — Load corpus -------------------------------------------------------
    phrases = load_corpus(
        cfg["data"]["corpus_path"],
        max_phrases=cfg["data"].get("max_phrases"),
    )

    # 2 — Extract embeddings ------------------------------------------------
    extractor = EmbeddingExtractor(
        cfg["model"]["name"],
        quantization=cfg["model"].get("quantization"),
    )
    corpus_vecs, _ = _build_corpus_embeddings(extractor, phrases)

    # 3 — Fit PCA + UMAP projection -----------------------------------------
    proj_cfg = cfg.get("projection", {})
    pca_cfg  = cfg.get("pca",  {})
    umap_cfg = cfg.get("umap", {})

    projection = LatentProjection(
        pca_components=pca_cfg.get("n_components", 50),
        umap_components=umap_cfg.get("n_components", 2),
        umap_neighbors=umap_cfg.get("n_neighbors", 15),
        umap_min_dist=umap_cfg.get("min_dist", 0.1),
        umap_metric=umap_cfg.get("metric", "cosine"),
        umap_random_state=umap_cfg.get("random_state", 42),
    )
    projection.fit(corpus_vecs)

    # 4 — Save the fitted projection -----------------------------------------
    projection.save(proj_cfg.get("output_path", "models/projection.pkl"))

    # 5 — Visualise trajectories ---------------------------------------------
    if not args.no_viz:
        viz_cfg = cfg.get("visualization", {})
        max_t   = viz_cfg.get("max_trajectories", 20)
        sample  = phrases[:max_t]

        trajectories = _phrase_trajectories_in_2d(extractor, projection, sample)
        _plot_trajectories(
            trajectories,
            output_path=viz_cfg.get("output_path", "outputs/trajectories.png"),
            dpi=viz_cfg.get("dpi", 150),
        )


if __name__ == "__main__":
    main()
