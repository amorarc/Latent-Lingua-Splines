"""
Two-stage dimensionality reduction: PCA → UMAP.

PCA handles the linear structure first (and removes noise in the high-dim space),
then UMAP captures the nonlinear manifold in the lower-dim PCA coordinates.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from umap import UMAP


class LatentProjection:
    def __init__(
        self,
        pca_components: int = 50,
        umap_components: int = 2,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_random_state: int = 42,
        umap_n_jobs: int = 1,
    ):
        self.pca = PCA(n_components=pca_components)
        self.umap = UMAP(
            n_components=umap_components,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
            n_jobs=umap_n_jobs,
        )
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> "LatentProjection":
        """Fit PCA then UMAP on the full embedding matrix."""
        print(f"Fitting PCA ({self.pca.n_components} components) …")
        pca_coords = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained:.2%}")

        print(f"Fitting UMAP ({self.umap.n_components} components) …")
        self.umap.fit(pca_coords)

        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        pca_coords = self.pca.transform(embeddings)
        return self.umap.transform(pca_coords)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"Fitting PCA ({self.pca.n_components} components) …")
        pca_coords = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained:.2%}")

        print(f"Fitting UMAP ({self.umap.n_components} components) …")
        coords = self.umap.fit_transform(pca_coords)

        self._fitted = True
        return coords

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pca": self.pca, "umap": self.umap}, path)
        print(f"Projection saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LatentProjection":
        data = joblib.load(path)
        pca: PCA = data["pca"]
        umap_: UMAP = data["umap"]

        obj = cls.__new__(cls)
        obj.pca = pca
        obj.umap = umap_
        obj._fitted = True
        return obj
