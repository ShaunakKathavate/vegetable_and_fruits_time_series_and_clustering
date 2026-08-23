"""Pure-NumPy KMeans (Lloyd's algorithm, k-means++ init, multi-restart).

Exists because scikit-learn's compiled KMeans (`sklearn.cluster.KMeans`) crashes on this
machine's conda environment -- its Cython/OpenMP internals hit a broken vcomp/OpenBLAS DLL
(reproducible native crash, exit code 0xc06d007f, unrelated to this project's code). Everything
else in scikit-learn used here (StandardScaler, Pipeline) is pure Python/NumPy and unaffected.

Drop-in enough for this project: exposes fit / predict / fit_predict, cluster_centers_,
labels_, inertia_, and get_params/set_params so it plugs into a sklearn.pipeline.Pipeline.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class SimpleKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_plusplus_init(self, X, rng):
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        first = rng.integers(n_samples)
        centers[0] = X[first]
        closest_sq_dist = np.sum((X - centers[0]) ** 2, axis=1)
        for i in range(1, self.n_clusters):
            probs = closest_sq_dist / closest_sq_dist.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centers[i] = X[next_idx]
            new_sq_dist = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)
        return centers

    def _lloyd_once(self, X, rng):
        centers = self._kmeans_plusplus_init(X, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)

            new_centers = centers.copy()
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # empty cluster: re-seed at the point farthest from its current center
                    farthest = dists[np.arange(X.shape[0]), new_labels].argmax()
                    new_centers[k] = X[farthest]

            shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            centers = new_centers
            labels = new_labels
            if shift < self.tol:
                break

        inertia = ((X - centers[labels]) ** 2).sum()
        return centers, labels, inertia

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_inertia = None
        for _ in range(self.n_init):
            centers, labels, inertia = self._lloyd_once(X, rng)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
