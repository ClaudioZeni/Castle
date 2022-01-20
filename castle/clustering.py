from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np


class Clustering(object):
    def __init__(self, clustering_type):
        self.clustering_type = clustering_type 
     
    def optimize_n_clusters(self, X):
        S = []
        for i in np.arange(2, min(len(X)//10, 10)):
            gmm = GaussianMixture(n_components=i, n_init=3, reg_covar=1e-2)
            labels = gmm.fit_predict(X)
            S.append(silhouette_score(X, labels, metric='euclidean'))
        nopt = np.argmax(S*np.arange(1, 1+len(S))**0.5)
        gmm = GaussianMixture(n_components=nopt, n_init=5, reg_covar=1e-2).fit(X)
        print("Using %i clusters" %(nopt))
        return gmm

    def fit(self, X, e, n_clusters='auto'):
        """Auxiliary function that calls the correct clustering
            algorithm. Options are: kmeans clustering and advanced
            density peaks clustering. If the latter is chosen, the
            adp python package must be installed first.
        Args:
            X (np.array): Descriptor vector for each atom in each structure
            e (np.array): Per-atom energy of each structure
            n_clusters (float): Number of clusters
            clustering (str): Clustering algorithm to use

        Returns:
            labels (np.array): cluster (int) assigned
                            to each atom in each structure

        """
        if self.clustering_type == 'e_gmm':
            # Resize X based only on global std and energy std, separately
            # So that e is comparable to X but we do not lose information
            # on the magnitude of each component of X.
            mean = np.mean(X, axis=0)
            std = np.std(X)
            X = (X - mean[None, :]) / std[None, None]
            e = (e - np.mean(e)) / np.std(e)
            X = np.concatenate((X, e[:, None]), axis=1)
            if n_clusters == 'auto':
                gmm = self.optimize_n_clusters(X)
            else:
                gmm = GaussianMixture(n_components=n_clusters, n_init=5, reg_covar=1e-2).fit(X)
            self.labels = gmm.predict(X)
            self.weights = gmm.weights_
            self.centers = gmm.means_[:, :-1] * std + mean[None, :]
            self.precisions = gmm.precisions_[:, :-1, :-1] / std
            self.cov_dets = np.array([1/np.linalg.det(self.precisions[i]) for i in range(len(self.weights))])

        elif self.clustering_type == 'gmm':
            if n_clusters == 'auto':
                gmm = self.optimize_n_clusters(X)
            else:
                gmm = GaussianMixture(n_components=n_clusters, n_init=5, reg_covar=1e-2).fit(X)
            self.labels = gmm.predict(X)
            self.weights = gmm.weights_
            self.centers = gmm.means_ 
            self.precisions = gmm.precisions_
            self.cov_dets = np.array([1/np.linalg.det(self.precisions[i]) for i in range(len(self.weights))])

        elif self.clustering_type == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters)
            self.labels = kmeans.fit_predict(X)
            self.centers = kmeans.cluster_centers_
            self.precisions = 1/np.array([np.std(X[self.labels == i], axis = 0) for i in range(n_clusters)])
            self.weights = np.array([len(self.labels == i) for i in range(n_clusters)])

        self.n_clusters = n_clusters

    def get_models_weight(self, X_avg, dX_dr=None, compute_der=False):
        if self.clustering_type == 'gmm' or self.clustering_type == 'e_gmm':
            return self.get_models_weight_gmm(X_avg, dX_dr, compute_der)

        elif self.clustering_type == 'kmeans':
            return self.get_models_weight_kmeans(X_avg, dX_dr, compute_der)

    def get_models_weight_kmeans(self, X_avg, dX_dr=None, compute_der=False):
        """Compute weight (and derivative w.r.t. each atom's position)
            of models for a single structure.

        m: number of atoms in configuration
        c: cartesian coordinates
        d: number of dimensions of descriptor
        s: number of models

        X_avg: (d)
        dX_dr: (m, c, d)
        self.centers : (s, d) 
        self.precisions : (s, d) 
        self.weights : (s)
        self.cov_dets : (s)
        proba: (s)
        single_proba_der: (s, m, c)
        softmax_der: (s, s)
        proba_der: (m, s, c)
        """
        # Distance from center
        diff = X_avg[None, :] - self.centers[:, :]
        # Compute exponential disrance
        proba = self.weights**0.5/np.sum((self.precisions*diff**2), axis = 1)
        # Normalize to get sums up to 1
        # To avoid nans if all clusters have probability 0 or nan
        proba = np.nan_to_num(proba, copy=False, nan=0)
        if not sum(proba) == 0:
            proba_sum = np.sum(proba)
            proba_norm = proba/proba_sum
        if compute_der:
            if not sum(proba) == 0:
                single_proba_der = np.einsum('s, mcd, sd -> smc', proba, dX_dr, -2/diff)
                sum_der = - proba[:, None] / proba_sum**2 + np.eye(len(proba)) / proba_sum
                proba_der = np.einsum('smc, ts -> msc', single_proba_der, sum_der)

            # To avoid absurd numbers if all clusters have probability 0 or nan
            else:
                proba_der = np.zeros((dX_dr.shape[0], len(proba), 3))
            return proba_norm, proba_der
        else:
            return proba_norm

    def get_models_weight_gmm(self, X_avg, dX_dr=None, compute_der=False):
        """Compute weight (and derivative w.r.t. each atom's position)
            of models for a single structure.

        m: number of atoms in configuration
        c: cartesian coordinates
        d: number of dimensions of descriptor
        s: number of models

        X_avg: (d)
        dX_dr: (m, c, d)
        self.centers : (s, d) 
        self.precisions : (s, d, d) 
        self.weights : (s)
        self.cov_dets : (s)
        proba: (s)
        single_proba_der: (s, m, c)
        softmax_der: (s, s)
        proba_der: (m, s, c)
        """
        # Distance from center
        diff = X_avg[None, :] - self.centers[:, :]
        # Compute exponential disrance
        proba = np.exp(-0.5*(np.einsum('sf, sdf, sd -> s', diff, self.precisions, diff)))
        # Normalize for determinant of covariance and number of dimensions
        proba = proba/(((2*np.pi)**len(X_avg)*self.cov_dets)**0.5)
        # Multiply by GMM weight
        proba = proba * self.weights
        # Normalize to get sums up to 1

        # To avoid nans if all clusters have probability 0 or nan
        proba = np.nan_to_num(proba, copy=False, nan=0)
        if not sum(proba) == 0:
            proba = proba/np.sum(proba)

        if compute_der:
            if not sum(proba) == 0:
                der_diff = np.einsum('sf, sdf -> sd', diff, self.precisions)
                single_proba_der = np.einsum('s, mcd, sd -> smc', proba, dX_dr, der_diff)
                softmax_der = - proba[:, None] * proba[None, :] + np.diag(proba)
                proba_der = np.einsum('smc, st -> msc', single_proba_der, softmax_der)

            # To avoid absurd numbers if all clusters have probability 0 or nan
            else:
                proba_der = np.zeros((dX_dr.shape[0], len(proba), 3))
            return proba, proba_der
        else:
            return proba
