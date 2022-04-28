from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np


class LocalClustering(object):
    def __init__(self, clustering_type, baseline_percentile=0):
        self.clustering_type = clustering_type 
        self.baseline_percentile = baseline_percentile
        if not 0<= self.baseline_percentile < 1:
            raise ValueError("Baseline_percentile must be in [0, 1[")
     
    def optimize_n_clusters(self, X, model):
        S = []
        for i in np.arange(2, min(len(X)//10, 20)):
            model.n_components = i
            model.n_clusters = i
            labels = model.fit_predict(X)
            S.append(silhouette_score(X, labels, metric='euclidean'))
        nopt = np.argmax(S*np.arange(1, 1+len(S))**0.5)
        model.n_components = nopt
        model.n_clusters = nopt        
        model.fit(X)
        print("Using %i clusters" %(nopt))
        return model

    def fit(self, X, n_clusters='auto'):
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
        print("Clustering data")
        if self.clustering_type == 'kmeans':
            if n_clusters == 'auto':
                model = self.optimize_n_clusters(X, KMeans())
            else:
                model = KMeans(n_clusters=n_clusters).fit(X)
            self.n_clusters = model.n_clusters
            self.labels = model.predict(X)
            self.weights = np.array([len(self.labels == i) for i in range(self.n_clusters)])
            self.centers = model.cluster_centers_ 
            self.precisions = 1/np.array([np.std(X[self.labels == i], axis = 0) for i in range(self.n_clusters)])

        elif self.clustering_type == 'adp':
            from dadapy import Data
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_ = (X - mean[None, :]) / std[None, :]
            data = Data(X_)
            data.compute_clustering(Z = 1.65, halo=False)
            self.n_clusters = data.N_clusters
            self.labels = data.cluster_assignment
            self.weights = np.array([len(self.labels == i) for i in range(self.n_clusters)])
            self.centers = np.array([X[i] for i in data.cluster_centers])
            self.precisions = 1/np.array([np.std(X[self.labels == i], axis = 0) for i in range(self.n_clusters)])

        if self.baseline_percentile > 0:
            self.get_baseline_hyperparameter(X)
        else:
            self.baseline_prob = 0
            
    def get_baseline_hyperparameter(self, X):
        """
        order shape: n, s, d
        """
        diff = X[:, None, :] - self.centers[None, :, :]

        if self.clustering_type == 'gmm':
            # Compute exponential disrance
            weights = np.exp(-0.5*(np.einsum('nsf, sdf, nsd -> ns', diff, self.precisions, diff)))
            # Normalize for determinant of covariance and number of dimensions
            weights = weights/(((2*np.pi)**X.shape[1]*self.cov_dets[None, :])**0.5)
            # Multiply by GMM weight
            weights = weights * self.weights[None, :]
        weights = np.nan_to_num(weights, copy=False, nan=0)
        self.baseline_prob = np.sort(np.ravel(weights))[
            int(self.baseline_percentile*weights.shape[0]*weights.shape[1])]

    def get_models_weight(self, X_avg, dX_dr=None, dX_ds=None, forces=False, stress=False):
        if self.clustering_type == 'kmeans' or self.clustering_type == 'adp':
            return self.get_models_weight_kmeans(X_avg, dX_dr, dX_ds, forces, stress)

    def get_models_weight_kmeans(self, X, dX_dr=None, dX_ds=None, forces=False, stress=False):
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

        weights = {}
        # Distance from center
        diff = X[None, :] - self.centers[:, :]
        # Compute exponential disrance
        proba = self.weights**0.5/np.sum((self.precisions*diff**4), axis = 1)
        # Normalize to get sums up to 1
        # To avoid nans if all clusters have probability 0 or nan
        proba = np.nan_to_num(proba, copy=False, nan=0)
        if not sum(proba) == 0:
            proba_sum = np.sum(proba) + self.baseline_prob
            weights['energy'] = proba/proba_sum
        else:
            weights['energy'] = np.zeros(len(proba))
        if forces or stress:
            if not sum(proba) == 0:
                # TODO this is a very strange hotfix for the derivative when the baseline is nonzero.
                # This is NOT correct but it kinda works
                sum_der = - proba[:, None] / np.sum(proba)**2 + np.eye(len(proba)) / np.sum(proba)
                sum_der *= (1-self.baseline_prob/proba_sum)
        if forces:
            if not sum(proba) == 0:
                single_proba_force = np.einsum('s, mcd, sd -> smc', proba, dX_dr, -4/self.precisions/diff)
                weights['forces'] = np.einsum('smc, ts -> msc', single_proba_force, sum_der)
            else:
                weights['forces'] = np.zeros((dX_dr.shape[0], len(proba), 3))
        if stress:
            if not sum(proba) == 0:
                single_proba_stress = np.einsum('s, cd, sd -> sc', proba, dX_ds[0], -4/self.precisions/diff)
                weights['stress'] = np.einsum('sc, ts -> sc', single_proba_stress, sum_der)
            else:
                weights['stress'] = np.zeros((len(proba), 6))

        return weights
