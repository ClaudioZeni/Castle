import numpy as np
from .linear_potential import train_linear_model
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def optimize_n_clusters(X):
    S = []
    for i in np.arange(2, min(len(X)//10, 10)):
        gmm = GaussianMixture(n_components=i, n_init=3)
        labels = gmm.fit_predict(X)
        S.append(silhouette_score(X, labels, metric='euclidean'))
    nopt = 2 + np.argmax(S*np.arange(1, 1+len(S))**0.5)
    gmm = GaussianMixture(n_components=nopt, n_init=5).fit(X)
    print("Using %i clusters" %(nopt))
    return gmm


def cluster_gvect(X, e, n_clusters='auto', clustering='e_gmm'):
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
    if clustering == 'e_gmm':
        # Resize X based only on global std and energy std, separately
        # So that e is comparable to X but we do not lose information
        # on the magnitude of each component of X.
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]
        e = (e - np.mean(e)) / np.std(e)
        X = np.concatenate((X, e[:, None]), axis=1)
        
        if n_clusters == 'auto':
            gmm = optimize_n_clusters(X)
        else:
            gmm = GaussianMixture(n_components=n_clusters, n_init=5).fit(X)
        labels = gmm.predict(X)

        weights = gmm.weights_
        centers = gmm.means_[:, :-1] * std + mean[None, :]
        precisions = gmm.precisions_[:, :-1, :-1] / std

    elif clustering == 'gmm':
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]

        if n_clusters == 'auto':
            gmm = optimize_n_clusters(X)
        else:
            gmm = GaussianMixture(n_components=n_clusters, n_init=5).fit(X)
        labels = gmm.predict(X)

        weights = gmm.weights_
        centers = gmm.means_ * std + mean[None, :]
        precisions = gmm.precisions_[:, :-1, :-1] / std

    return labels, centers, precisions, weights


class LPEnsamble(object):
    def __init__(self, potentials, representation,
                 centers, precisions, weights, 
                 train_labels, X_training):
        self.potentials = potentials
        self.representation = representation
        self.train_labels = train_labels
        self.X_training = X_training
        self.centers = centers 
        self.precisions = precisions
        self.weights = weights
        self.alphas = np.array([self.potentials[i].weights for i in range(len(self.potentials))])
        self.cov_dets = np.array([1/np.linalg.det(precisions[i]) for i in range(len(weights))])
        

    def predict(self, features):
        nat = features.get_nb_atoms_per_frame()
        e_pred = np.zeros(len(features.X))
        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights, der_weighs = self.get_models_weight(norm_feat, feat.dX_dr, True)

            # First part of the force component, easy
            f_1 = np.einsum("mcd, sd, s -> mc",
                            feat.dX_dr, self.alphas, weights)
            
            # Descriptor multiplied by the derivative of the weights, not sure about the sign
            f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, der_weighs) 

            f_pred[nat_counter:nat_counter+nat[i]] = f_1 - f_2
            nat_counter += nat[i]

            e_pred[i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, weights)

        return e_pred, f_pred

    def predict_energy(self, features):
        nat = features.get_nb_atoms_per_frame()
        e_pred = np.zeros(len(features.X))
        for i in range(len(features.X)):
            feat = features.get_subset([i])
            weights = self.get_models_weight(feat.X[0] / nat[i, None])
            e_pred[i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, weights)
        return e_pred

    def predict_forces(self, features):
        nat = features.get_nb_atoms_per_frame()
        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights, der_weighs = self.get_models_weight(norm_feat, feat.dX_dr, True)            
            # First part of the force component, easy
            f_1 = np.einsum("mcd, sd, s -> mc",
                            feat.dX_dr, self.alphas, weights)
            
            # Descriptor multiplied by the derivative of the weights, not sure about the sign
            f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, der_weighs) 

            f_pred[nat_counter:nat_counter+nat[i]] = f_1 - f_2
            nat_counter += nat[i]

        return f_pred
    
    # def predict_forces(self, features):
    #     nat = features.get_nb_atoms_per_frame()
    #     neigh_dist, neigh_idx = self.tree.query(features.X / nat[:, None],
    #                                 k=self.n_neighbours)
    #     f_pred = np.zeros(features.dX_dr.shape[:2])
    #     nat_counter = 0
    #     for i in np.arange(len(features)):
    #         feat = features.get_subset([i])
    #         f_ = self.predict_forces_single(feat, neigh_dist[i],
    #                                      neigh_idx[i], nat[i])
    #         f_pred[nat_counter:nat_counter+nat[i]] = f_
    #         nat_counter += nat[i]
    #     return f_pred
                                                       
    # def predict_forces_single(self, feat, neigh_dist, neigh_idx, nat):
    #     """ Definitions:
    #     m: number of atoms in configuration
    #     c: cartesian coordinates
    #     d: number of dimensions of descriptor
    #     s: number of nearest neighbours in clustering model
        
    #     Shapes:
        
    #     clusters:   (s)
    #     feat.X:     (1, d)
    #     feat.dX_dr: (m, c, d)
    #     model_weight:    (s)
    #     alphas:     (s, d)
    #     diff_unity_vector: (s, d)
    #     d_weights_d_r: (m, s, c)
    #     d_weights_d_descr: (s, d)
    #     """
    #     clusters = self.train_labels[neigh_idx]
    #     # If all neighbours are in the same cluster, easy
    #     if len(np.unique(clusters)) == 1:
    #         f_ = self.potentials[clusters[0]].predict_forces(feat)

    #     # Else, there are two terms to the total energy derivative
    #     # The first term, f_1 is the derivative of the descirptor multiplied by the model_weights
    #     # The second term, f_2 is the descriptor multiplied by the derivative of the model_weights
    #     else:
    #         # Coefficients for the potentials used
    #         alphas = np.array([self.potentials[i].weights
    #                             for i in clusters])
    #         # Weight of each of the neighbours employed for interpolation, computed via softmax
    #         # of the distances of the descriptor per atom
    #         model_weights = np.exp(-neigh_dist)/np.sum(np.exp(-neigh_dist))

    #         # First part of the force component, easy
    #         f_1 = np.einsum("mcd, sd, s -> mc",
    #                         feat.dX_dr, alphas, model_weights)

    #         # Second part of the force component, complex
    #         # Direction from the descriptor to each nearest neighbour reference descriptor
    #         diff_unity_vector = - (feat.X[0][None, :] / nat - self.X_training[neigh_idx])/neigh_dist[:, None]

    #         # Derivative of weights w.r.t. position of descriptor,
    #         d_weights_d_descr =  np.sum(np.exp(-neigh_dist)[:, None, None]*(diff_unity_vector[:, None, :] - diff_unity_vector[None, :, :]), axis = 0)
    #         d_weights_d_descr /= np.sum(np.exp(-neigh_dist))

    #         # Derivative of weights w.r.t. position of atom.
    #         d_weights_d_r = np.einsum('s, sd, mcd -> msc', 
    #                                  model_weights, d_weights_d_descr, feat.dX_dr)
            
    #         # Descriptor multiplied by the derivative of the weights, not sure about the sign
    #         f_2 = np.einsum("d, sd, msc -> mc", feat.X[0]/nat, alphas, d_weights_d_r) 
    #         f_ = f_1 - f_2

    #     return f_

    def predict_stress(self, features):
        nat = features.get_nb_atoms_per_frame()
        neigh_dist, neigh_idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)
        s_pred = np.zeros((len(features), 6))
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            s_pred[i] = self.predict_stress_single(feat, neigh_dist[i], neigh_idx[i])
        return s_pred

    def predict_stress_single(self, feat, neigh_dist, neigh_idx):
        clusters = self.train_labels[neigh_idx]
        # If all neighbours are in the same cluster, easy
        if len(np.unique(clusters)) == 1:
            s_ = self.potentials[clusters[0]].predict_stress(feat)
        else:
            # TODO
            alphas = np.array([self.potentials[i].weights
                                for i in clusters])
            weights = np.exp(-neigh_dist)/np.sum(np.exp(-neigh_dist))
            s_ = -np.einsum("cd, ld, l -> c", feat.dX_ds[0],
                            alphas, weights)
        return s_

    def get_models_weight(self, X_avg, dX_dr=None, compute_der=False):
        """Compute weight (and derivative) of models for a single structure.
        w.r.t. each atom's position.
        m: number of atoms in configuration
        c: cartesian coordinates
        d: number of dimensions of descriptor
        s: number of models

        order of indexing: s, d, m, c

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
        proba = proba/np.sum(proba)

        if compute_der:
            der_diff = np.einsum('sf, sdf -> sd', diff, self.precisions)
            single_proba_der = np.einsum('s, mcd, sd -> smc', proba, dX_dr, der_diff)
            softmax_der = -proba[None, :]*proba[:, None] + np.diag(proba)
            proba_der = np.einsum('tmc, ts -> msc', single_proba_der, softmax_der)
            return proba, proba_der
        else:
            return proba


def train_ensamble_linear_model(
    features, noise, e, f, n_clusters=10, clustering="e_gmm"
):
    nat = features.get_nb_atoms_per_frame()
    labels, centers, precisions, weights = cluster_gvect(features.X / nat[:, None],
                                 e / nat, n_clusters, clustering)

    potentials = {}
    structure_ids = np.arange(len(features))
    for lab in list(set(labels)):
        mask = labels == lab
        features_ = features.get_subset(structure_ids[mask])
        fmask = np.zeros(0, dtype="bool")
        for i in np.arange(len(nat)):
            fmask = np.append(fmask, np.array([mask[i]] * nat[i]))
        pot = train_linear_model(features_, noise, e[mask], f[fmask])

        potentials[lab] = pot

    model = LPEnsamble(
        potentials, features.representation, centers, precisions, 
        weights, labels, features.X / nat[:, None]
    )
    return model
