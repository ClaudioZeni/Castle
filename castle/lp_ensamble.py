import numpy as np
from sklearn.neighbors import BallTree
from .linear_potential import train_linear_model
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
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
        G (np.array): Descriptor vector for each atom in each structure
        e (np.array): Per-atom energy of each structure
        n_clusters (float): Number of clusters
        clustering (str): Clustering algorithm to use

    Returns:
        labels (np.array): cluster (int) assigned
                           to each atom in each structure

    """
    if clustering == "kmeans":

        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        labels = kmeans.labels_

    elif clustering == 'e_gmm':
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

    elif clustering == 'gmm':
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]

        if n_clusters == 'auto':
            gmm = optimize_n_clusters(X)
        else:
            gmm = GaussianMixture(n_components=n_clusters, n_init=5).fit(X)
        labels = gmm.predict(X)

    # elif clustering == "dada":
    #     try:
    #         from dadapy import data
    #         adp = data.Data(X)
    #         adp.compute_distances(maxk=max(len(X) // 100, 100))
    #         adp.compute_id_2NN()
    #         # adp.compute_density_kNN(int(np.median(adp.kstar)))
    #         adp.compute_density_kstarNN()
    #         print("Selected k is : %i" % (int(np.median(adp.kstar))))
    #         adp.compute_clustering_optimised(halo=False)
    #         labels = adp.labels
    #     except ModuleNotFoundError:
    #         print(
    #             "WARNING: DADApy package required to perform dada clustering.\
    #                Defaulting to kmeans clustering."
    #         )
    #         labels = cluster_gvect(X, e, n_clusters, "kmeans")

    return labels


class LPEnsamble(object):
    def __init__(self, potentials, representation,
                 tree, train_labels, n_neighbours, X_training):
        self.potentials = potentials
        self.representation = representation
        self.tree = tree
        self.train_labels = train_labels
        self.n_neighbours = n_neighbours
        self.X_training = X_training

    def predict_energy(self, features):
        nat = features.get_nb_atoms_per_frame()
        neigh_dist, neigh_idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)
        e_pred = np.zeros(len(features))
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            e_pred[i] = self.predict_energy_single(feat, neigh_dist[i],
                                                   neigh_idx[i], nat[i])
        return e_pred

    def predict_energy_single(self, feat, neigh_dist, neigh_idx, nat):
        clusters = self.train_labels[neigh_idx]
        # If all neighbours are in the same cluster, easy
        if len(np.unique(clusters)) == 1:
            e_ = self.potentials[clusters[0]].predict_energy(feat)
        else:
            alphas = np.array([self.potentials[i].weights
                                for i in clusters])
            weights = np.exp(-neigh_dist)/np.sum(np.exp(-neigh_dist))
            e_ = np.einsum("d, ld, l -> ", feat.X[0], alphas, weights)
        return e_

    def predict_forces(self, features):
        nat = features.get_nb_atoms_per_frame()
        neigh_dist, neigh_idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)
        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            f_ = self.predict_forces_single(feat, neigh_dist[i],
                                         neigh_idx[i], nat[i])
            f_pred[nat_counter:nat_counter+nat[i]] = f_
            nat_counter += nat[i]
        return f_pred

    def predict_forces_single(self, feat, neigh_dist, neigh_idx, nat):
        clusters = self.train_labels[neigh_idx]
        # If all neighbours are in the same cluster, easy
        if len(np.unique(clusters)) == 1:
            f_ = self.potentials[clusters[0]].predict_forces(feat)
        else:
            alphas = np.array([self.potentials[i].weights
                                for i in clusters])
            weights = np.exp(-neigh_dist)/np.sum(np.exp(-neigh_dist))
            diff_unity_vector = (feat.X[0][None, :] / nat - self.X_training[neigh_idx])/neigh_dist[:, None]
            
            # TODO
            f_1 = np.einsum("mcd, ld, l -> mc",
                            feat.dX_dr, alphas, weights)

            # p1 = 1 - np.einsum("d, ld, l-> ld", feat.X[0], diff_unity_vector, (1-weights))
            # p2 = np.einsum("ld, l -> d", p1, weights)
            # f_ = np.einsum("ld, mcd, d -> mc", alphas, feat.dX_dr, p2)

            f_ = f_1 #+ f_2
        return f_

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


def train_ensamble_linear_model(
    features, noise, e, f, n_neighbours=1, n_clusters=10, clustering="kmeans"
):
    nat = features.get_nb_atoms_per_frame()

    train_labels = cluster_gvect(features.X / nat[:, None],
                                 e / nat, n_clusters, clustering)

    potentials = {}
    structure_ids = np.arange(len(features))
    for lab in list(set(train_labels)):
        mask = train_labels == lab
        features_ = features.get_subset(structure_ids[mask])
        fmask = np.zeros(0, dtype="bool")
        for i in np.arange(len(nat)):
            fmask = np.append(fmask, np.array([mask[i]] * nat[i]))

        pot = train_linear_model(features_, noise, e[mask], f[fmask])

        potentials[lab] = pot

    # Construct reference ball tree
    tree = BallTree(features.X / nat[:, None], leaf_size=2)
    model = LPEnsamble(
        potentials, features.representation, tree, train_labels, n_neighbours, features.X / nat[:, None]
    )
    return model
