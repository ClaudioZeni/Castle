import numpy as np
from sklearn.neighbors import BallTree
from .linear_potential import train_linear_model


def cluster_gvect(X, e, hyper, clustering):
    """Auxiliary function that calls the correct clustering
        algorithm. Options are: kmeans clustering and advanced
        density peaks clustering. If the latter is chosen, the
        adp python package must be installed first.
    Args:
        G (np.array): Descriptor vector for each atom in each structure
        e (np.array): Per-atom energy of each structure
        hyper (float): hyperparameter used for any clsutering
        clustering (str): clustering algorithm to use

    Returns:
        labels (np.array): cluster (int) assigned
                           to each atom in each structure

    """
    if clustering == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=hyper).fit(X)
        labels = kmeans.labels_

    elif clustering == "adp":
        try:
            from adpy import data

            adp = data.Data(X)
            adp.compute_distances(maxk=max(len(X) // 100, 100), njobs=2)
            adp.compute_id()
            adp.compute_optimal_k()
            adp.compute_density_kNN(int(np.median(adp.kstar)))
            print("Selected k is : %i" % (int(np.median(adp.kstar))))
            adp.compute_clustering_optimised(Z=hyper, halo=False)
            labels = adp.labels
        except ModuleNotFoundError:
            print(
                "WARNING: ADP package required to perform adp clustering.\
                   Defaulting to kmeans clustering."
            )
            labels = cluster_gvect(X, e, hyper, "kmeans")

    elif clustering == 'e_gmm':
        from sklearn.mixture import GaussianMixture

        # Resize X based only on global std and energy std, separately
        # So that e is comparable to X but we do not lose information
        # on the magnitude of each component of X.
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]
        e = (e - np.mean(e)) / np.std(e)
        X = np.concatenate((X, e[:, None]), axis=1)

        gmm = GaussianMixture(n_components=hyper).fit(X)
        labels = gmm.predict(X)

    elif clustering == 'gmm':
        from sklearn.mixture import GaussianMixture
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]
        gmm = GaussianMixture(n_components=hyper).fit(X)
        labels = gmm.predict(X)

    return labels


class LPEnsamble(object):
    def __init__(self, potentials, representation,
                 tree, train_labels, n_neighbours):
        self.potentials = potentials
        self.representation = representation
        self.tree = tree
        self.train_labels = train_labels
        self.n_neighbours = n_neighbours

    def predict(self, features):
        nat = features.get_nb_atoms_per_frame()
        dist, idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)

        e_pred = np.zeros(len(features))
        for i in np.arange(len(features)):
            clusters = self.train_labels[idx[i]]
            # If all neighbours are in the same cluster, easy
            if len(np.unique(clusters)) == 1:
                feat = features.get_subset([i])
                e_ = self.potentials[clusters[0]].predict(feat)
            else:
                alphas = np.array([self.potentials[i].weights
                                   for i in clusters])
                weights = np.exp(-dist[i])
                weights /= np.sum(np.exp(-dist[i]))
                e_ = np.einsum("d, ld, l -> ", features.X[i], alphas, weights)

            e_pred[i] = e_

        return e_pred

    def predict_stress(self, features):
        nat = features.get_nb_atoms_per_frame()
        dist, idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)

        s_pred = np.zeros((len(features), 6))
        for i in np.arange(len(features)):
            clusters = self.train_labels[idx[i]]
            # If all neighbours are in the same cluster, easy
            if len(np.unique(clusters)) == 1:
                feat = features.get_subset([i])
                s_ = self.potentials[clusters[0]].predict_stress(feat)
            else:
                alphas = np.array([self.potentials[i].weights
                                   for i in clusters])
                weights = np.exp(-dist[i])
                weights /= np.sum(np.exp(-dist[i]))
                s_ = -np.einsum("cd, ld, l -> c", features.dX_ds[i],
                                alphas, weights)
            s_pred[i] = s_

        return s_pred

    def predict_forces(self, features):
        nat = features.get_nb_atoms_per_frame()
        dist, idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)

        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            clusters = self.train_labels[idx[i]]
            # If all neighbours are in the same cluster, easy
            feat = features.get_subset([i])
            if len(np.unique(clusters)) == 1:
                f_ = self.potentials[clusters[0]].predict_forces(feat)
            else:
                alphas = np.array([self.potentials[i].weights
                                   for i in clusters])
                weights = np.exp(-dist[i])
                weights /= np.sum(np.exp(-dist[i]))
                f_ = -np.einsum("mcd, ld, l -> mc",
                                feat.dX_dr, alphas, weights)

            f_pred[nat_counter:nat_counter+nat[i]] = f_
            nat_counter += nat[i]

        return f_pred


def train_ensamble_linear_model(
    features, noise, e, f, n_neighbours=1, hyper=6, clustering="kmeans"
):
    nat = features.get_nb_atoms_per_frame()

    train_labels = cluster_gvect(features.X / nat[:, None],
                                 e / nat, hyper, clustering)

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
        potentials, features.representation, tree, train_labels, n_neighbours
    )
    return model
