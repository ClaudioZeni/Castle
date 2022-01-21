import numpy as np
from .linear_potential import train_linear_model
from .clustering import Clustering


class LPEnsamble(object):
    def __init__(self, potentials, representation,
                 mean_peratom_energy, clustering):
        self.potentials = potentials
        self.representation = representation
        self.alphas = np.array([self.potentials[i].weights for i in range(len(self.potentials))])
        self.mean_peratom_energy = mean_peratom_energy
        self.clustering = clustering

    def predict(self, features):
        nat = features.get_nb_atoms_per_frame()
        e_pred = np.zeros(len(features.X))
        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights, der_weighs = self.clustering.get_models_weight(norm_feat, feat.dX_dr, True)
            # First part of the force component, easy
            f_1 = np.einsum("mcd, sd, s -> mc",
                            feat.dX_dr, self.alphas, weights)
            
            # Descriptor multiplied by the derivative of the weights, not sure about the sign
            f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, der_weighs) 

            f_pred[nat_counter:nat_counter+nat[i]] = f_1 - f_2
            nat_counter += nat[i]

            e_pred[i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, weights) + self.mean_peratom_energy*nat[i]

        return e_pred, f_pred

    def predict_energy(self, features):
        nat = features.get_nb_atoms_per_frame()
        e_pred = np.zeros(len(features.X))
        for i in range(len(features.X)):
            feat = features.get_subset([i])
            weights = self.clustering.get_models_weight(feat.X[0] / nat[i, None])
            e_pred[i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, weights) + self.mean_peratom_energy*nat[i]
        return e_pred

    def predict_forces(self, features):
        nat = features.get_nb_atoms_per_frame()
        f_pred = np.zeros(features.dX_dr.shape[:2])
        nat_counter = 0
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights, der_weighs = self.clustering.get_models_weight(norm_feat, feat.dX_dr, True)            
            # First part of the force component, easy
            f_1 = np.einsum("mcd, sd, s -> mc",
                            feat.dX_dr, self.alphas, weights)
            
            # Descriptor multiplied by the derivative of the weights, not sure about the sign
            f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, der_weighs) 

            f_pred[nat_counter:nat_counter+nat[i]] = f_1 - f_2
            nat_counter += nat[i]

        return f_pred

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
    features, noise, e, f, n_clusters=10, clustering_type="kmeans"):
    nat = features.get_nb_atoms_per_frame()
    clustering = Clustering(clustering_type)
    clustering.fit(features.X / nat[:, None], e / nat, n_clusters)

    # Remove atomic energy contributions
    mean_peratom_energy = np.mean(e / nat)
    e_adj = e - nat*mean_peratom_energy

    potentials = {}
    structure_ids = np.arange(len(features))
    for lab in list(set(clustering.labels)):
        mask = clustering.labels == lab
        features_ = features.get_subset(structure_ids[mask])
        fmask = np.zeros(0, dtype="bool")
        for i in np.arange(len(nat)):
            fmask = np.append(fmask, np.array([mask[i]] * nat[i]))
        pot = train_linear_model(features_, noise, e_adj[mask], f[fmask], mean_peratom=False)

        potentials[lab] = pot

    model = LPEnsamble(
        potentials, features.representation, mean_peratom_energy, clustering
    )
    return model


# Old stuff
    
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