import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from .linear_potential import LinearPotential
from .clustering import Clustering
from ase.calculators.calculator import Calculator


class LPEnsamble(object):
    def __init__(self, clustering_type='kmeans', n_clusters='auto',
                 baseline_calculator=None, baseline_percentile=0):
        self.clustering_type = clustering_type
        self.baseline_calculator = baseline_calculator
        if self.baseline_calculator is not None and baseline_percentile==0:
            baseline_percentile = 0.1
            print("""WARNING: Baseline has been given but baseline_percentile is set to 0.
        This would cause baseline to be ignored.
        Setting baseline_percentile to 0.1""")
        self.clustering = Clustering(self.clustering_type, baseline_percentile)
        self.n_clusters = n_clusters
        self.e_b = None
        self.f_b = None

    def fit(self, traj, representation, noise=1e-6, features=None):
        self.representation = representation
        self.noise = noise
        if self.baseline_calculator:
            self.compute_baseline_predictions(traj)

        e = np.array([t.info[representation.energy_name] for t in traj])
        if representation.force_name is not None:
            f = []
            [f.extend(t.get_array(representation.force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if features is None:
            features = self.representation.transform(traj)
        features = self.representation.transform(traj)
        self.fit_from_features(features, e, f, noise=1e-6)

    def fit_from_features(self, features, e, f, noise=1e-6):
        self.noise = noise
        self.representation = features.representation
        nat = features.get_nb_atoms_per_frame()
        self.clustering.fit(features.X / nat[:, None], e / nat, self.n_clusters)

        if self.e_b is not None:
            e -= self.e_b
            f -= self.f_b

        # Remove atomic energy contributions
        self.mean_peratom_energy = np.mean(e / nat)
        e_adj = e - nat*self.mean_peratom_energy

        potentials = {}
        structure_ids = np.arange(len(features))
        for lab in list(set(self.clustering.labels)):
            mask = self.clustering.labels == lab
            features_ = features.get_subset(structure_ids[mask])
            fmask = np.zeros(0, dtype="bool")
            for i in np.arange(len(nat)):
                fmask = np.append(fmask, np.array([mask[i]] * nat[i]))
            pot = LinearPotential()
            pot.fit_from_features(features_, self.noise, e_adj[mask], f[fmask], mean_peratom=False)

            potentials[lab] = pot

        self.potentials = potentials
        self.alphas = np.array([self.potentials[i].weights for i in range(len(self.potentials))])

    def compute_baseline_predictions(self, traj):
        e_b = []
        f_b = []
        for t in traj:
            t.set_calculator(self.baseline_calculator)
            e_b.append(t.get_potential_energy())
            f_b.extend(t.get_forces())
        e_b = np.array(e_b)
        f_b = np.array(f_b)
        self.e_b = e_b
        self.f_b = f_b

    def predict(self, atoms, forces=True, stress=False, features=None):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        manager = [at]
        if features is None:
            features = self.representation.transform(manager)
        if forces:
            e_model, f_model = self.predict_from_features(features)
        else:
            e_model = self.predict_energy_from_features(features)
        if stress:
            s_model = self.predict_stress_from_features(features)

        if self.baseline_calculator is not None:
            at.set_calculator(self.baseline_calculator)
            e_model += at.get_potential_energy()
            if forces:
                f_model += at.get_forces()
            if stress:
                s_model += at.get_stresses()

        if forces and stress:
            return e_model, f_model, s_model
        elif forces:
            return e_model, f_model
        else:
            return e_model

    def predict_from_features(self, features):
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

    def predict_energy_from_features(self, features):
        nat = features.get_nb_atoms_per_frame()
        e_pred = np.zeros(len(features.X))
        for i in range(len(features.X)):
            feat = features.get_subset([i])
            weights = self.clustering.get_models_weight(feat.X[0] / nat[i, None])
            e_pred[i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, weights) + self.mean_peratom_energy*nat[i]
        return e_pred

    def predict_forces_from_features(self, features):
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

    def predict_stress_from_features(self, features):
        nat = features.get_nb_atoms_per_frame()
        neigh_dist, neigh_idx = self.tree.query(features.X / nat[:, None],
                                    k=self.n_neighbours)
        s_pred = np.zeros((len(features), 6))
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            s_pred[i] = self.predict_stress_single(feat, neigh_dist[i], neigh_idx[i])
        return s_pred

    def predict_stress_single_from_features(self, feat, neigh_dist, neigh_idx):
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