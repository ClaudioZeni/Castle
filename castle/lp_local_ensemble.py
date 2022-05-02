import numpy as np
from .linear_potential import LinearPotential
from .local_clustering import LocalClustering

class LPLocalEnsemble(object):
    def __init__(self, representation, clustering_type='kmeans', n_clusters='auto',
                 baseline_calculator=None, baseline_percentile=0):
        self.representation = representation
        self.clustering_type = clustering_type
        self.baseline_calculator = baseline_calculator
        if self.baseline_calculator is not None and baseline_percentile==0:
            baseline_percentile = 0
            print("""WARNING: Baseline has been given but baseline_percentile is set to 0.
        This would cause baseline to be ignored.
        Setting baseline_percentile to 0""")
        self.clustering = LocalClustering(self.clustering_type, baseline_percentile)
        self.n_clusters = n_clusters
        self.e_b = None
        self.f_b = None

    def fit(self, traj, e_noise=1e-8, f_noise=1e-8, local_features=None, noise_optimization=False):
        if self.baseline_calculator:
            self.compute_baseline_predictions(traj)
        e = np.array([t.info[self.representation.energy_name] for t in traj])
        if self.representation.force_name is not None:
            f = []
            [f.extend(t.get_array(self.representation.force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if local_features is None:
            local_features = self.representation.transform_local(traj, verbose=True)
        self.fit_from_local_features(local_features, e, f, e_noise, f_noise, noise_optimization)

    def fit_from_local_features(self, local_features, e, f, e_noise=1e-8, f_noise=1e-8, noise_optimization=False):
        self.e_noise = e_noise
        self.f_noise = f_noise
        self.representation = local_features.representation
        nsp = len(self.representation.species)
        nat = local_features.get_nb_atoms_per_frame()
        flat_X  = np.concatenate([local_features.X[i] for i in range(len(local_features.X))])
        self.clustering.fit(flat_X[:, :-nsp], self.n_clusters)

        global_features = self.representation.local_to_global_features(local_features)

        if self.e_b is not None:
            e -= self.e_b
            f -= self.f_b
        flat_dX_dr = np.concatenate([np.sum(local_features.dX_dr[i], axis = 0) - 
                                     np.sum(local_features.dX_dr[i], axis = 1) 
                                     for i in range(len(local_features.dX_dr))])

        potentials = {}
        structure_ids = np.arange(len(local_features))
        # Global mask is constructed for each cluster using the structures that have 
        # "a lot of" local environment belonging to that cluster. Always contain at least one structure
        global_mask = get_structure_indexes_from_local_labels(self.clustering.labels, local_features.strides)

        for lab in list(set(self.clustering.labels)):
            # Local mask is directly derived from clustering 
            local_mask = self.clustering.labels == lab
            features_ = global_features.get_subset(structure_ids[global_mask[lab]])
            f_global_mask = [np.arange(local_features.strides[i], local_features.strides[i+1]) for i in global_mask[lab]]
            f_global_mask = np.ravel(f_global_mask)
            pot = LinearPotential(self.representation)
            pot.fit_from_features(features_, e[global_mask[lab]], f[f_global_mask, :], self.e_noise, self.f_noise, 
                                  noise_optimization, flat_dX_dr[local_mask], f[local_mask])
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

    def predict(self, atoms, forces=True, stress=False, local_features=None):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        if local_features is None:
            local_features = self.representation.transform_local([at])
        
        prediction = self.predict_from_local_features(local_features, forces, stress)
        if self.baseline_calculator is not None:
            at.set_calculator(self.baseline_calculator)
            prediction['energy'] += at.get_potential_energy()
            if forces:
                prediction['forces'] += at.get_forces()
            if stress:
                prediction['stress'] += at.get_stresses()
        return prediction

    def predict_from_local_features(self, local_features, forces=False, stress=False):
        prediction = {}
        nat = local_features.get_nb_atoms_per_frame()
        nsp = len(self.representation.species)
        prediction['energy'] = np.zeros(len(local_features.X))
        if forces:
            prediction['forces'] = np.zeros(local_features.dX_dr.shape[:2])
        if stress:
            prediction['stress'] = np.zeros((len(local_features.X), 6))
        nat_counter = 0
        for i in np.arange(len(local_features)):
            feat = local_features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights = self.clustering.get_models_weight(norm_feat[..., :-nsp], feat.dX_dr[..., :-nsp], 
                                                        feat.dX_ds[..., :-nsp], forces=forces, stress=stress)
            if forces:
                # First part of the force component, easy
                f_1 = np.einsum("mcd, sd, s -> mc",
                                feat.dX_dr, self.alphas, weights['energy'])
                
                # Descriptor multiplied by the derivative of the weights, not sure about the sign
                f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, weights['forces']) 

                prediction['forces'][nat_counter:nat_counter+nat[i]] = f_1 - f_2

            if stress:
                # First part of the force component, easy   
                s_1 = np.einsum("cd, sd, s -> c",
                                feat.dX_ds[0], self.alphas, weights['energy'])
                
                # Descriptor multiplied by the derivative of the weights, not sure about the sign
                s_2 = np.einsum("d, sd, sc -> c", norm_feat, self.alphas, weights['stress']) 

                prediction['stress'][i] = s_1 - s_2

            nat_counter += nat[i]

            prediction['energy'][i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, 
                                                 weights['energy'])

        # Dumb hotfix because ASE wants stress to be shape (6) and not (1, 6)
        if prediction['energy'].shape[0] == 1 and stress:
            prediction['stress'] = prediction['stress'][0]
        return prediction


def get_structure_indexes_from_local_labels(labels, strides):
    # Assign structures to clusters based on how many local environments of that structure belong to that cluster.
    # This algorithm assigns the same number of structures to each cluster, and at least always 1.
    # This algorithm does NOT assign all structures to at least one cluster, so some energy data might not be used.
    clusters = list(set(labels))
    n_clusters = len(clusters)
    n_structures = len(strides) - 1
    n_data_per_cluster = 1 + n_structures // n_clusters
    grouped_labels = [labels[strides[i]:strides[i+1]] for i in range(n_structures)]
    probs = np.zeros((n_structures, n_clusters))
    for i in np.arange(n_structures):
        probs[i] = np.array([sum(grouped_labels[i] == c) for c in clusters])
    probs = probs/np.sum(probs, axis = 0)
    indexes = {c:np.random.choice(np.arange(len(probs), dtype = 'int'), n_data_per_cluster,
                       p = probs[:, i], replace=False) for i, c in enumerate(clusters)}

    return indexes  # Has shape N Clusters x N structures per cluster

