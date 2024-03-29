import numpy as np
from .linear_potential import LinearPotential
from .clustering import Clustering


class LPEnsemble(object):
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
        self.clustering = Clustering(self.clustering_type, baseline_percentile)
        self.n_clusters = n_clusters
        self.e_b = None
        self.f_b = None

    def fit(self, traj, e_noise=1e-8, f_noise=1e-8, features=None, noise_optimization=False):
        if self.baseline_calculator:
            self.compute_baseline_predictions(traj)

        e = np.array([t.info[self.representation.energy_name] for t in traj])
        if self.representation.force_name is not None:
            f = []
            [f.extend(t.get_array(self.representation.force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if features is None:
            features = self.representation.transform(traj, verbose=True)
        self.fit_from_features(features, e, f, e_noise, f_noise, noise_optimization)

    def fit_from_features(self, features, e, f, e_noise=1e-8, f_noise=1e-8, noise_optimization=False):
        self.e_noise = e_noise
        self.f_noise = f_noise
        self.representation = features.representation
        nat = features.get_nb_atoms_per_frame()
        nsp = len(self.representation.species)
        self.clustering.fit(features.X[:, :-nsp] / nat[:, None], e / nat, self.n_clusters)

        if self.e_b is not None:
            e -= self.e_b
            f -= self.f_b

        potentials = {}
        structure_ids = np.arange(len(features))
        for lab in list(set(self.clustering.labels)):
            mask = self.clustering.labels == lab
            features_ = features.get_subset(structure_ids[mask])
            fmask = np.zeros(0, dtype="bool")
            for i in np.arange(len(nat)):
                fmask = np.append(fmask, np.array([mask[i]] * nat[i]))
            pot = LinearPotential(self.representation)
            pot.fit_from_features(features_, e[mask], f[fmask], self.e_noise, self.f_noise, noise_optimization)

            potentials[lab] = pot

        self.potentials = potentials
        self.alphas = np.array([self.potentials[i].alpha for i in range(len(self.potentials))])

    def update(self, traj, features=None):
        if self.baseline_calculator:
            # TODO: add baseline handling
            pass

        e = np.array([t.info[self.representation.energy_name] for t in traj])
        if self.representation.force_name is not None:
            f = []
            [f.extend(t.get_array(self.representation.force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if features is None:
            features = self.representation.transform(traj, verbose=True)
        self.update_from_features(features, e, f)

    def update_from_features(self, features, e, f=None):
        nat = features.get_nb_atoms_per_frame()
        nsp = len(self.representation.species)
        labels = np.empty(len(features), dtype='int')
        for i, X in enumerate(features.X):
            weights = self.clustering.get_models_weight(X[:-nsp] / nat[i])['energy']
            labels[i] = np.argmax(weights)

        if self.e_b is not None:
            # TODO: add baseline handling
            pass

        structure_ids = np.arange(len(features))
        for lab in list(set(labels)):
            mask = labels == lab
            features_ = features.get_subset(structure_ids[mask])
            fmask = np.zeros(0, dtype="bool")
            for i in np.arange(len(nat)):
                fmask = np.append(fmask, np.array([mask[i]] * nat[i]))
            self.potentials[lab].update_from_features(features_, e[mask], f[fmask])
        self.alphas = np.array([self.potentials[i].alpha for i in range(len(self.potentials))])

    def compute_baseline_predictions(self, traj):
        print("Computing Baseline Predictions")
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

    def predict(self, atoms, forces=True, stress=False, features=None, debug_mode=False):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        if features is None:
            features = self.representation.transform([at])

        if debug_mode:
            prediction = self.predict_from_features_debug_mode(features, forces, stress)
        else:
            prediction = self.predict_from_features(features, forces, stress)

        if self.baseline_calculator is not None:
            at.set_calculator(self.baseline_calculator)
            prediction['energy'] += at.get_potential_energy()
            if forces:
                prediction['forces'] += at.get_forces()
            if stress:
                prediction['stress'] += at.get_stresses()
        return prediction

    def predict_from_features(self, features, forces=False, stress=False):
        prediction = {}
        nat = features.get_nb_atoms_per_frame()
        nsp = len(self.representation.species)
        prediction['energy'] = np.zeros(len(features.X))
        if forces:
            prediction['forces'] = np.zeros(features.dX_dr.shape[:2])
        if stress:
            prediction['stress'] = np.zeros((len(features.X), 6))
        nat_counter = 0

        self.predicted_weights = np.zeros((len(features), self.n_clusters))
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights = self.clustering.get_models_weight(norm_feat[..., :-nsp], feat.dX_dr[..., :-nsp], 
                                                        feat.dX_ds[..., :-nsp], forces=forces, stress=stress)
            self.predicted_weights[i] = weights['energy']
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


    def predict_from_features_debug_mode(self, features, forces=False, stress=False):
        prediction = {}
        prediction_breakdown = {}
        nat = features.get_nb_atoms_per_frame()
        nsp = len(self.representation.species)
        prediction_breakdown['energy'] = np.zeros((self.n_clusters, len(features.X)))
        prediction['energy'] = np.zeros(len(features.X))

        if forces:
            prediction['forces'] = np.zeros(features.dX_dr.shape[:2])
            prediction_breakdown['forces'] = np.zeros((self.n_clusters, features.dX_dr.shape[0], features.dX_dr.shape[1]))
        if stress:
            prediction['stress'] = np.zeros((len(features.X), 6))
            prediction_breakdown['stress'] = np.zeros((self.n_clusters, len(features.X), 6))

        nat_counter = 0

        all_weights = np.zeros((len(features), self.n_clusters))
        for i in np.arange(len(features)):
            feat = features.get_subset([i])
            norm_feat = feat.X[0] / nat[i]
            weights = self.clustering.get_models_weight(norm_feat[..., :-nsp], feat.dX_dr[..., :-nsp], 
                                                        feat.dX_ds[..., :-nsp], forces=forces, stress=stress)
            self.predicted_weights[i] = weights['energy']
            if forces:
                # First part of the force component, easy
                f_1 = np.einsum("mcd, sd, s -> mc",
                                feat.dX_dr, self.alphas, weights['energy'])
                
                # Descriptor multiplied by the derivative of the weights, not sure about the sign
                f_2 = np.einsum("d, sd, msc -> mc", norm_feat, self.alphas, weights['forces']) 

                prediction['forces'][nat_counter:nat_counter+nat[i]] = f_1 - f_2
                prediction_breakdown['forces'][:, nat_counter:nat_counter+nat[i]] = np.einsum("mcd, sd -> smc", feat.dX_dr, self.alphas)
            
            if stress:
                # First part of the force component, easy   
                s_1 = np.einsum("cd, sd, s -> c",
                                feat.dX_ds[0], self.alphas, weights['energy'])
                
                # Descriptor multiplied by the derivative of the weights, not sure about the sign
                s_2 = np.einsum("d, sd, sc -> c", norm_feat, self.alphas, weights['stress']) 

                prediction['stress'][i] = s_1 - s_2
                prediction_breakdown['stress'][:, i] = np.einsum("cd, sd -> sc", feat.dX_ds[0], self.alphas)

            nat_counter += nat[i]

            prediction['energy'][i] = np.einsum("d, ld, l -> ", feat.X[0], self.alphas, 
                                                 weights['energy'])

            prediction_breakdown['energy'][:, i] = np.einsum('d, ld-> l', feat.X[0], self.alphas)


        # Dumb hotfix because ASE wants stress to be shape (6) and not (1, 6)
        if prediction['energy'].shape[0] == 1 and stress:
            prediction['stress'] = prediction['stress'][0]
            
        return prediction, prediction_breakdown, all_weights
