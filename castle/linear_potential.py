import numpy as np


class LinearPotential(object):
    def __init__(self):
        pass

    def fit(self, traj, representation, noise=1e-6, 
            features=None, energy_name=None, force_name=None):
        self.representation = representation
        self.noise = noise

        e = np.array([t.info[energy_name] for t in traj])
        if force_name is not None:
            f = []
            [f.extend(t.get_array(force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if features is None:
            features = self.representation.transform(traj)
        self.fit_from_features(self, features, e, f, noise=1e-6)

    def fit_from_features(self, features, noise, e, f=None, mean_peratom=True):
        self.representation = features.representation
        self.noise = noise

        if mean_peratom:
            nat = features.get_nb_atoms_per_frame()
            mean_peratom_energy = np.mean(e / nat)
            e_adj = e - nat*mean_peratom_energy
        else:
            mean_peratom_energy = 0
            e_adj = e

        if f is None:
            X_tot = features.X
            Y_tot = e_adj
        else:
            X_tot = np.concatenate(
                (
                    features.X,
                    features.dX_dr[:, 0, :],
                    features.dX_dr[:, 1, :],
                    features.dX_dr[:, 2, :],
                ),
                axis=0)
            Y_tot = np.concatenate((e_adj, f[:, 0], f[:, 1], f[:, 2]), axis=0)

        # ftf shape is (S, S)
        gtg = np.einsum("na, nb -> ab", X_tot, X_tot)
        # Calculate fY
        gY = np.einsum("na, n -> a", X_tot, Y_tot)
        # Add regularization
        noise = self.noise * np.ones(len(gtg))
        gtg[np.diag_indices_from(gtg)] += noise
        weights, _, _, _ = np.linalg.lstsq(gtg, gY, rcond=None)
        self.weights = weights
        self.mean_peratom_energy = mean_peratom_energy

    def predict(self, atoms, forces=True, stress=False):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        features = self.representation.transform([at])
        e_model = self.predict_energy_from_features(features)
        if forces:
            f_model = self.predict_forces_from_features(features)
        if stress:
            s_model = self.predict_stress_from_features(features)

        if forces and stress:
            return e_model, f_model, s_model
        elif forces:
            return e_model, f_model
        else:
            return e_model

    def predict_energy_from_features(self, features):
        nat = features.get_nb_atoms_per_frame()
        e = np.dot(features.X, self.weights) + nat * self.mean_peratom_energy
        return e

    def predict_forces_from_features(self, features):
        f = np.einsum("mcd, d -> mc", features.dX_dr, self.weights)
        return f

    def predict_stress_from_features(self, features):
        # TODO CHECK IF CORRECT
        v = -np.einsum("mcd, d -> mc", features.dX_ds, self.weights)
        return v

    def predict_from_features(self, features):
        e = self.predict_energy_from_features(features)
        f = self.predict_forces_from_features(features)
        return e, f

