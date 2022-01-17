import numpy as np


class LinearPotential(object):
    def __init__(self, weights, representation, mean_peratom_energy):
        self.weights = weights
        self.representation = representation
        self.mean_peratom_energy = mean_peratom_energy

    def predict_energy(self, features):
        nat = features.get_nb_atoms_per_frame()
        e = np.dot(features.X, self.weights) + nat * self.mean_peratom_energy
        return e

    def predict_forces(self, features):
        f = np.einsum("mcd, d -> mc", features.dX_dr, self.weights)
        return f

    def predict_stress(self, features):
        # TODO CHECK IF CORRECT
        v = -np.einsum("mcd, d -> mc", features.dX_ds, self.weights)
        return v

    def predict(self, features):
        e = self.predict_energy(features)
        f = self.predict_forces(features)
        return e, f

def train_linear_model(features, noise, e, f=None):
    nat = features.get_nb_atoms_per_frame()
    mean_peratom_energy = np.mean(e / nat)
    e_adj = e - nat*mean_peratom_energy

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
            axis=0,
        )
        Y_tot = np.concatenate((e_adj, f[:, 0], f[:, 1], f[:, 2]), axis=0)

    # ftf shape is (S, S)
    gtg = np.einsum("na, nb -> ab", X_tot, X_tot)
    # Calculate fY
    gY = np.einsum("na, n -> a", X_tot, Y_tot)
    # Add regularization
    noise = noise * np.ones(len(gtg))
    gtg[np.diag_indices_from(gtg)] += noise
    weights, _, _, _ = np.linalg.lstsq(gtg, gY, rcond=None)
    model = LinearPotential(weights, features.representation, mean_peratom_energy)
    return model
