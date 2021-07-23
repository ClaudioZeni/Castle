import numpy as np


class LinearPotential(object):
    def __init__(self, weights, representation):
        self.weights = weights
        self.representation = representation

    def predict(self, features):
        e = np.dot(features.X, self.weights)
        return e

    def predict_forces(self, features):
        f = -np.einsum("mcd, d -> mc", features.dX_dr, self.weights)
        return f

    def predict_stress(self, features):
        v = -np.einsum("mcd, d -> mc", features.dX_ds, self.weights)
        return v


def train_linear_model(features, noise, e, f):
    X_tot = np.concatenate(
        (
            features.X,
            -features.dX_dr[:, 0, :],
            -features.dX_dr[:, 1, :],
            -features.dX_dr[:, 2, :],
        ),
        axis=0,
    )
    Y_tot = np.concatenate((e, f[:, 0], f[:, 1], f[:, 2]), axis=0)

    # ftf shape is (S, S)
    gtg = np.einsum("na, nb -> ab", X_tot, X_tot)
    # Calculate fY
    gY = np.einsum("na, n -> a", X_tot, Y_tot)
    # Add regularization
    noise = noise * np.ones(len(gtg))
    gtg[np.diag_indices_from(gtg)] += noise
    weights, _, _, _ = np.linalg.lstsq(gtg, gY, rcond=None)
    model = LinearPotential(weights, features.representation)
    return model
