import numpy as np
from .representation import progressbar

class ExpandedPotential(object):
    def __init__(self, representation, D, activation='sigmoid'):
        self.representation = representation
        self.D = D
        self.projector = None
        self.mean = None
        self.activation = activation

    def create_projector(self, X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        self.random_matrix = np.random.normal(size = (X.shape[1], self.D)) / X.shape[1]**0.5
        self.projector = self.random_matrix / self.std[:, None]

    def project_x(self, X):
        return np.einsum('ns, sd -> nd', X - self.mean[None, :], self.projector)

    def project_dx_dr(self, dX_dr):
        return np.einsum('mcs, sd -> mcd', dX_dr, self.projector/np.std(self.projector, axis =0)[None, :])

    def project_dx_ds(self, dX_ds):
        return np.einsum('ncs, sd -> ncd', dX_ds, self.projector)

    def sigmoid(self, features, X_t, dX_dr_t=None, dX_ds_t=None, forces=False, stress=False):
        X_p = 1/(1 + np.exp(-X_t))
        dX_dr_p = np.zeros_like(dX_dr_t)
        dX_ds_p = np.zeros_like(dX_ds_t)
        if forces:
            for i in np.arange(len(X_t)):
                start = features.strides[i]
                end = features.strides[i+1]
                dX_dr_p[start:end, :, :] =  (X_p[i] * (1 - X_p[i]))[None, None, :] * dX_dr_t[start:end] 
        if stress:
            dX_ds_p = np.einsum('nd, nd, ncd -> ncd', X_p, (1-X_p), dX_ds_t)
        return X_p, dX_dr_p, dX_ds_p

    def expand_basis(self, features, forces=False, stress=False):
        X_t = self.project_x(features.X)
        dX_dr_t = None
        dX_ds_t = None
        if forces:
            dX_dr_t = self.project_dx_dr(features.dX_dr)
        if stress:
            dX_ds_t = self.project_dx_ds(features.dX_ds)

        if self.activation == 'sigmoid':
            return self.sigmoid(features, X_t, dX_dr_t, dX_ds_t, forces=forces, stress=stress)

    def fit(self, traj, e_noise=1e-8, f_noise=1e-8, features=None, 
            noise_optimization=False, iterations=1, kfold=5):

        e = np.array([t.info[self.representation.energy_name] for t in traj])
        if self.representation.force_name is not None:
            f = []
            [f.extend(t.get_array(self.representation.force_name)) for t in traj]
            f = np.array(f)
        else:
            f = None
        if features is None:
            features = self.representation.transform(traj, verbose=True)
            
        self.fit_from_features(features, e, f, e_noise, f_noise, noise_optimization, iterations, kfold)

    def fit_from_features(self, features, e, f=None, e_noise=1e-8, f_noise=1e-8, 
                          noise_optimization=False, iterations=1, kfold=5):
        self.representation = features.representation
        self.e_noise = e_noise
        self.f_noise = f_noise
        
        if self.projector is None:
            self.create_projector(features.X)

        if iterations > 1:
            self.choose_random_projection(features, e, f, e_noise, 
                                          f_noise, iterations, kfold)
            
        X_p, dX_dr_p, dX_ds_p = self.expand_basis(features, forces=bool(f.any()))
        if f is None:
            X_tot = X_p
            Y_tot = e
        else:
            X_tot = np.concatenate(
                (
                    np.hstack(((features.X - self.mean[None, :]) / self.std[None, :], X_p)),
                    np.hstack((features.dX_dr[:, 0, :], dX_dr_p[:, 0, :])),
                    np.hstack((features.dX_dr[:, 1, :], dX_dr_p[:, 1, :])),
                    np.hstack((features.dX_dr[:, 2, :], dX_dr_p[:, 2, :])),
                ),
                axis=0)
            Y_tot = np.concatenate((e, f[:, 0], f[:, 1], f[:, 2]), axis=0)

        # ftf shape is (S, S)
        gtg = np.einsum("na, nb -> ab", X_tot, X_tot)
        # Calculate fY
        gY = np.einsum("na, n -> a", X_tot, Y_tot)
        # Add regularization
        if noise_optimization:
            self.noise_optimization(features, e, f)
        noise = self.e_noise*np.ones(len(gtg))
        noise[len(e):] = self.f_noise
        gtg[np.diag_indices_from(gtg)] += noise
        alpha, _, _, _ = np.linalg.lstsq(gtg, gY, rcond=None)
        self.alpha = alpha

    def predict(self, atoms, forces=True, stress=False, features=None):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        if features is None:
            features = self.representation.transform([at])
        return self.predict_from_features(features, forces, stress)

    def predict_from_features(self, features, forces=False, stress=False):
        X_p, dX_dr_p, dX_ds_p = self.expand_basis(features, forces=forces, stress=stress)
        prediction = {}
        prediction['energy'] = np.dot(np.hstack(((features.X - self.mean[None, :]) / self.std[None, :], X_p)), self.alpha)
        if forces:
            prediction['forces'] = np.einsum("mcd, d -> mc", np.concatenate(
                (features.dX_dr, dX_dr_p), axis=-1), self.alpha)
        if stress:
            prediction['stress'] = np.einsum("ncd, d -> nc", np.concatenate(
                (features.dX_ds, dX_ds_p), axis=-1), self.alpha)

        # Dumb hotfix because ASE wants stress to be shape (6) and not (1, 6)
        if prediction['energy'].shape[0] == 1 and stress:
            prediction['stress'] = prediction['stress'][0]
        return prediction

    def choose_random_projection(self, features, e, f=None, e_noise=1e-8, f_noise=1e-8, iterations=10, kfold=5):
        best_loss = np.inf

        l = len(features.X)
        ind = np.random.choice(np.arange(l), l, replace=False)
        for i in progressbar(np.arange(iterations), prefix = "Choosing random projection"):
            self.create_projector(features.X)
            random_matrix = self.random_matrix
            force_rmse = []
            energy_rmse = []
            for k in np.arange(kfold):
                (tr_features, val_features, e_tr, e_val, f_tr, 
                 f_val, nat_tr, nat_val) = get_subsets(features, e, f, ind, kfold, k)

                self.fit_from_features(tr_features, e_tr, f_tr, e_noise, f_noise)
                prediction = self.predict_from_features(val_features, forces=True)
                energy_rmse.append(np.mean(((prediction['energy'] - e_val)/nat_val)**2)**0.5)
                force_rmse.append(np.mean((prediction['forces'] - f_val)**2)**0.5)

            this_loss = np.mean(energy_rmse/np.std(e)) + np.mean(force_rmse/np.mean(np.sum(f, axis = -1)**2)**0.5)

            if this_loss < best_loss:
                best_matrix = random_matrix

        self.random_matrix = best_matrix
        self.mean = np.mean(features.X, axis = 0)
        self.projector = self.random_matrix / np.std(features.X, axis = 0)[:, None]


def get_subsets(features, e, f, ind, kfold, k):
    l = len(features.X)
    val_ind = ind[int(k*l/kfold):int((k+1)*l/kfold)]
    tr_ind = ind[~np.in1d(ind,val_ind)]
    tr_features = features.get_subset(tr_ind)
    val_features = features.get_subset(val_ind)
    e_tr = e[tr_ind]
    e_val = e[val_ind]
    f_tr = []
    for i in np.arange(len(tr_features.X)):
        f_tr.extend(f[tr_features.strides[i]:tr_features.strides[i+1]])
    f_tr = np.array(f_tr)

    f_val = []
    for i in np.arange(len(val_features.X)):
        f_val.extend(f[val_features.strides[i]:val_features.strides[i+1]])
    f_val = np.array(f_val)

    nat_val = val_features.get_nb_atoms_per_frame()
    nat_tr = val_features.get_nb_atoms_per_frame()

    return tr_features, val_features, e_tr, e_val, f_tr, f_val, nat_tr, nat_val