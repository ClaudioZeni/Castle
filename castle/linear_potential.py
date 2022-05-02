import numpy as np
from sklearn.metrics import mean_squared_error
from .representation import progressbar

class LinearPotential(object):
    def __init__(self, representation):
        self.representation = representation

    def fit(self, traj, e_noise=1e-8, f_noise=1e-8, features=None, noise_optimization=False):

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

    def fit_from_features(self, features, e, f=None, e_noise=1e-8, f_noise=1e-8, noise_optimization=False,
                          additional_local_dX_dr=None, additional_f=None):
        self.representation = features.representation
        self.e_noise = e_noise
        self.f_noise = f_noise

        if f is None:
            X_tot = features.X
            Y_tot = e
        else:
            X_tot = np.concatenate(
                (
                    features.X,
                    features.dX_dr[:, 0, :],
                    features.dX_dr[:, 1, :],
                    features.dX_dr[:, 2, :],
                ),
                axis=0)
            Y_tot = np.concatenate((e, f[:, 0], f[:, 1], f[:, 2]), axis=0)

            if additional_local_dX_dr is not None and additional_f is not None:
                X_tot = np.concatenate((X_tot, additional_local_dX_dr[:, 0, :], 
                additional_local_dX_dr[:, 1, :], additional_local_dX_dr[:, 2, :]))
                Y_tot = np.concatenate((Y_tot, additional_f[:, 0], additional_f[:, 1], additional_f[:, 2]), axis=0)

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
        weights, _, _, _ = np.linalg.lstsq(gtg, gY, rcond=None)
        self.weights = weights

    def predict(self, atoms, forces=True, stress=False, features=None):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        if features is None:
            features = self.representation.transform([at])
        return self.predict_from_features(features, forces, stress)

    def predict_from_features(self, features, forces=False, stress=False):
        prediction = {}
        prediction['energy'] = np.dot(features.X, self.weights)
        if forces:
            prediction['forces'] = np.einsum("mcd, d -> mc", features.dX_dr, self.weights)
        if stress:
            prediction['stress'] = np.einsum("ncd, d -> nc", features.dX_ds, self.weights)

        # Dumb hotfix because ASE wants stress to be shape (6) and not (1, 6)
        if prediction['energy'].shape[0] == 1 and stress:
            prediction['stress'] = prediction['stress'][0]
        return prediction

    def predict_local(self, atoms, forces=True, local_features=None):
        at = atoms.copy()
        at.wrap(eps=1e-11)
        if local_features is None:
            local_features = self.representation.transform_local([at])
        return self.predict_from_local_features(local_features, forces)

    def predict_from_local_features(self, local_features, forces=False):
        prediction = {}
        prediction['local_energy'] = np.einsum("md, d -> m", local_features.X[0], self.weights)
        if forces:
            prediction['local_forces'] = np.einsum("nmcd, d -> nmc", local_features.dX_dr[0], self.weights)

        prediction['energy'] = np.sum(prediction['local_energy'], axis = -1)
        prediction['forces'] = np.sum(prediction['local_forces'], axis = 0) - np.sum(prediction['local_forces'], axis = 1)
        return prediction

    def noise_optimization(self, features, e, f, bounds = [1e-10, 1e-2], maxiter=5, kfold=5):
        noises = np.array([bounds, bounds])
        print("Noise Optimization")
        for i in progressbar(np.arange(maxiter)):
            loss = np.zeros((2, 2))
            for j in np.arange(4):
                en = noises[0, j%2]
                fn = noises[1, j//2]
                loss[j%2, j//2] = kfold_validation(features, e, f, en, fn, kfold)
            best_e = np.argmin(loss)%2
            best_f = np.argmin(loss)//2
            noises[0, abs(1-best_e)] = logmean(noises[0, 0], noises[0, 1])
            noises[1, abs(1-best_f)] = logmean(noises[1, 0], noises[1, 1])

        self.e_noise = logmean(noises[0, 0], noises[0, 1])
        self.f_noise = logmean(noises[1, 0], noises[1, 1])
        print(f"Energy noise: {self.e_noise}, Force noise: {self.f_noise}")


def logmean(a, b):
    return np.exp(0.5*np.log(a) + 0.5*np.log(b))


def kfold_selection(n, k):
    ind = np.arange(n)
    np.random.shuffle(ind)
    fold_ind = [ind[int(k_*n/k):int((k_+1)*n/k)] for k_ in np.arange(k)]
    fold_ind[-1] = ind[int(n*(k-1)/k):]
    return fold_ind


def kfold_validation(features, e, f, en, fn, kfold):
    fold_ind = kfold_selection(len(e), kfold)
    loss_e, loss_f = np.zeros(kfold), np.zeros(kfold)
    for k in np.arange(kfold):
        tr_inds = np.concatenate([fold_ind[i] for i in np.delete(np.arange(kfold), k)])
        val_inds = fold_ind[k]
        tr_e = e[tr_inds]
        val_e = e[val_inds]
        val_nat = np.array([features.strides[i+1] - features.strides[i]  for i in val_inds])
        tr_f = []
        [tr_f.extend(f[i[0]:i[1], :]) for i in [[features.strides[i],features.strides[i+1]]  for i in tr_inds]]
        tr_f = np.array(tr_f)
        val_f = []
        [val_f.extend(f[i[0]:i[1], :]) for i in [[features.strides[i],features.strides[i+1]]  for i in val_inds]]
        val_f = np.array(val_f)
        tr_feats = features.get_subset(tr_inds)
        val_feats = features.get_subset(val_inds)
        model = LinearPotential(features.representation)
        model.fit_from_features(tr_feats, tr_e, tr_f, en, fn)
        pred = model.predict_from_features(val_feats, forces=True, stress=False)
        # Small regularization added for the corner case of clusters where all data is almost identical
        loss_e[k] = mean_squared_error(val_e/val_nat, pred['energy']/val_nat, squared=False)/(1e-10+np.std(val_e/val_nat))
        loss_f[k] = mean_squared_error(np.ravel(val_f), np.ravel(pred['forces']), squared=False)/(1e-10+np.std(np.ravel(val_f)))
        if not (loss_f[k] > 0 and loss_f[k] < np.inf):
            print(loss_f[k])
            print(loss_e[k])

    loss = np.mean(loss_e) + np.mean(loss_f)        
    return loss


