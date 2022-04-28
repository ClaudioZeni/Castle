import numpy as np


class GlobalFeatures(object):
    def __init__(self, representation, X, dX_dr, dX_ds, strides, species):
        self.representation = representation
        self.X = X
        self.dX_dr = dX_dr
        self.dX_ds = dX_ds
        self.strides = strides
        self.species = species

    def get_subset(self, ids):
        dX_dr_n = []
        strides = [0]
        for i_frame in ids:
            st, nd = self.strides[i_frame], self.strides[i_frame + 1]
            dX_dr_n.append(self.dX_dr[st:nd])
            strides.append(nd - st)

        obj = GlobalFeatures(
            self.representation,
            self.X[ids],
            np.vstack(dX_dr_n),
            self.dX_ds[ids],
            np.cumsum(strides),
            self.species,
        )
        return obj

    def __len__(self):
        return self.X.shape[0]

    def get_nb_atoms_per_frame(self):
        nat = []
        for st, nd in zip(self.strides[:-1], self.strides[1:]):
            nat.append(nd - st)
        return np.array(nat)

class LocalFeatures(object):
    def __init__(self, representation, X, dX_dr, dX_ds, strides, species):
        self.representation = representation
        self.X = X
        self.dX_dr = dX_dr
        self.dX_ds = dX_ds
        self.strides = strides
        self.species = species

    def __len__(self):
        return len(self.X)

    def get_nb_atoms_per_frame(self):
        nat = []
        for st, nd in zip(self.strides[:-1], self.strides[1:]):
            nat.append(nd - st)
        return np.array(nat)

    def get_subset(self, ids):

        strides = [0]
        for i_frame in ids:
            st, nd = self.strides[i_frame], self.strides[i_frame + 1]
            strides.append(nd - st)
        obj = LocalFeatures(
            self.representation,
            self.X[ids],
            self.dX_dr[ids],
            self.dX_ds[ids],
            np.cumsum(strides),
            self.species,
        )
        return obj