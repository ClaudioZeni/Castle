import sys
import numpy as np

from .ace_interface import (descriptors_from_frame,
                            descriptors_from_frame_no_forces,
                            local_descriptors_from_frame,
                            local_descriptors_from_frame_no_forces,
                            get_basis)
from .features import GlobalFeatures, LocalFeatures
from ase.data import atomic_numbers


class AceRepresentation(object):
    def __init__(self, N, maxdeg, rcut, species, r0=1.0,
                 rin=1.0, constants=False,
                 energy_name="dft_energy", force_name="dft_force",
                 virial_name="dft_virial"):
        self.N = N
        self.maxdeg = maxdeg
        self.rcut = float(rcut)
        if type(species) is not list:
            species = [species]
        for i in range(len(species)):
            if type(species[i])==str:
                species[i] = atomic_numbers[species[i]]
        self.species = species
        self.r0 = r0
        self.rin = rin
        self.constants = constants
        self.energy_name = energy_name
        self.force_name = force_name
        self.virial_name = virial_name
        if not hasattr(self, "basis"):
            self.basis, self.n_feat = get_basis(
                self.N, self.maxdeg, self.rcut, self.species,
                self.r0,  self.rin, self.constants)
            self.n_feat += len(self.species)

    def transform(self, frames, compute_derivative=True, verbose=False):
        if not hasattr(self, "basis"):
            self.basis, self.n_feat = get_basis(
                self.N, self.maxdeg, self.rcut, self.species,
                self.r0,  self.rin, self.constants)
            self.n_feat += len(self.species)
        if not isinstance(frames, list):
            frames = [frames]
        n_atoms = 0
        n_frames = len(frames)
        strides = [0]
        for frame in frames:
            n_atoms += len(frame)
            strides.append(len(frame))

        strides = np.cumsum(strides)
        X = np.zeros((n_frames, self.n_feat))
        if compute_derivative:
            dX_dr = np.zeros((n_atoms, 3, self.n_feat))
            dX_ds = np.zeros((n_frames, 6, self.n_feat))
        for i_frame in progressbar(range(len(frames)), verbose=verbose):
            frame = frames[i_frame]
            if compute_derivative:
                st, nd = strides[i_frame], strides[i_frame + 1]
                (
                    X[i_frame],
                    dX_dr[st:nd],
                    dX_ds[i_frame],
                ) = self._get_global_representation(self.basis, frame)
                dX_ds[i_frame] /= frame.get_volume()
            else:
                X[i_frame] = self._get_global_representation_no_forces(
                    self.basis, frame)

        if compute_derivative:
            return GlobalFeatures(self, X, dX_dr, dX_ds, strides, self.species)
        else:
            return GlobalFeatures(self, X, None, None, strides, self.species)

    def _get_global_representation(self, basis, frame):
        X, dX_dr_global, dX_ds_global = descriptors_from_frame(basis, frame, self.species,
                                                               self.energy_name,
                                                               self.force_name,
                                                               self.virial_name)
        return X, dX_dr_global, dX_ds_global

    def _get_global_representation_no_forces(self, basis, frame):
        X = descriptors_from_frame_no_forces(basis, frame, self.species, self.energy_name)
        return X

    def transform_local(self, frames, compute_derivative=True, verbose=False):
        if not hasattr(self, "basis"):
            self.basis, self.n_feat = get_basis(
                self.N, self.maxdeg, self.rcut, self.species,
                self.r0,  self.rin, self.constants)
            self.n_feat += len(self.species)
        if not isinstance(frames, list):
            frames = [frames]
        n_atoms = 0
        strides = [0]
        for frame in frames:
            n_atoms += len(frame)
            strides.append(len(frame))

        strides = np.cumsum(strides)
        X = []
        dX_dr = []
        dX_ds = []
        if compute_derivative:
            dX_dr = []
            dX_ds = []
        for i_frame in progressbar(range(len(frames)), verbose=verbose, local=True):
            frame = frames[i_frame]
            if compute_derivative:
                X_, dX_dr_, dX_ds_ =  self._get_local_representation(self.basis, frame)
                dX_ds_ /= frame.get_volume()
                X.append(X_)
                dX_dr.append(np.array(dX_dr_))
                dX_ds.append(np.array(dX_ds_))
            else:
                X.append(self._get_local_representation_no_forces(
                    self.basis, frame))

        if compute_derivative:
            return LocalFeatures(self, X, dX_dr, dX_ds, strides, self.species)
        else:
            return LocalFeatures(self, X, None, None, strides, self.species)

    def transform_single_local(self, frame, compute_derivative=True):
        if not hasattr(self, "basis"):
            self.basis, self.n_feat = get_basis(
                self.N, self.maxdeg, self.rcut, self.species,
                self.r0,  self.rin, self.constants)
            self.n_feat += len(self.species)
        strides = [0]
        strides.append(len(frame))
        strides = np.cumsum(strides)

        if compute_derivative:
            X_, dX_dr_, dX_ds_ =  self._get_local_representation(self.basis, frame)
            dX_ds_ /= frame.get_volume()
            X = np.array([X_])
            dX_dr = np.array([dX_dr_])
            dX_ds = np.array([dX_ds_])
        else:
            X = np.array([self._get_local_representation_no_forces(
                self.basis, frame)])

        if compute_derivative:
            return LocalFeatures(self, np.array(X), np.array(dX_dr), np.array(dX_ds), strides, self.species)
        else:
            return LocalFeatures(self, np.array(X), None, None, strides, self.species)
    
    def _get_local_representation(self, basis, frame):
        X, dX_dr_local, dX_ds_local = local_descriptors_from_frame(basis, frame, self.species,
                                                               self.energy_name,
                                                               self.force_name,
                                                               self.virial_name)
        dX_dr_local = np.transpose(dX_dr_local, axes = [0, 1, 3, 2])
        return X, dX_dr_local, dX_ds_local
        
    def _get_local_representation_no_forces(self, basis, frame):
        X_local = local_descriptors_from_frame_no_forces(basis, frame, self.species, self.energy_name)
        return X_local



def progressbar(it, prefix="", size=60, file=sys.stdout, verbose=True, local=False):
    if verbose and not local:
        file.write("Computing Features \n")
    if verbose and local:
        file.write("Computing Local Features \n")
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush() 
    if verbose:       
        show(0)
    for i, item in enumerate(it):
        yield item
        if verbose:
            show(i+1)
    if verbose:
        file.write("\n")
        file.flush()


# Why is this here, you might ask?
# Well I really don't know why, but if you call an 
# AceRepresentation object with big N or maxdeg right away, python (or julia, dk) freezes.
# If instead you first call AceRepresentation with small N and maxdeg, and then call the big one, no issue.
# So there you have it
a = AceRepresentation(3, 3, 3, 3)