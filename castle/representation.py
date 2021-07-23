import ase
import numpy as np

from .ace_interface import get_basis, descriptors_from_frame
from .features import GlobalFeatures


class AceGlobalRepresentation(object):
    def __init__(self, n_body, maxdeg, rcut, species, r0=1.0,
                 reg=1e-8, rin=1.0, constants=False,
                 energy_name="dft_energy", force_name="dft_force",
                 virial_name="dft_virial"):
        self.n_body = n_body
        self.maxdeg = maxdeg
        self.rcut = rcut
        self.species = species
        self.r0 = r0
        self.reg = reg
        self.rin = rin
        self.constants = constants
        self.energy_name = energy_name
        self.force_name = force_name
        self.virial_name = virial_name

    def transform(self, frames):
        basis, self.n_feat = get_basis(
            self.n_body, self.maxdeg, self.rcut, self.species,
            self.r0, self.reg, self.rin, self.constants)
        if not isinstance(frames, list):
            frames = [frames]
        n_atoms = 0
        n_frames = len(frames)
        species = []
        strides = [0]
        for frame in frames:
            n_atoms += len(frame)
            strides.append(len(frame))
            if isinstance(frame, ase.Atoms):
                species.extend(frame.get_atomic_numbers())
            else:
                for at in frame:
                    species.append(at.atom_type)

        species = np.unique(species)
        strides = np.cumsum(strides)
        X = np.zeros((n_frames, self.n_feat))
        dX_dr = np.zeros((n_atoms, 3, self.n_feat))
        dX_ds = np.zeros((n_frames, 6, self.n_feat))
        for i_frame in range(len(frames)):
            frame = frames[i_frame]
            st, nd = strides[i_frame], strides[i_frame + 1]
            (
                X[i_frame],
                dX_dr[st:nd],
                dX_ds[i_frame],
            ) = self._get_global_representation_single_species(basis, frame)
            dX_ds[i_frame] /= frame.get_volume()
        return GlobalFeatures(self, X, dX_dr, dX_ds, strides, species)

    def _get_global_representation_single_species(self, basis, frame):
        X, dX_dr_global, dX_ds_global = descriptors_from_frame(basis, frame,
                                                               self.energy_name,
                                                               self.force_name,
                                                               self.virial_name)
        return X, dX_dr_global, dX_ds_global
