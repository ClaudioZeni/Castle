import julia
import numpy as np

JULIA_UTILS_PATH = "/home/claudio/postdoc/Castle/castle/julia_utils.jl"

j = julia.Julia(compiled_modules=False)
j.using("JuLIP: Chemistry.AtomicNumber, Atoms")
j.using("ACE: rpi_basis")
j.using(
    """IPFitting.Data: read_Atoms, read_energy,
     read_forces, read_virial, read_configtype""")
j.using("IPFitting: Dat")
j.include(JULIA_UTILS_PATH)

matrix_indices_in_voigt_notation = [
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
]


def frame_to_julia_at(frame, energy_name="energy",
                      force_name="force", virial_name="virial"):
    at = j.read_Atoms(frame)
    E = j.read_energy(frame, energy_name)
    F = j.read_forces(frame, force_name)
    V = j.read_virial(frame, virial_name)
    cf = j.read_configtype(frame)
    if cf is None:
        cf = 'unknown'
    at_ = j.Dat(at, cf, E=E, F=F, V=V)
    return at_


def descriptors_from_at(basis, at):
    X, dX_dr, dX_ds = j.extract_info_frame(basis, at.at)
    return (X.T, np.transpose(np.array(dX_dr), (1, 2, 0)),
            np.transpose(np.array(dX_ds), (1, 2, 0)))


def get_basis(N, maxdeg, rcut, species, r0=1.0,
              reg=1e-8, rin=1.0, constants=False):
    basis = j.rpi_basis(N=N, r0=r0,
                        maxdeg=maxdeg, rcut=rcut,
                        species=species,
                        rin=rin, constants=constants)
    return basis, j.length(basis)


def descriptors_from_frame(basis, frame, energy_name="energy",
                           force_name="force",
                           virial_name="virial"):

    at = frame_to_julia_at(frame, energy_name,
                           force_name, virial_name)
    X, dX_dr, dX_ds = descriptors_from_at(basis, at)
    dX_ds = dX_ds[matrix_indices_in_voigt_notation, :][:, 0, 0, :]
    return X, dX_dr, dX_ds


def descriptors_from_frame_no_forces(basis, frame, energy_name="energy"):
    at = frame_to_julia_at(frame, energy_name)
    X = j.sum_descriptor(basis, at.at)
    return X
