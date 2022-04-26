import julia
import numpy as np

JULIA_UTILS_PATH = "/home/claudio/postdoc/Castle/castle/julia_utils.jl"

jl = julia.Julia(compiled_modules=False)
from julia import Main
Main.using("JuLIP: Chemistry.AtomicNumber, Atoms")
Main.using("ACE: rpi_basis")
Main.using(
    """IPFitting.Data: read_Atoms, read_energy,
     read_forces, read_virial, read_configtype""")
Main.using("IPFitting: Dat")
Main.include(JULIA_UTILS_PATH)

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
    at = Main.read_Atoms(frame)
    E = Main.read_energy(frame, energy_name)
    F = Main.read_forces(frame, force_name)
    V = Main.read_virial(frame, virial_name)
    cf = Main.read_configtype(frame)
    if cf is None:
        cf = 'unknown'
    at_ = Main.Dat(at, cf, E=E, F=F, V=V)
    return at_


def descriptors_from_at(basis, at):
    X, dX_dr, dX_ds = Main.extract_info_frame(basis, at.at)
    return (X.T, np.transpose(np.array(dX_dr), (1, 2, 0)),
            np.transpose(np.array(dX_ds), (1, 2, 0)))


def get_basis(N, maxdeg, rcut, species, r0=1.0,
              rin=1.0, constants=False):
    basis = Main.rpi_basis(N=N, r0=r0,
                        maxdeg=maxdeg, rcut=rcut,
                        species=species,
                        rin=rin, constants=constants)
    return basis, Main.length(basis)


def descriptors_from_frame(basis, frame, species,
                           energy_name="energy",
                           force_name="force",
                           virial_name="virial"):

    at = frame_to_julia_at(frame, energy_name,
                           force_name, virial_name)
    X, dX_dr, dX_ds = descriptors_from_at(basis, at)
    dX_ds = dX_ds[matrix_indices_in_voigt_notation, :][:, 0, 0, :]
    X, dX_dr, dX_ds = add_onebody_term(frame, species, X, dX_dr, dX_ds)
    return X, dX_dr, dX_ds


def descriptors_from_frame_no_forces(basis, frame, species, energy_name="energy"):
    at = frame_to_julia_at(frame, energy_name)
    X = Main.sum_descriptor(basis, at.at)
    X, _, __ = add_onebody_term(frame, species, X)
    return X


def local_descriptors_from_frame(basis, frame, species,
                           energy_name="energy",
                           force_name="force",
                           virial_name="virial"):
    at = frame_to_julia_at(frame, energy_name,
                           force_name, virial_name)
    X = Main.environment_descriptor(basis, at.at)   
    dX_dr, dX_ds = Main.environment_d_descriptor(basis, at.at)
    dX_dr = np.array(dX_dr)
    X, dX_dr = add_onebody_local_term(frame, species, X, dX_dr)
    return  X, dX_dr, dX_ds


def local_descriptors_from_frame_no_forces(basis, frame, species, energy_name="energy"):
    at = frame_to_julia_at(frame, energy_name)  
    X = Main.environment_descriptor(basis, at.at)
    X, _ = add_onebody_local_term(frame, species, X)
    return X


def add_onebody_term(frame, species, X, dX_dr=None, dX_ds=None):
    at_ns = frame.get_atomic_numbers()
    unique, counts = np.unique(at_ns, return_counts=True)
    unique_counts = dict(zip(unique, counts))
    tally = np.zeros(len(species))
    for i, s in enumerate(species):
        tally[i] = unique_counts[s]
    if dX_dr is not None:
        dX_dr = np.concatenate([dX_dr, np.zeros((dX_dr.shape[0], 3, len(tally)))], axis = -1)
    if dX_ds is not None:
        dX_ds = np.concatenate([dX_ds, np.zeros((6, len(tally)))], axis = -1)
    X = np.concatenate([X, tally])
    return X, dX_dr, dX_ds


def add_onebody_local_term(frame, species, X, dX_dr=None):
    at_ns = frame.get_atomic_numbers()
    tally = np.zeros((len(at_ns), len(species)))
    for i in np.arange(len(at_ns)):
        tally[i, :] = 1*(species == at_ns[i])
    X = np.concatenate([X, tally], axis = -1)
    if dX_dr is not None:
        dX_dr = np.concatenate([dX_dr, np.zeros((dX_dr.shape[0], dX_dr.shape[1], len(species), 3))], axis = -2)
    return X, dX_dr