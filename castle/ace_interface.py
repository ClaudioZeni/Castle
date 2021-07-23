from julia import Main
import numpy as np
import julia
JULIA_UTILS_PATH = "/home/claudio/postdoc/Castle/castle/julia_utils.jl"

j = julia.Julia(compiled_modules=False)


Main.using("JuLIP: Chemistry.AtomicNumber, Atoms")
Main.using("ACE: rpi_basis")
Main.using(
    "IPFitting.Data: read_Atoms, read_energy, read_forces, read_virial, read_configtype")
Main.using("IPFitting: Dat")
Main.include(JULIA_UTILS_PATH)


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
    X, dX_dr = Main.extract_info_frame(basis, at.at)
    # TODO: the non-summed descriptor must be
    # yielded at some point by the Julia interface

    dX_dr_local = 0
    return X.T, np.transpose(np.array(dX_dr), (1, 2, 0)), dX_dr_local


def compute_virial_descriptor(frame, dX_dr_local):
    pos = frame.positions
    posdiff = pos[:, None, :] - pos[None, :, :]
    dX_ds = np.einsum('nmc, nmds -> cds', posdiff, dX_dr_local)
    return dX_ds


def get_basis(N, maxdeg, rcut, species, r0=1.0,
              reg=1e-8, rin=1.0, constants=False):
    basis = Main.rpi_basis(N=N, r0=r0,
                           maxdeg=maxdeg, rcut=rcut,
                           species=species,
                           rin=rin, constants=constants)
    return basis, Main.length(basis)


def descriptors_from_frame(basis, frame, energy_name="energy",
                           force_name="force",
                           virial_name="virial"):

    at = frame_to_julia_at(frame, energy_name,
                           force_name, virial_name)
    X, dX_dr, dX_dr_local = descriptors_from_at(basis, at)
    dX_ds = compute_virial_descriptor(frame, dX_dr_local)
    return X, dX_dr, dX_ds
