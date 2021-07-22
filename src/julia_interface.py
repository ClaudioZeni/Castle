import julia
# julia.install()
j = julia.Julia(compiled_modules=False)
from julia import Main


Main.using("JuLIP: Chemistry.AtomicNumber, Atoms")
Main.using("ACE: rpi_basis")
Main.using(
    "IPFitting.Data: read_Atoms, read_energy, read_forces, read_virial, read_configtype")
Main.using("IPFitting: Dat")
Main.include("../src/utils.jl")


def convert_julia_at(a, energy_name="energy",
                     force_name="force", virial_name="virial"):
    at = Main.read_Atoms(a)
    E = Main.read_energy(a, energy_name)
    F = Main.read_forces(a, force_name)
    V = Main.read_virial(a, virial_name)
    cf = Main.read_configtype(a)
    at = Main.Dat(at, cf, E=E, F=F, V=V)
    return at


def extract_descriptors(basis, at):
    XE, YE, XF, YF, nat = Main.extract_info(basis, [at])
    return XE[0], YE[0], XF[0], YF[0], nat


def get_basis(N, maxdeg, rcut, species, r0=1.0,
              reg=1e-8, rin=1.0, constants=False):
    basis = Main.rpi_basis(N=N, r0=r0,
                           maxdeg=maxdeg, rcut=rcut,
                           species=species,
                           rin=rin, constants=constants)
    return basis
