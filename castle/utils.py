import numpy as np
import joblib
import pickle
from sklearn.metrics import r2_score
from ase.io import read
from . import AceGlobalRepresentation, AceLocalRepresentation
from ase.data import atomic_numbers


def dump(fn, obj, compress=3, protocol=pickle.HIGHEST_PROTOCOL):
    joblib.dump(obj, fn, protocol=protocol, compress=compress)


def load(fn):
    return joblib.load(fn)


def get_forces_and_energies(frames, energy_name=None, force_name=None):

    # Look for energy name if not provided
    if energy_name is None:
        infos = list(frames[0].info.keys())
        matching = [
            s
            for s in infos
            if (
                ("energy" in s)
                or ("ENERGY" in s)
                or ("Energy" in s)
                or ("PE" in s)
                or ("EN" in s)
                or ("En" in s)
                or ("en" in s)
            )
        ]
        # Only one matching name is acceptable
        if len(matching) == 1:
            energy_name = matching[0]
        else:
            print(
                "WARNING: energy name must be specified \
                    to extract energies."
            )

    # Look for force name if not provided
    if force_name is None:
        arrays = list(frames[0].arrays.keys())
        matching = [
            s
            for s in arrays
            if (
                ("force" in s)
                or ("FORCE" in s)
                or ("Force" in s)
                or ("forces" in s)
                or ("Forces" in s)
                or ("FORCES" in s)
            )
        ]
        # Only one matching name is acceptable
        if len(matching) == 1:
            force_name = matching[0]
        else:
            print(
                "WARNING: force name must be specified \
                    to extract forces."
            )

    # Get energies from xyz
    if energy_name is not None:
        energies = []
        for f in frames:
            energies.append(f.info[energy_name])
        energies = np.array(energies)

    else:
        energies = None

    # Get forces from xyz
    if force_name is not None:
        forces = []
        for f in frames:
            forces.extend(f.arrays[force_name])
        forces = np.array(forces)
    else:
        forces = None

    return energies, forces


matrix_indices_in_voigt_notation = [
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
]


def full_3x3_to_voigt_6_stress(stress_matrix):
    """
    Form a 6 component stress vector in Voigt notation from a 3x3 matrix
    """
    stress_matrix = np.asarray(stress_matrix)
    return np.array(
        [stress_matrix[i, j] for (i, j) in matrix_indices_in_voigt_notation]
    )


def get_virials(frames, virial_name):
    virials = []
    for f in frames:
        virials.append(full_3x3_to_voigt_6_stress(
            f.info[virial_name].reshape((3, 3))))
    virials = np.array(virials).reshape((len(frames), 6))
    return virials


def get_nat(frames):
    # Get number of atoms per frame from xyz
    nat = []
    for f in frames:
        nat.append(len(f))
    nat = np.array(nat)
    return nat


def print_score(y1, y2):
    diff = y1-y2
    mae = np.mean(abs(diff))
    rmse = np.mean(diff**2)**0.5
    sup = max(diff)
    r2 = r2_score(y1, y2)
    print("MAE=%.3f RMSE=%.3f SUP=%.3f R2=%.3f" % (mae, rmse, sup, r2))


def get_score(y1, y2):
    diff = y1-y2
    mae = np.mean(abs(diff))
    rmse = np.mean(diff**2)**0.5
    sup = max(diff)
    r2 = r2_score(y1, y2)
    return mae, rmse, sup, r2


def extract_features(folder, train_filename, validation_filename=None, 
                    N=8, maxdeg=10, rcut=4.0, r0 = 1.0, reg = 1e-8, species = None,
                    force_name = None, energy_name = None):
    
    if validation_filename is None:
        tr_frames = read(folder + train_filename, index = ':')
        
    else:
        tr_frames = read(folder + train_filename, index = ':')
        val_frames = read(folder + validation_filename, index = ':')

    if species is None:
        species = list(set(tr_frames[0].get_atomic_numbers()))
    if type(species)==str:
        species = [atomic_numbers[species]]

    representation = AceGlobalRepresentation(N, maxdeg, rcut, species, r0, reg, 
                                             energy_name=energy_name, force_name=force_name)

    tr_features = representation.transform(tr_frames)
    if validation_filename is not None:
        val_features = representation.transform(val_frames)
        dump(folder + f"/tr_features_N_{N}_d_{maxdeg}.xz", tr_features)
        dump(folder + f"/val_features_N_{N}_d_{maxdeg}.xz", val_features)
        return tr_features, val_features
    	
    else:
        dump(folder + f"/features_N_{N}_d_{maxdeg}.xz", tr_features)
        return tr_features


def extract_local_features(folder, train_filename, validation_filename=None, 
                    N=8, maxdeg=10, rcut=4.0, r0 = 1.0, reg = 1e-8, species = None,
                    force_name = None, energy_name = None, compute_derivative=False):
    
    if validation_filename is None:
        tr_frames = read(folder + train_filename, index = ':')
        
    else:
        tr_frames = read(folder + train_filename, index = ':')
        val_frames = read(folder + validation_filename, index = ':')

    if species is None:
        species = list(set(tr_frames[0].get_atomic_numbers()))
    if type(species)==str:
        species = atomic_numbers[species]

    representation = AceLocalRepresentation(N, maxdeg, rcut, species, r0, reg, 
                                             energy_name=energy_name, force_name=force_name)

    tr_features = representation.transform(tr_frames, compute_derivative)
    if validation_filename is not None:
        val_features = representation.transform(val_frames)
        # dump(folder + f"/tr_local_features_N_{N}_d_{maxdeg}.xz", tr_features)
        # dump(folder + f"/val_local_features_N_{N}_d_{maxdeg}.xz", val_features)
        return tr_features, val_features
    	
    else:
        # dump(folder + f"/local_features_N_{N}_d_{maxdeg}.xz", tr_features)
        return tr_features


def load_everything(folder, tr_traj_name, val_traj_name, tr_features_name, val_features_name,
                    force_name = None, energy_name = None, validation_split = 0.8):
    if val_traj_name is not None and val_features_name is not None:
        tr_frames = read(folder + tr_traj_name, index = ':')
        val_frames = read(folder + val_traj_name, index = ':')
        tr_features = load(folder + tr_features_name)
        val_features = load(folder + val_features_name)
    else:
        frames = read(folder + tr_traj_name, index = ':')
        ind = np.arange(len(frames))
        np.random.shuffle(ind)
        tr_ind = ind[:int(len(ind)*validation_split)]
        val_ind = ind[int(len(ind)*validation_split):]
        tr_frames = [frames[i] for i in tr_ind]
        val_frames = [frames[i] for i in val_ind]
        features = load(folder + tr_features_name)
        tr_features = features.get_subset(tr_ind)
        val_features = features.get_subset(val_ind)
    e_t, f_t = get_forces_and_energies(tr_frames, energy_name = energy_name, force_name = force_name)
    e_val, f_val = get_forces_and_energies(val_frames, energy_name = energy_name, force_name = force_name)
    nat_val = get_nat(val_frames)
    nat_tr = get_nat(tr_frames)
    return e_t, f_t, e_val, f_val, nat_tr, nat_val, tr_features, val_features
