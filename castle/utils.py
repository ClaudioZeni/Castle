import numpy as np
import joblib
import pickle
from sklearn.metrics import r2_score


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
    print("MAE=%.2f RMSE=%.2f SUP=%.2f R2=%.2f" % (mae, rmse, sup, r2))
