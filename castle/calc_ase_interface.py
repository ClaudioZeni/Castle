from asyncio.format_helpers import _format_callback_source
from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy


class ASEMLCalculator(Calculator):
    """Wrapper class to use a rascal model as an interatomic potential in ASE

    Parameters
    ----------
    model : class
        a trained model of the rascal library that can predict the energy and
        derivaties of the energy w.r.t. atomic positions
    representation : class
        a representation calculator of rascal compatible with the trained model
    """

    implemented_properties = ["energy", "forces", "stress"]
    "Properties calculator can handle (energy, forces, ...)"

    default_parameters = {}
    "Default parameters"

    nolabel = True

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.kwargs = kwargs

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=all_changes,
    ):
        
        if "forces" and "stress" in properties:
            energy, forces, stress = self.model.predict(atoms, forces=True, stress=True)
            self.results["forces"] = forces
            self.results["stress"] = stress
        elif "forces" in properties:
            energy, forces = self.model.predict(atoms, forces=True)
            self.results["forces"] = forces
        else:
            energy = self.model.predict(atoms, forces=False)

        self.results["energy"] = energy
        self.results["free_energy"] = energy


