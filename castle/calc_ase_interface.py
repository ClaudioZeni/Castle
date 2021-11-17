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

    def __init__(self, model, representation, **kwargs):
        super(ASEMLCalculator, self).__init__(**kwargs)
        self.model = model
        self.representation = representation
        self.kwargs = kwargs
        self.manager = None

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms.copy()
        at.wrap(eps=1e-11)
        self.manager = [at]
        features = self.representation.transform(self.manager)
        energy = self.model.predict_energy(features)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        
        if "forces" in properties:
            self.results["forces"] = self.model.predict_forces(features)
        if "stress" in properties:
            self.results["stress"] = self.model.predict_stress(features).flatten()

