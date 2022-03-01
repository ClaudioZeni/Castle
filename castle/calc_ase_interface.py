from ase.calculators.calculator import Calculator, all_changes


class ASEMLCalculator(Calculator):
    """Wrapper class to use a rascal model as an interatomic potential in ASE

    Parameters
    ----------
    model : class
        a trained model of the castle library that can predict the energy and
        derivaties of the energy w.r.t. atomic positions
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
        if "forces" in properties:
            forces=True
        else:
            forces=False
        if "stress" in properties:
            stress=True
        else:
            stress=False

        self.results = self.model.predict(atoms, forces, stress)


