# Castle

Ensemble of ridge regression force fields in Python.

This repository is based on the ACE descriptor [1] as descibed in [2] and implemented in [3]


# Instructions

Install julia-1.5.3

go in julia (julia in terminal)

press ] to go to package manager

run the following:


	registry add https://github.com/JuliaMolSim/MolSim.git
	add JuLIP, IPFitting
	pin JuLIP@v0.10; pin IPFitting@v0.5

Download ACE from https://github.com/ACEsuit/ACE.jl, branch dev-v0.8.x.

Then, install the julia Python package with:

	pip install julia


# References
[1] Drautz, R. (2019). Atomic cluster expansion for accurate and transferable interatomic potentials. Physical Review B, 99(1), 014104.

[2] Dusson, G., Bachmayr, M., Csanyi, G., Drautz, R., Etter, S., van der Oord, C., & Ortner, C. (2022). Atomic cluster expansion: Completeness, efficiency and stability. Journal of Computational Physics, 110946.

[3] ACE.jl https://github.com/ACEsuit/ACE.jl
