# Castle
Ensamble learning force fields within the PACE framework

# Instructions

Install julia-1.5.3

go in julia (julia in terminal)

press ] to go to package manager

run the following:


registry add https://github.com/JuliaMolSim/MolSim.git

add JuLIP, IPFitting
pin JuLIP@v0.10; pin IPFitting@v0.5

Download ACE from https://github.com/ACEsuit/ACE.jl, branch dev-v0.8.x.

In the ACE folder, change lines 53 and 54 in sphericalharmonics.jl from :

	cosθ = R[3] / r
	sinθ = sqrt(R[1]^2+R[2]^2) / r
	
to:

	θ = atan(sqrt(R[1]^2+R[2]^2+ 1e-8), R[3])
	sinθ, cosθ = sincos(θ)

Change the second line of the notebook from:

  Pkg.activate("/home/claudio/postdoc/ACE.jl/Project.toml")

to:

  Pkg.activate(where/is/your/ACE/ACE.jl/Project.toml")
