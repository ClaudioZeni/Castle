using ACE: alloc_B, alloc_temp, evaluate!,alloc_dB, alloc_temp_d, evaluate_d!
using NeighbourLists:maxneigs
using JuLIP:  neighbourlist, cutoff, JVec, AbstractAtoms, fltype, AtomicNumber
using JuLIP.Potentials:neigsz!
using IPFitting.Data:read_xyz
using Einsum

# Local : environment descriptor
function environment_descriptor(shipB, at::AbstractAtoms{T}) where {T}
    E = zeros(fltype(shipB), length(at), length(shipB))
    B = alloc_B(shipB)
    nlist = neighbourlist(at, cutoff(shipB))
    maxnR = maxneigs(nlist)
    tmp = alloc_temp(shipB, maxnR)
    tmpRZ = (R = zeros(JVec{T}, maxnR), Z = zeros(AtomicNumber, maxnR))
    for i = 1:length(at)
        j, R, Z = neigsz!(tmpRZ, nlist, at, i)
        fill!(B, 0)
        evaluate!(B, tmp, shipB, R, Z, at.Z[i])
        E[i,:] = B[:]
    end
    return E
end

# Local : environments for all the atoms in trajectory
function environment_descriptor_traj(basis, traj)
    A_all = []
    for i in 1:length(traj)
        atoms = traj[i].at
        X = environment_descriptor(basis, atoms);
        push!(A_all, X)
    end
    return A_all
end;


# Global : sum descriptor
function sum_descriptor(shipB, at::AbstractAtoms{T}) where {T}
    E = zeros(fltype(shipB), length(shipB))
    B = alloc_B(shipB)
    nlist = neighbourlist(at, cutoff(shipB))
    maxnR = maxneigs(nlist)
    tmp = alloc_temp(shipB, maxnR)
    tmpRZ = (R = zeros(JVec{T}, maxnR), Z = zeros(AtomicNumber, maxnR))
    for i = 1:length(at)
        j, R, Z = neigsz!(tmpRZ, nlist, at, i)
        fill!(B, 0)
        evaluate!(B, tmp, shipB, R, Z, at.Z[i])
        E[:] .+= B[:]
    end
    return E
end

function sum_descriptor_traj(basis, traj)
    A_all = Array{Float64,2}(undef, length(traj), length(basis)) 
    for i in 1:length(traj)
        atoms = traj[i].at
        A_all[i, :] = sum_descriptor(basis, atoms)
    end
    return A_all
end

function sum_d_descriptor(shipB, at::AbstractAtoms{T}) where {T}
   # precompute the neighbourlist to count the number of neighbours
    nlist = neighbourlist(at, cutoff(shipB); storelist=false)
    maxR = maxneigs(nlist)
   # allocate space accordingly
    F = zeros(JVec{T}, length(at), length(shipB))
    F_local = zeros(JVec{T}, length(at), length(at), length(shipB))
    B = alloc_B(shipB, maxR)
    dB = alloc_dB(shipB, maxR)
    tmp = alloc_temp_d(shipB, maxR)
    tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(AtomicNumber, maxR))
    return sum_d_descriptor_inner!(shipB, at, nlist, F, F_local, B, dB, tmp, tmpRZ)
end

# this is a little hack to remove a type instability. It probably makes no
# difference in practise...
function sum_d_descriptor_inner!(shipB, at::AbstractAtoms{T},
                       nlist, F, F_local, B, dB, tmp, tmpRZ) where {T}
   # assemble site gradients and write into F
    for i = 1:length(at)
        j, R, Z = neigsz!(tmpRZ, nlist, at, i)
        fill!(dB, zero(JVec{T}))
        fill!(B, 0)
        evaluate_d!(B, dB, tmp, shipB, R, Z, at.Z[i])
        for a = 1:length(R)
            F[j[a], :] .-= dB[:, a]
            F[i, :] .+= dB[:, a]
            F_local[i, j[a], :] .-= dB[:, a]
        end
    end
    virial = compute_virial(F_local, at)
    return [ F[:, iB] for iB = 1:length(shipB) ], [virial[:, :, iB] for iB = 1:length(shipB) ]
end

function compute_virial(F_local, at)
    p = [tup[k] for tup in at.X, k in 1:3]
    d = permutedims(repeat(p[:, :], 1, 1, length(at.X)) .- permutedims(repeat(p[:, :], 1, 1, length(at.X)), (3, 2, 1)), (1, 3, 2))
    F_local = [tup[k] for tup in F_local, k in 1:3]
    @einsum v[c1, c2, s] :=  - F_local[n, m, s, c1] * d[n, m, c2]
    return v
end

function sum_d_descriptor_traj(basis, traj)
    dA_all = []
    dA_ds = []
    for i in 1:length(traj)
        atoms = traj[i].at
        XF, virial = sum_d_descriptor(basis, atoms);
        XF = hcat([collect(Iterators.flatten(a)) for a in XF]...)
        push!(dA_all, XF)
        push!(dA_ds, virial)

    end
    return dA_all, dA_ds
end

function fit_potential(XE_tr, YE_tr, XF_tr, YF_tr, alpha=1.0)
    YF_tr = hcat(YF_tr'...)'
    XF_tr = hcat(XF_tr'...)'
    X_tr = vcat(XE_tr, XF_tr)
    Y_tr = vcat(YE_tr, YF_tr);
    ridge_pred = fit!(Ridge(alpha), X_tr, Y_tr)
    return ridge_pred
end

function predict_potential(ridge_pred, XE_tst, XF_tst)
    n_struc = length(XE_tst[:, 1])
    XF_tst = hcat(XF_tst'...)'
    X_tst = vcat(XE_tst, XF_tst)
    result = predict(ridge_pred, X_tst)
    return result[1:n_struc], result[n_struc + 1:end]
end

function extract_info_traj(B, traj)
    X = sum_descriptor_traj(B, traj);
    dX_dr, dX_ds = sum_d_descriptor_traj(B, traj);
    
    return X, dX_dr, dX_ds
end

function extract_info_frame(B, frame)
    X = sum_descriptor(B, frame);
    dX_dr, dX_ds = sum_d_descriptor(B, frame);
    
    return X, dX_dr, dX_ds
end
