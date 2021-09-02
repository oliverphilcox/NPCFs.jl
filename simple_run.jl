using Pkg
Pkg.activate(".")
Pkg.instantiate()
using NPCFs

# Addition testing packages
using Statistics, Random, Plots, TimerOutputs, Printf

## Input parameters
_D = 3
_N = 4
_Npart = 300
_boxsize = 1000.
_coords = "cartesian"
_r_min = 50
_r_max = 250
_lmax = 2
_periodic = true

## Create input data
if _coords=="spherical"
    phi_arr = rand(_Npart)*2pi
    theta_arr = acos.(rand(_Npart).*2 .-1)
    pos = hcat(phi_arr,theta_arr,ones(_Npart)); # assuming spherical geometry, labelled phi, theta
else
    pos = hcat(rand(_Npart,_D)*_boxsize,ones(_Npart)); # assuming Cartesian grid
end

## Define class
npcf = NPCF(N=_N,D=_D,verb=true,periodic=_periodic,coords=_coords,r_min=_r_min,r_max=_r_max,lmax=_lmax);

## First run to compile
compute_npcf_simple(pos, npcf);
compute_npcf_pairwise(pos, npcf);

## Second run to time
@time npcf_output1 = compute_npcf_simple(pos, npcf);
@time npcf_output2 = compute_npcf_pairwise(pos, npcf);

## Error diagnostics
mean_err = mean(npcf_output1-npcf_output2)/mean(npcf_output2)
std_err = std(npcf_output1-npcf_output2)/mean(npcf_output2)
@printf("\nError: %.1e +- %.1e",mean_err,std_err)
