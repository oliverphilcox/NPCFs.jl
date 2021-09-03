module NPCFs
export NPCF, compute_npcf_simple, compute_npcf_pairwise, summarize, angular_indices

### Load packages
using Printf: @printf
using Distributed: @sync, @async, @spawnat, nworkers, workers, Future, myid
using Parameters: @with_kw
using Rotations: AngleAxis
using WignerSymbols: wigner3j, wigner6j
using SphericalHarmonics: computePlmcostheta
using LinearAlgebra: cross, norm
using LegendrePolynomials: Pl
using HypergeometricFunctions: _₂F₁
using ClassicalOrthogonalPolynomials: chebyshevt
using SpecialFunctions: gamma

### Define basic NPCF structure

"""
    NPCF(N, D, coords = "cartesian", periodic = "false", volume = 1e9, nbins = 10, r_min = 50, r_max = 200, lmax = 2, verb = false, _dr = -1, _boxsize = -1)

Create the NPCF object, containing all required attributes and output arrays.
Variables can be specified by their keywords, and take default values if not given.
The key variables are the NPCF order `N` and the dimension `D`.

# Arguments
- `N::Int64`: which NPCF to use (N = 2, 3, 4, 5 currently supported)
- `D::Int64`: number of dimensions

# Optional Arguments
- `coords::String = "cartesian"`: whether input coordinates are "cartesian" or "spherical"
- `periodic::Bool = false`: whether the particles use a periodic wrapping
- `volume::Float64 = 1e9`: volume of input (used for normalization, and periodic wrapping)

- `nbins::Int64 = 10`: number of radial bins
- `r_min::Float64 = 50.`: minimum bin
- `r_max::Float64 = 200`: # maximum bin

- `lmax::Int64 = 2`: maximum ell
- `verb::Bool = false`: whether to print throughout runtime

- `_dr::Float64 = -1:` bin-size, this will be reset on initialization
- `_boxsize::Float64 = -1:` periodic boxsize, this will be reset on initialization

# Examples
```julia-repl
julia> npcf = NPCF(N=4, D=3); # create a 4PCF object in 3D
...
julia> summarize(npcf); # summarize the NPCF inputs
...
julia> inputs = hcat(rand(100,3)*1000.,ones(100)); # generate 100 randomly distributed particles in [0, 1000], plus weights
...
julia> output_simple = compute_npcf_simple(npcf, inputs); # run the simple NPCF estimator
julia> output_pairwise = compute_npcf_pairwise(npcf, inputs); # run the efficient NPCF estimator
...
julia> l1, l2, l3  = angular_indices(npcf); # return a list of ell indices corresponding to the one-dimensional indexing used in the code.
```
"""
@with_kw struct NPCF
    # NPCF Structure
    N::Int64 # which NPCF to use (N = 2, 3, 4, 5 currently supported)
    D::Int64 # number of dimensions

    # Coordinates
    coords::String = "cartesian" # whether input coordinates are "cartesian" or "spherical"

    # Periodicity
    periodic::Bool = false # whether the particles use a periodic wrapping
    volume::Float64 = 1e9 # volume of input (used for normalization, and periodic wrapping)

    # Radial Binning
    nbins::Int64 = 10 # number of radial bins
    r_min::Float64 = 50. # minimum bin
    r_max::Float64 = 200. # maximum bin

    # Angular Binning
    lmax::Int64 = 2; # maximum ell

    # Verbosity
    verb::Bool = false # whether to print throughout runtime

    # Other parameters, set on initialization
    _dr::Float64 = -1; # bin-size, computed from binning parameters
    _boxsize::Float64 = -1; # boxsize, computed from volume

    # Check bounds and create object
    function NPCF(N,D,coords,periodic,volume,nbins,r_min,r_max,lmax,verb,_dr,_boxsize)

        # Check that derived parameters haven't been set
        @assert _boxsize==-1 "This will be reset on initialization!"
        @assert _dr==-1 "This will be reset on initialization!"

        # Set additional parameters
        _dr = (r_max-r_min)/nbins;
        if periodic
            _boxsize = volume^(1/D)
        end

        # Check input parameters
        @assert (D==4 || D==3 || D==2) "Only 2-, 3- and 4-dimensions implemented"
        @assert (N==5 || N==4 || N==3 || N==2) "Only 2PCF, 3PCF, 4PCF and 5PCF implemented"
        @assert r_max>r_min "Maximum radius must be greater than minimum radius"
        if coords=="spherical"
            @assert D==2 "Spherical coordinates only implemented in 2-dimensions"
            @assert periodic==false "Periodic wrapping not available for spherical coordinates"
            @assert (r_min>=-1 && r_max<=1) "Radial bins must be in range [-1,1]"
            if volume!=4pi
                println("Resetting volume to 4pi")
                volume=4pi
            end
        elseif coords=="cartesian"
            @assert (r_min>=0 && r_max>=0) "Radial bins must be in range [0,inf)"
            if periodic
                @assert (r_max<_boxsize/2) "r-max must be smaller than boxsize/2"
                @assert _boxsize>0 "Boxsize must be greater than zero"
            end
        end

        npcf=new(N,D,coords,periodic,volume,nbins,r_min,r_max,lmax,verb,_dr,_boxsize)
        if verb; summarize(npcf); end
        return npcf
    end
end

### Load utility functions

include("utils.jl")

### Key NPCF functions

"""
    compute_npcf_simple(inputs, npcf)

Compute the NPCF by summing over `N`-tuplets of particles given in the `inputs` dataset.
For `n` particles, the computational cost scales as `n^N`, and is thus not recommended for large `n`.
Work is distributed approximately equally over all ```nworkers()``` workers provided.

The function is initialized with the `NPCF` class, which specifies all relevant parameters.
The `inputs` array gives a list of particle positions in `D`-dimensions, and a `D+1`-th column specifying the particle weight.
The output is a vector array, where the first `N-1` indices specify the radial bins, and the `N`-th index gives the angular bin, collapsed into one dimension (see ```angular_indices``` for conversion to ells).
Note that we only fill up radial bins satisfying bin1<bin2<...<binN.
The actual computation is performed in the ```_compute_npcf_simple()``` function; this is just a wrapper function to facilitate multiprocessing.

# Examples
```julia-repl
julia> inputs = hcat(rand(100,3)*1000.,ones(100)); # 100 randomly distributed 3D particles in [0, 1000], plus weights
julia> npcf = NPCF(N=4, D=3); # create a 4PCF object in 3D
julia> output_simple = compute_npcf_simple(inputs, npcf); # run the simple NPCF estimator
```
"""
function compute_npcf_simple(inputs::Matrix{Float64}, npcf::NPCF)

    # First check inputs
    check_inputs(inputs, npcf)

    Npart = length(inputs[:,1])
    if npcf.verb; @printf("\nAnalyzing dataset with %d particles using %d worker(s) by summing over all N-tuplets of galaxies\n",Npart,nworkers()); end

    # Set up output array to hold NPCF
    output = create_npcf_array(npcf)

    # Work out how many workers are present and chunk memory
    tasks_per_worker = div(Npart,nworkers())+1

    # Now perform the chunking and run the algorithm across separate workers
    if npcf.verb;
        if nworkers()>1
            println("Starting distributed computation");
        else
            println("Starting computation");
        end
    end
    chunked_output = Dict{Int,Future}()
    @sync for (w_i,w) in enumerate(workers())
        imin = tasks_per_worker*(w_i-1)+1
        imax = min(imin+tasks_per_worker,Npart+1)-1
        @async chunked_output[w_i] = @spawnat w _compute_npcf_simple(imin, imax, inputs, npcf)
    end

    # Now combine the output together
    for w_i in 1:nworkers()
        output .+= chunked_output[w_i][]
    end

    # Apply normalization to the output array
    normalize_npcf!(output, npcf)

    if npcf.verb; @printf("Calculations complete!\n"); end

    return output
end;

"""
    compute_npcf_pairwise(inputs, npcf)

Compute the NPCF of the `inputs` dataset by summing over pairs of particles and utilizing hyperspherical harmonic decompositions.
For `n` particles, the computational cost scales as `n^2`, and thus can be extended to large `n`.
Work is distributed approximately equally over all ```nworkers()``` workers provided.

The function is initialized with the `NPCF` class, which specifies all relevant parameters.
The `inputs` array gives a list of particle positions in `D`-dimensions, and a `D+1`-th column specifying the particle weight.
The output is a vector array, where the first `N-1` indices specify the radial bins, and the `N`-th index gives the angular bin, collapsed into one dimension (see ```angular_indices``` for conversion to ells).
Note that we only fill up radial bins satisfying bin1<bin2<...<binN.
The actual computation is performed in the ```_compute_npcf_pairwise()``` function; this is just a wrapper function to facilitate multiprocessing.

# Examples
```julia-repl
julia> inputs = hcat(rand(100,3)*1000.,ones(100)); # 100 randomly distributed 3D particles in [0, 1000], plus weights
julia> npcf = NPCF(N=4, D=3); # create a 4PCF object in 3D
julia> output_pairwise = compute_npcf_pairwise(inputs, npcf); # run the pairwise NPCF estimator
```
"""
function compute_npcf_pairwise(inputs::Matrix{Float64}, npcf::NPCF)

    # First check inputs
    check_inputs(inputs, npcf)

    Npart = length(inputs[:,1])
    if npcf.verb; @printf("\nAnalyzing dataset with %d particles using %d worker(s) by summing over all N-tuplets of galaxies\n",Npart,nworkers()); end

    # Set up output array to hold NPCF
    output = create_npcf_array(npcf)

    # Compute array of weights
    if npcf.N>2
        if npcf.verb; println("Loading coupling weights"); end
        coupling_weights = create_weight_array(npcf)
    else
        # create an empty array for compatibility
        coupling_weights = []
    end

    # Work out how many workers are present and chunk memory
    tasks_per_worker = div(Npart,nworkers())+1

    # Now perform the chunking and run the algorithm across separate workers
    if npcf.verb;
        if nworkers()>1
            println("Starting distributed computation");
        else
            println("Starting computation");
        end
    end
    chunked_output = Dict{Int,Future}()
    @sync for (w_i,w) in enumerate(workers())
        imin = tasks_per_worker*(w_i-1)+1
        imax = min(imin+tasks_per_worker,Npart+1)-1
        @async chunked_output[w_i] = @spawnat w _compute_npcf_pairwise(imin, imax, inputs, coupling_weights, npcf)
    end

    # Now combine the output together
    for w_i in 1:nworkers()
        output .+= chunked_output[w_i][]
    end

    # Apply normalization to the output array
    normalize_npcf!(output, npcf)

    if npcf.verb; @printf("Calculations complete!\n"); end

    return output
end;

## Worker Functions

"""
    _compute_npcf_simple(imin,imax,inputs,npcf)

Compute the NPCF by summing over `N`-tuplets of particles given in the `inputs` dataset using primary particles `imin` to `imax`.
This will run the code for a single worker; see ```compute_npcf_simple()``` for the wrapper and documentation.
"""
function _compute_npcf_simple(imin::Int64, imax::Int64, inputs::Matrix{Float64}, npcf::NPCF)

    Npart = length(inputs[:,1])
    if npcf.verb;
        if nworkers()>1
            println("Running on thread id $(myid()) for particles $imin to $imax");
        else
            println("Iterating over primary particles");
        end
    end

    # Set up output array for this worker
    output = create_npcf_array(npcf)

    # Define positions and weights
    positions = inputs[:,1:npcf.D]
    weights = inputs[:,npcf.D+1]

    # Iterate over first particle in required range
    for i in imin:imax

        p1 = positions[i,:]
        w1 = weights[i]

        if npcf.coords=="spherical"
            # Reorient points such that p1 is at the origin
            shift_positions = reorient_points(positions,p1)
        else
            # Translate positions such that p1 is at the origin
            shift_positions = positions
            for ii in 1:npcf.D
                shift_positions[:,ii] .-= p1[ii]
            end
            if npcf.periodic
                periodic_wrapping!(shift_positions,npcf)
            end
        end

        # Iterate over all secondary particles
        for j in 1:Npart
            p2 = shift_positions[j,:]
            w12 = w1*weights[j]

            # Find separation bin
            sep12 = sep(p2,npcf)
            bin12 = return_bin(sep12,npcf)
            if bin12 == -1; continue; end

            # Accumulate 2PCF
            if npcf.N==2
                output[bin12] += w12
                continue
            end

            # Define unit vector
            if npcf.coords=="cartesian"; p2 /= sep12; end

            # Iterate over third particle
            for k in 1:Npart
                p3 = shift_positions[k,:]
                w123 = w12*weights[k]

                if j==k; continue; end

                # Find separation bin
                sep13 = sep(p3,npcf)
                bin13 = return_bin(sep13,npcf)
                if bin13==-1 || bin13<=bin12; continue; end

                # Define unit vector
                if npcf.coords=="cartesian"; p3 /= sep13; end

                if npcf.N==3
                    # Compute NPCF contribution

                    for l in 0:npcf.lmax
                        # Find basis vector
                        basis = basis_3pcf(l,p2,p3,npcf)

                        # Add to output array
                        output[bin12,bin13,l+1] += w123*conj(basis)
                    end
                else
                    # Iterate over fourth particle
                    for l in 1:Npart
                        p4 = shift_positions[l,:]
                        w1234 = w123*weights[l]

                        if k==l; continue; end

                        # Find separation bin
                        sep14 = sep(p4,npcf)
                        bin14 = return_bin(sep14,npcf)
                        if bin14==-1 || bin14<=bin13; continue; end

                        # Define unit vector
                        if npcf.coords=="cartesian"; p4/=sep14; end

                        if npcf.N==4
                            # Compute NPCF contribution
                            if npcf.D==2
                                l_index = 1
                                for l1_1 in -npcf.lmax:npcf.lmax
                                    for l2_1 in -npcf.lmax:npcf.lmax
                                        l3_1 = -l1_1-l2_1
                                        if abs(l3_1)>npcf.lmax || l3_1>0; continue; end

                                        # Find basis vector
                                        basis = basis_4pcf(l1_1,l2_1,l3_1,p2,p3,p4,npcf)

                                        # Add to output array
                                        output[bin12,bin13,bin14,l_index] += w1234*conj(basis)
                                        l_index += 1
                                    end
                                end
                            elseif npcf.D>=3
                                l_index  = 1
                                for l1_2 in 0:npcf.lmax
                                    for l2_2 in 0:npcf.lmax
                                        for l3_2 in abs(l1_2-l2_2):min(l1_2+l2_2,npcf.lmax)

                                            # remove odd-parity NPCF contributions
                                            if (-1)^(l1_2+l2_2+l3_2)==-1; continue; end

                                            # Find basis vector
                                            basis = basis_4pcf(l1_2,l2_2,l3_2,p2,p3,p4,npcf)

                                            # Add to output array
                                            output[bin12,bin13,bin14,l_index] += w1234*conj(basis)
                                            l_index += 1
                                        end
                                    end
                                end
                            end
                        else
                            # Iterate over fifth particle
                            for m in 1:Npart
                                p5 = shift_positions[m,:]
                                w12345 = w1234*weights[m]

                                if l==m; continue; end

                                # Find separation bin
                                sep15 = sep(p5,npcf)
                                bin15 = return_bin(sep15,npcf)

                                if bin15==-1 || bin15<=bin14; continue; end

                                # Define unit vector
                                if npcf.coords=="cartesian"; p5/=sep15; end

                                if npcf.N==5
                                    # Compute NPCF contribution
                                    if npcf.D==2
                                        l_index = 1
                                        for l1_1 in -npcf.lmax:npcf.lmax
                                            for l2_1 in -npcf.lmax:npcf.lmax
                                                for l3_1 in -npcf.lmax:npcf.lmax
                                                    l4_1 = -l1_1-l2_1-l3_1
                                                    if abs(l4_1)>npcf.lmax || l4_1>0; continue; end

                                                    # Find basis vector
                                                    basis = basis_5pcf(l1_1,l2_1,l1_1+l2_1,l3_1,l4_1,p2,p3,p4,p5,npcf)

                                                    # Add to output array
                                                    output[bin12,bin13,bin14,bin15,l_index] += w12345*conj(basis)
                                                    l_index += 1
                                                end
                                            end
                                        end
                                    elseif npcf.D>=3
                                        l_index  = 1
                                        for l1_2 in 0:npcf.lmax
                                            for l2_2 in 0:npcf.lmax
                                                for l12_2 in abs(l1_2-l2_2):l1_2+l2_2
                                                    for l3_2 in 0:npcf.lmax
                                                        for l4_2 in abs(l12_2-l3_2):min(l12_2+l3_2,npcf.lmax)

                                                            # remove odd-parity NPCF contributions
                                                            if (-1)^(l1_2+l2_2+l3_2+l4_2)==-1; continue; end

                                                            # Find basis vector
                                                            basis = basis_5pcf(l1_2,l2_2,l12_2,l3_2,l4_2,p2,p3,p4,p5,npcf)

                                                            # Add to output array
                                                            output[bin12,bin13,bin14,bin15,l_index] += w12345*conj(basis)
                                                            l_index += 1
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return output
end;

"""
    _compute_npcf_pairwise(imin, imax, inputs, coupling_weights, npcf)

Compute the NPCF of the `inputs` dataset using primary particles `imin` to `imax` by summing over pairs of particles and utilizing hyperspherical harmonic decompositions.
This will run the code for a single worker; see ```compute_npcf_pairwise()``` for the wrapper and documentation.
"""
function _compute_npcf_pairwise(imin::Int64, imax::Int64, inputs::Matrix{Float64}, coupling_weights, npcf::NPCF)

    Npart = length(inputs[:,1])
    if npcf.verb;
        if nworkers()>1
            println("Running on thread id $(myid()) for particles $imin to $imax");
        else
            println("Iterating over primary particles");
        end
    end

    # Set up output array for this worker
    output = create_npcf_array(npcf)

    # Define positions and weights
    positions = inputs[:,1:npcf.D]
    weights = inputs[:,npcf.D+1]

    # Define a_l array
    nl = length(hyperspherical_harmonic(zeros(npcf.D),npcf))
    al = zeros(ComplexF64,npcf.nbins,nl)

    # Iterate over first particle in required range
    for i in imin:imax
        p1 = positions[i,:]
        w1 = weights[i]

        if npcf.coords=="spherical"
            # Rotate particles such that p1 is at the origin
            shift_positions = reorient_points(inputs,p1)
        else
            # Translate particles such that p1 is at the origin
            shift_positions = positions
            for ii in 1:npcf.D
                shift_positions[:,ii] .-= p1[ii]
            end
            if npcf.periodic
                periodic_wrapping!(shift_positions, npcf)
            end
        end

        # Empty array
        al .= 0

        # Iterate over all secondary particles and compute al array
        for j in 1:Npart
            p2 = shift_positions[j,:]
            w2 = weights[j]

            # Find separation bin
            sep12 = sep(p2,npcf)
            bin12 = return_bin(sep12,npcf)
            if bin12 == -1; continue; end

            # Accumulate 2PCF
            if npcf.N==2
                output[bin12] += w1*w2
                continue
            end

            # Define unit vector
            if npcf.coords=="cartesian"; p2/=sep12; end

            # Define spherical harmonics (all at once)
            yl = hyperspherical_harmonic(p2,npcf)

            # Add these to the al array
            al[bin12,:] += w2*conj(yl[:])
        end

        # Now fill up NPCF array
        if npcf.N==2
            # no further computations needed!
            continue
        elseif npcf.N==3
            for b1 in 1:npcf.nbins
                for b2 in b1+1:npcf.nbins
                    # iterate over primary ells
                    for l1 in 0:npcf.lmax

                        # Compute the NPCF contribution in this bin and set of primary ells
                        contribution = accumulate_npcf_contribution([b1,b2],l1,al,coupling_weights,npcf)

                        # Add to output array
                        if contribution!=0; output[b1,b2,l1+1] += w1*contribution; end

                    end
                end
            end
        elseif npcf.N==4
            # Iterate over radial bins
            for b1 in 1:npcf.nbins
                for b2 in b1+1:npcf.nbins
                    for b3 in b2+1:npcf.nbins
                        if npcf.D==2
                            # Iterate over primary ells
                            l_index = 1
                            for l1_1 in -npcf.lmax:npcf.lmax
                                for l2_1 in -npcf.lmax:npcf.lmax
                                    if abs(l1_1+l2_1)>npcf.lmax || l1_1+l2_1<0; continue; end

                                    # Compute the NPCF contribution in this bin and set of primary ells
                                    contribution = accumulate_npcf_contribution([b1,b2,b3],[l1_1,l2_1],al,coupling_weights,npcf)

                                    # Add to output array
                                    if contribution!=0; output[b1,b2,b3,l_index] += w1*contribution; end
                                    l_index += 1

                                end
                            end
                        else
                            # Iterate over primary ells for D>2
                            l_index = 1
                            for l1 in 0:npcf.lmax
                                for l2 in 0:npcf.lmax
                                    for l3 in abs(l1-l2):min(l1+l2,npcf.lmax)
                                        # skip parity-odd contributions
                                        if (-1)^(l1+l2+l3)==-1; continue; end

                                        # Compute the NPCF contribution in this bin and set of primary ells
                                        contribution = accumulate_npcf_contribution([b1,b2,b3],[l1,l2,l3],al,coupling_weights,npcf)

                                        # Add to output array
                                        if contribution!=0; output[b1,b2,b3,l_index] += w1*contribution; end
                                        l_index += 1
                                    end
                                end
                            end
                        end
                    end
                end
            end
        elseif npcf.N==5
            # Iterate over radial bins
            for b1 in 1:npcf.nbins
                for b2 in b1+1:npcf.nbins
                    for b3 in b2+1:npcf.nbins
                        for b4 in b3+1:npcf.nbins
                            if npcf.D==2
                                # Iterate over primary ells
                                l_index = 1
                                for l1_1 in -npcf.lmax:npcf.lmax
                                    for l2_1 in -npcf.lmax:npcf.lmax
                                        for l3_1 in -npcf.lmax:npcf.lmax

                                            if abs(l1_1+l2_1+l3_1)>npcf.lmax || l1_1+l2_1+l3_1<0; continue; end

                                            # Compute the NPCF contribution in this bin and set of primary ells
                                            contribution = accumulate_npcf_contribution([b1,b2,b3,b4],[l1_1,l2_1,l3_1],al,coupling_weights,npcf)

                                            # Add to output array
                                            if contribution!=0; output[b1,b2,b3,b4,l_index] += w1*contribution; end
                                            l_index += 1
                                        end
                                    end
                                end
                            else
                                # Iterate over primary ells for D>2
                                l_index = 1
                                for l1 in 0:npcf.lmax
                                    for l2 in 0:npcf.lmax
                                        for l12 in abs(l1-l2):(l1+l2)
                                            for l3 in 0:npcf.lmax
                                                for l4 in abs(l12-l3):min(l12+l3,npcf.lmax)
                                                    # skip parity-odd contributions
                                                    if (-1)^(l1+l2+l3+l4)==-1; continue; end

                                                    # Compute the NPCF contribution in this bin and set of primary ells
                                                    contribution = accumulate_npcf_contribution([b1,b2,b3,b4],[l1,l2,l12,l3,l4],al,coupling_weights,npcf)

                                                    # Add to output array
                                                    if contribution!=0; output[b1,b2,b3,b4,l_index] += w1*contribution; end
                                                    l_index += 1
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        else
            error("Not yet implemented!")
        end
    end

    return output
end;


### Other functions

"""
    summarize(npcf)

Print a summary of the NPCF hyperparameters
"""
function summarize(npcf::NPCF)
    @printf("Computing the %dPCF in %d dimensions\n\n",npcf.N,npcf.D)
    @printf("Coordinates = %s\n",npcf.coords)
    @printf("Periodic = %s\n",npcf.periodic)
    if npcf.periodic
        @printf("Periodic boxsize: %f\n",npcf._boxsize)
    end
    @printf("Radial bins: [%.2f, %.2f, %.2f]\n",npcf.r_min,npcf.r_max,npcf._dr)
    @printf("l-max: %d\n",npcf.lmax)
end

"""
    angular_indices(npcf)

Return a list of primary angular indices corresponding to the collapsed indices used in the output of the ```compute_npcf_simple``` and ```compute_npcf_pairwise``` functions.
For `N = 3`, we return a single list of `l1`, for `N = 4`, we return a list of `l1, l2, l3`, and for `N = 5`, we return a list of `l1, l2, l12, l3, l4`.
"""
function angular_indices(npcf::NPCF)
    if npcf.N==2
        return error("Isotropic 2PCF contains no angular indices!")
    elseif npcf.N==3
        l1s = Int64[]

        # Fill up array of ells
        l_index = 1
        for l1 in 0:npcf.lmax
            push!(l1s,l1)
        end

        return l1s

    elseif npcf.N==4
        l1s = Int64[]
        l2s = Int64[]
        l3s = Int64[]

        if npcf.D==2

            # Fill up array of ells
            for l1 in -npcf.lmax:npcf.lmax
                for l2 in -npcf.lmax:npcf.lmax
                    l3 = -l1-l2
                    if abs(l3)>npcf.lmax || l3>0; continue; end

                    push!(l1s,l1)
                    push!(l2s,l2)
                    push!(l3s,l3)
                end
            end

        elseif npcf.D>=3

            # Fill up array of ells
            for l1 in 0:npcf.lmax
                for l2 in 0:npcf.lmax
                    for l3 in abs(l1-l2):min(l1+l2,npcf.lmax)
                        # only even-parity NPCF contributions
                        if (-1)^(l1+l2+l3)==-1; continue; end

                        push!(l1s,l1)
                        push!(l2s,l2)
                        push!(l3s,l3)
                    end
                end
            end
        end

        return l1s, l2s, l3s

    elseif npcf.N==5

        if npcf.D==2
            l1s = Int64[]
            l2s = Int64[]
            l3s = Int64[]
            l4s = Int64[]

            # Fill up array of ells
            for l1 in -npcf.lmax:npcf.lmax
                for l2 in -npcf.lmax:npcf.lmax
                    for l3 in -npcf.lmax:npcf.lmax
                        l4 = -l1-l2-l3
                        if abs(l4)>npcf.lmax || l4>0; continue; end

                        push!(l1s,l1)
                        push!(l2s,l2)
                        push!(l3s,l3)
                        push!(l4s,l4)
                    end
                end
            end
            return l1s,l2s,l3s,l4s

        elseif npcf.D>=3
            l1s = Int64[]
            l2s = Int64[]
            l12s = Int64[]
            l3s = Int64[]
            l4s = Int64[]

            # Fill up array of ells
            for l1 in 0:npcf.lmax
                for l2 in 0:npcf.lmax
                    for l12 in abs(l1-l2):l1+l2
                        for l3 in 0:npcf.lmax
                            for l4 in abs(l12-l3):min(l12+l3,npcf.lmax)
                                # only even-parity NPCF contributions
                                if (-1)^(l1+l2+l3+l4)==-1; continue; end

                                push!(l1s,l1)
                                push!(l2s,l2)
                                push!(l12s,l12)
                                push!(l3s,l3)
                                push!(l4s,l4)
                            end
                        end
                    end
                end
            end
            return l1s,l2s,l12s,l3s,l4s
        end
    end
end

end
