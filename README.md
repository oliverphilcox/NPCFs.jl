# NPCFs.jl

Efficient computation of isotropic N-point correlation function (NPCF) in Julia. This implements two algorithms to compute the NPCF of n particles in D dimensions: (1) a naive count over N-tuplets of particles, with complexity O(n^N), and (2) the O(n^2) algorithm of [Philcox & Slepian 2021](https://arxiv.org/abs/2106.10278), which makes use of hyperspherical harmonics to convert the computation into a pair count. In both cases, we compute the NPCF in the N-point basis discussed in [Philcox & Slepian 2021](https://arxiv.org/abs/2106.10278).

#### Features
- N-Point Correlation Functions with N = 2, 3, 4, 5
- Cartesian geometries in D = 2, 3, 4 dimensions, optionally with periodic boundary conditions
- Spherical geometries in D = 2 dimensions
- Arbitrary particle weights
- Distributed computation over arbitrary numbers of processors

#### Notes
- In spherical coordinates, we parametrize by `phi, theta` polar coordinates with `phi` in [0, 2pi), `theta` in [0, pi).
- Radial bins are equally spaced in `cos(sigma)` where `sigma` is the great-circle angle between two points on the 2-sphere
- For each primary particle, all secondary particles are shifted to put the primary particle at the origin
- For ```D=2```, we use only real basis functions (*i.e.* those with `ell^(1) + ell^(2) + ... >= 0`)

#### Current Limitations
- Only even-parity basis functions are computed (*i.e.* those with even `ell^(1) + ell^(2) + ...`)
- As yet, support is not included for *anisotropic* basis functions

## Installation

To install the ```NPCFs.jl``` package simply run the following in a Julia terminal:
```julia
] add "https://github.com/oliverphilcox/NPCFs.jl"
```

The package can then be loaded using ```using NPCFs```, as usual.

## Quickstart Examples

#### 1) Compute the 3PCF of 3D particles on a single node

We first load the relevant modules:

```julia
# Load the NPCF code
using NPCFs

# Load additional packages for testing
using Statistics, Random, Printf
```

Next, initialize the NPCF structure with the relevant parameters. Here, we'll assume a 3D periodic box of size 1000 in Cartesian coordinates. We'll use 10 radial bins in the range [50, 200], and `lmax` of 5.

```julia
boxsize = 1000
npcf = NPCF(N=3,D=3,verb=true,periodic=true,volume=boxsize^3,
            coords="cartesian",r_min=50,r_max=200,nbins=10,lmax=5);
```

Now let's create some data (i.e. particle positions and weights) with the above specifications. Let's use 500 particles:

```julia
pos = hcat(rand(500,3)*boxsize,ones(500)); # columns: [x, y, z, weight]
```

We can now run the code, using both simple and pairwise estimators:
```julia
npcf_output1 = compute_npcf_simple(pos, npcf);
npcf_output2 = compute_npcf_pairwise(pos, npcf);

# Compute the error
mean_err = mean(npcf_output1-npcf_output2)/mean(npcf_output2);
std_err = std(npcf_output1-npcf_output2)/mean(npcf_output2);
@printf("\nFractional Error: %.1e +- %.1e",mean_err,std_err)
```

Now we wait for the code to run and look at the output. This is an array of shape `(nbins, nbins, n_angular)` for the 3PCF, where the first two columns give the index of the first and second radial bin (filling only entries with `bin2>bin1`), and the final column gives the angular information (here indexing the `l` array).

Two other functions might be of use:
1. ```summarize(npcf)```: Print a summary of the `NPCF` parameters in use
2. ```angular_indices(npcf)```: Return lists of the angular indices used in the final column of the `npcf_output{X}` arrays. For example, for the 3PCF (4PCF), this returns a list of `l` (`l1`, `l2` and `l3`), in flattened form.

#### 2) Compute the 4PCF of 2D particles on a sphere, with distributed computing

To use distributed computing, we'll need to load the `NPCFs` module both on the main process and 4 workers:
```julia
using Distributed
addprocs(4) # add 4 workers
println("Using $nworkers() workers")

# Load the NPCF code both locally and on workers.
using NPCFs
@everywhere using NPCFs

# Load additional packages for testing
using Statistics, Random, Printf
```

Next, we initialize the NPCF structure and create some data, here spherical coordinates (`theta` and `phi`) of particles randomly positioned on the 2-sphere. The radial bins are now equal to the cosine of the angular distance along the two-sphere connecting two points, and are restricted to [-1,1]:
```julia
npcf = NPCF(N=4,D=2,verb=true,periodic=false,volume=4pi,coords="spherical",r_min=-0.5,r_max=0.5,nbins=10,lmax=2);

Npart = 500
phi_arr = rand(Npart)*2pi # uniform in [0, 2pi)
theta_arr = acos.(rand(Npart).*2 .-1) # cos(theta) is uniform in [-1, 1)
pos = hcat(phi_arr,theta_arr,ones(Npart));
```

Now run the code as before. No additional information is required to specify that we're using distributed computing; the code will figure this out automatically, and chunk the operations across all available workers.
```julia
npcf_output1 = compute_npcf_simple(pos, npcf);
npcf_output2 = compute_npcf_pairwise(pos, npcf);

# Compute the error
mean_err = mean(npcf_output1-npcf_output2)/mean(npcf_output2)
std_err = std(npcf_output1-npcf_output2)/mean(npcf_output2)
@printf("\nFractional Error: %.1e +- %.1e",mean_err,std_err)
```

The output takes a similar form to before; an array of size `(nbins, nbins, nbins, n_angular)`, where the first three columns give the radial bins (with `bin3>bin2>bin1`), and the fourth gives the angular index, which can be translated to `l1, l2, l3` indices using ```l1, l2, l3 = angular_indices(npcf)```.

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Princeton / IAS)
