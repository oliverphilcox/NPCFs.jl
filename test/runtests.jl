# Load the NPCF code
println("Loading NPCF module")
using NPCFs, Test, Distributed
Npart = 50 # number of particles
boxsize = 200 # input size

## Single thread runtest, comparing NPCF simple + pairwise outputs for different N and D values
println("Single-worker runtest")
for N in 2:5
    for D in 2:4
        println("Testing Cartesian N=$N, D=$D")
        npcf = NPCF(N=N,D=D,verb=true,periodic=true,volume=boxsize^D,coords="cartesian",r_min=20,r_max=95,nbins=6,lmax=1);

        pos = hcat(rand(Npart,D)*boxsize,ones(Npart)); # columns: [x, y, z, weight]

        npcf_output1 = compute_npcf_simple(pos, npcf);
        npcf_output2 = compute_npcf_pairwise(pos, npcf);

        # Compute the error
        err = sum(npcf_output1[:]-npcf_output2[:])/sum(npcf_output2[:]);
        println("Error is $err\n")

        @test abs(err)<1e-10
    end
end

## As above, but for spherical geometry in 2 dimensions
println("Spherical geometry runtest")
for N in 2:5
    println("Testing spherical N=$N, D=2")
    npcf = NPCF(N=N,D=2,verb=true,periodic=false,volume=4pi,coords="spherical",r_min=-0.8,r_max=0.8,nbins=6,lmax=1);

    phi_arr = rand(Npart)*2pi
    theta_arr = acos.(rand(Npart).*2 .-1)
    pos = hcat(phi_arr,theta_arr,ones(Npart));

    npcf_output1 = compute_npcf_simple(pos, npcf);
    npcf_output2 = compute_npcf_pairwise(pos, npcf);

    # Compute the error
    err = sum(npcf_output1[:]-npcf_output2[:])/sum(npcf_output2[:]);
    println("Error is $err\n")

    @test abs(err)<1e-10
end

## As above, but for an aperiodic geometry
println("Aperiodic box runtest")
for N in 2:5
    for D in 2:4
        println("Testing Cartesian N=$N, D=$D")
        npcf = NPCF(N=N,D=D,verb=true,periodic=false,volume=boxsize^D,coords="cartesian",r_min=20,r_max=95,nbins=6,lmax=1);

        pos = hcat(rand(Npart,D)*boxsize,ones(Npart)); # columns: [x, y, z, weight]

        npcf_output1 = compute_npcf_simple(pos, npcf);
        npcf_output2 = compute_npcf_pairwise(pos, npcf);

        # Compute the error
        err = sum(npcf_output1[:]-npcf_output2[:])/sum(npcf_output2[:]);
        println("Error is $err\n")

        @test abs(err)<1e-10
    end
end

## Multiple worker runtest
# Here we use N = D = 3, and run on two workers
println("Multi-worker runtest")
addprocs(2) # run on two workers
@everywhere using NPCFs
npcf = NPCF(N=3,D=3,verb=true,periodic=true,volume=boxsize^3,coords="cartesian",r_min=20,r_max=95,nbins=6,lmax=1);

pos = hcat(rand(Npart,3)*boxsize,ones(Npart)); # columns: [x, y, z, weight]

npcf_output1 = compute_npcf_simple(pos, npcf);
npcf_output2 = compute_npcf_pairwise(pos, npcf);

# Compute the error
err = sum(npcf_output1[:]-npcf_output2[:])/sum(npcf_output2[:]);
println("Error is $err\n")

@test abs(err)<1e-10

println("All tests passed!")
