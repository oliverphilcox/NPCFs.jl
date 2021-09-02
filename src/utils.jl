
"""
    check_inputs(inputs, npcf)

Check the input array of particles, and raise an error if it is of the wrong format.
"""
function check_inputs(inputs, npcf::NPCF)
    if npcf.coords=="spherical"
        @assert (maximum(inputs[:,1])<2pi && minimum(inputs[:,1])>=0) "Phi coordinate must be in range [0, 2pi)"
        @assert (maximum(inputs[:,2])<pi && minimum(inputs[:,2])>=0) "Theta coordinate must be in range [0, pi)"
    else
        if npcf.periodic
            @assert maximum(inputs[:,1:npcf.D])<=npcf._boxsize "Maximum position must be less than boxsize"
            @assert minimum(inputs[:,1:npcf.D])>=0. "Minimum position must be lessgreater than zero"
        end
    end
    @assert length(inputs[1,:])==npcf.D+1 "Inputs should have (D+1) values per particles!"
end

"""
    sep(p12, npcf)

Compute the 1D separation of two particles with separation vector `p12` in Cartesian or Spherical geometries.
For spherical geometries, we return the cosine of the angular separation along the great circle connecting the two points.
In this case, we assume that the separation vector is the coordinates of the second point with the first at the origin.
"""
@inline function sep(p12,npcf::NPCF)
    if npcf.coords=="cartesian"
        return sqrt(sum(p12.^2))
    elseif npcf.coords=="spherical"
        return cos(p12[2])
    else
        return error("unrecognized coordinate system!")
    end
end

"""
    periodic_wrapping!(sep_array, npcf)

Add periodic wrapping to an array of separation vectors `sep_array`.
"""
@inline function periodic_wrapping!(sep_array,npcf::NPCF)
    for k in length(sep_array[:,1])
        for i in 1:npcf.D
            while sep_array[k,i]<-npcf._boxsize/2
                sep_array[k,i] += npcf._boxsize
            end
            while sep_array[k,i]>npcf._boxsize/2
                sep_array[k,i] -= npcf._boxsize
            end
        end
    end
    return sep_array
end

"""
    reorient_points(input, origin)

Rotate a set of input points (`input`) such that a given origin point (`origin`) is on the z-axis.
This applies only to points on the 2-sphere in spherical coordinates.
"""
@inline function reorient_points(input,origin)

    ## First generate a rotation matrix
    X=[sin(origin[2])cos(origin[1]),sin(origin[2])sin(origin[1]),cos(origin[2])]

    # Generate rotation axis
    axis=cross(X,[0,0,1])
    axis/=norm(axis)

    # Compute rotation matrix
    rotation_matrix = AngleAxis(origin[2],axis[1],axis[2],axis[3])

    ## Now rotate the input grid
    output = copy(input)
    for i in 1:length(input[:,1])
        phi = input[i,1]
        th = input[i,2]
        # Put into Cartesian coordinates
        cartesian_form = [sin(th)cos(phi),sin(th)sin(phi),cos(th)]
        # Apply rotation
        rot_cartesian = rotation_matrix*cartesian_form
        # Revert to polar coordinates
        rot_cartesian/=norm(rot_cartesian)
        output[i,1:2] = [atan(rot_cartesian[2],rot_cartesian[1]),acos(rot_cartesian[3])]
    end
    return output
end

"""
    return_bin(sep, npcf)

Compute the bin corresponding to a given separation, `sep`.
"""
@inline function return_bin(sep,npcf::NPCF)
    if sep<npcf.r_min
        return -1
    elseif sep>npcf.r_max
        return -1
    else
        return Int64(div(sep-npcf.r_min,npcf._dr)+1)
    end
end

"""
    computePlm(x, l, m)

Compute associated Legendre polynomial at `x` for integer or half-integer `l`,`m`
"""
function computePlm(x,l,m)
    eps = 1e-12
    1/gamma(1-m-eps)*_₂F₁(-l+eps,l+1+eps,1-m-eps,(1-x)/2)*((1+x)/(1-x))^(m/2)
end

"""
    computeYl(theta_array, l_array, NPCF)

Compute hyperspherical harmonic for input arrays of theta (`theta_array`) and ell (`l_array`)
"""
function computeYl(theta_array,l_array,npcf::NPCF)
    output = (-1)^(l_array[1])/sqrt(2*pi)*exp(im*l_array[1]*theta_array[1])
    for j in 2:npcf.D-1
        L = l_array[j]
        l = l_array[j-1]
        jPlL = sqrt((2L+j-1)/2*gamma(L+l+j-1)/gamma(L-l+1))*sin(theta_array[j])^((2-j)/2)*computePlm(cos(theta_array[j]),L+(j-2)/2,-(l+(j-2)/2))
        output *= jPlL
    end
    return output
end

"""
    hyperspherical_harmonic_single(r12, l_arr, npcf)

Return hyperspherical harmonics of particle, given position vector `r12` and single ell array `l_arr`.
For `D = 3` these are just the usual spherical harmonics.
In spherical coordinates, `r12` are the coordinates of the second particle, rotated such that the first particle is at the origin.
"""
function hyperspherical_harmonic_single(r12,l_arr,npcf::NPCF)

    if npcf.D==2
        # First define the angles
        if npcf.coords=="cartesian"
            theta1 = atan(r12[2],r12[1])
        elseif npcf.coords=="spherical"
            theta1 = r12[1]
        end
        return exp(im*l_arr[1]*theta1)/sqrt(2*pi)
    elseif npcf.D==3
        # First define the angles
        theta2 = acos(r12[3]) # i.e. theta
        theta1 = atan(r12[2],r12[1]) # i.e. phi

        # Compute P_l^m(theta) with normalization
        Plm = computePlmcostheta(theta2,l_arr[2],abs(l_arr[1]))[(l_arr[2],abs(l_arr[1]))]
        if l_arr[1]<0
            phase = (-1)^l_arr[1]
        else
            phase = 1
        end
        return phase/sqrt(2)*exp(im*l_arr[1]*theta1)*Plm
    elseif npcf.D==4
        # First define angles
        theta3 = acos(r12[4])
        theta2 = atan(sqrt(r12[2]^2+r12[1]^2),r12[3])
        theta1 = atan(r12[2],r12[1])

        # Assemble hyperspherical harmonics
        return computeYl([theta1,theta2,theta3],l_arr,npcf)
    else
        error("Not yet defined!")
    end
end

"""
    hyperspherical_harmonic(r12, npcf)

Return all hyperspherical harmonics of particle up to `lmax`, given Cartesian position vector `r12`.
For `D = 3`, these are just the usual spherical harmonics.
In spherical coordinates, `r12` are the coordinates of the second particle, rotated such that the first particle is at the origin.
"""
function hyperspherical_harmonic(r12,npcf::NPCF)

    if npcf.D==2
        # First define the angles
        if npcf.coords=="cartesian"
            theta1 = atan(r12[2],r12[1])
        elseif npcf.coords=="spherical"
            theta1 = r12[1]
        end

        # Assemble hyperspherical harmonics
        Ylm = zeros(ComplexF64,2*npcf.lmax+1)
        l_index = 1
        for l1 in -npcf.lmax:npcf.lmax
            Ylm[l_index] = exp(im*l1*theta1)/sqrt(2*pi)
            l_index += 1
        end
    elseif npcf.D==3
        # First define the angles
        theta2 = acos(r12[3]) # i.e. theta
        theta1 = atan(r12[2],r12[1]) # i.e. phi

        # Compute P_l^m(theta) with normalization
        Plm = computePlmcostheta(theta2,npcf.lmax)

        # Assemble hyperspherical harmonics
        Ylm = zeros(ComplexF64,(npcf.lmax+1)^2)
        l_index = 1
        for l_2 in 0:npcf.lmax # i.e. l
            for l_1 in -l_2:l_2 # i.e. m
                if l_1<0
                    phase = (-1)^l_1
                else
                    phase = 1
                end
                Ylm[l_index] = phase/sqrt(2)*exp(im*l_1*theta1)*Plm[(l_2,abs(l_1))]
                l_index += 1
            end
        end
    elseif npcf.D==4
        # First define angles
        theta3 = acos(r12[4])
        theta2 = atan(sqrt(r12[2]^2+r12[1]^2),r12[3])
        theta1 = atan(r12[2],r12[1])

        # Assemble hyperspherical harmonics
        Ylm = zeros(ComplexF64,Int64((npcf.lmax+1)*(npcf.lmax+2)*(2*npcf.lmax+3)/6))
        l_index = 1
        for l_3 in 0:npcf.lmax
            for l_2 in 0:l_3
                for l_1 in -l_2:l_2 # i.e. m
                    Ylm[l_index] = computeYl([theta1,theta2,theta3],[l_1,l_2,l_3],npcf)
                    l_index += 1
                end
            end
        end
    else
        error("Not yet defined!")
    end
    return Ylm
end

"""
    wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)

Define the Wigner 9j function in terms of the Wigner 6j functions
"""
function wigner9j(j1,j2,j3,j4,j5,j6,j7,j8,j9)
   output = 0
   for x in max(abs(j1-j9),abs(j4-j8),abs(j2-j6)):min(abs(j1+j9),abs(j4+j8),abs(j2+j6))
        tmp = (-1)^(2x)*(2x+1)*wigner6j(j1,j4,j7,j8,j9,x)*wigner6j(j2,j5,j8,j4,x,j6)*wigner6j(j3,j6,j9,x,j1,j2)
        output += tmp
    end
    return output
end

"""
    clebsch_gordan(l1_arr, l2_arr, L_arr, D)

Return the Clebsch-Gordan coefficient for three arrays of ells, `l1_arr`, `l2_arr` and `L_arr` in `D` dimensions.
"""
function clebsch_gordan(l1_arr,l2_arr,L_arr,D)
    if D==2
        (l1_arr+l2_arr)==L_arr ? 1. : 0.
    elseif D==3
        return (-1)^(-l1_arr[2]+l2_arr[2]-L_arr[1])*sqrt(2*L_arr[2]+1)*wigner3j(l1_arr[2],l2_arr[2],L_arr[2],l1_arr[1],l2_arr[1],-L_arr[1])
    elseif D==4
        pref = (-1)^(-l1_arr[2]+l2_arr[2]-L_arr[1])*(L_arr[3]+1)*sqrt((2*l1_arr[2]+1)*(2*l2_arr[2]+1)*(2*L_arr[2]+1))
        threej = wigner3j(l1_arr[2],l2_arr[2],L_arr[2],l1_arr[1],l2_arr[1],-L_arr[1])
        ninej = wigner9j(l1_arr[3]/2,l2_arr[3]/2,L_arr[3]/2,l1_arr[3]/2,l2_arr[3]/2,L_arr[3]/2,l1_arr[2],l2_arr[2],L_arr[2])
        return pref*threej*ninej
    else
        return error("not yet implemented!")
    end
end

"""
    basis_3pcf(l, r1, r2, npcf)

Return isotropic basis function of separation vectors `r1` and `r2` with primary index `l`.
For `D = 2`, this is a rescaled Chebyshev polynomial.
For `D = 3`, this is a rescaled Legendre polynomial.
For `D = 4`, we just use the spherical harmonic representation, for simplicity.
For spherical coordinates, we assume that `r1` and `r2` are rotated such that their polar angles give the opening angle of the triangle.
"""
function basis_3pcf(l,r1,r2,npcf::NPCF)
    if npcf.coords=="cartesian"
        mu = sum(r1.*r2)
    end

    if npcf.D==2
        if npcf.coords=="cartesian"
            return 1/(2*pi)*chebyshevt(l,mu)
        elseif npcf.coords=="spherical"
            phi = r1[1]-r2[1]
            return 1/(2*pi)*cos(l*phi)
        end
    elseif npcf.D==3
        return Pl(mu,l)*(-1)^l*sqrt(2*l+1)/(4*pi)
    elseif npcf.D==4
        if l==0
            return 1/(2pi^2)
        else
            # Define hyperspherical harmonics
            Yl1 = hyperspherical_harmonic(r1,npcf)
            Yl2 = hyperspherical_harmonic(r2,npcf)

            # Sum over intermediate states
            partial_sum = 0.
            for l_2 in 0:l
                for l_1 in -l_2:l_2
                    ax1 = Int64(1/6*l*(1+l)*(1+2l)+l_2*(l_2+1)+l_1+1)
                    ax2 = Int64(1/6*l*(1+l)*(1+2l)+l_2*(l_2+1)-l_1+1)
                    partial_sum += (-1)^(l_2-l_1)/(l+1)*real(Yl1[ax1]*Yl2[ax2])
                end
            end
            return partial_sum
        end
    else
        return error("not yet implemented!")
    end
end

"""
    basis_4pcf(l1, l2, l3, r1, r2, r3, npcf)

Return isotropic basis function of separation vectors `r1`, `r2` and `r3` with primary indices `l1`, `l2`, `l3`.
For `D = 2` it is simpler (and not inefficient) to work in polar coordinates, though a representation can be found in terms of dot products.
For `D = 3` and `D = 4`, we construct the basis from the spherical harmonics, which is straightforward, though more efficient representations can be found in terms of Cartesian dot products.
"""
function basis_4pcf(l1,l2,l3,r1,r2,r3,npcf::NPCF)

    if npcf.D==2
        # Cases with a zero are easy!
        if l1==0
            return (l2+l3==0)*basis_3pcf(abs(l2),r2,r3,npcf)/sqrt(2pi)
        elseif l2==0
            return (l1+l3==0)*basis_3pcf(abs(l1),r1,r3,npcf)/sqrt(2pi)
        elseif l3==0
            return (l1+l2==0)*basis_3pcf(abs(l1),r1,r2,npcf)/sqrt(2pi)
        end

        if npcf.coords=="cartesian"
            theta1 = atan(r1[2],r1[1])
            theta2 = atan(r2[2],r2[1])
            theta3 = atan(r3[2],r3[1])
        elseif npcf.coords=="spherical"
            theta1 = r1[1]
            theta2 = r2[1]
            theta3 = r3[1]
        end

        return cos(l1*theta1+l2*theta2+l3*theta3)/(2pi)^(3/2)
    elseif npcf.D==3
        # Cases with a zero are easy!
        if l1==0
            return (l2==l3)*basis_3pcf(l2,r2,r3,npcf)/sqrt(4pi)
        elseif l2==0
            return (l1==l3)*basis_3pcf(l1,r1,r3,npcf)/sqrt(4pi)
        elseif l3==0
            return (l1==l2)*basis_3pcf(l1,r1,r2,npcf)/sqrt(4pi)
        end

        output = 0.
        for l1_1 in -l1:l1
            Y1 = hyperspherical_harmonic_single(r1,[l1_1,l1],npcf)
            for l2_1 in -l2:l2
                Y2 = hyperspherical_harmonic_single(r2,[l2_1,l2],npcf)

                l3_1=-l1_1-l2_1
                if abs(l3_1)>l3; continue; end

                clebsch = clebsch_gordan([l1_1,l1],[l2_1,l2],[-l3_1,l3],npcf.D)
                clebsch *= (-1)^(l3-l3_1)/sqrt(2l3+1) # final CG coefficient
                if clebsch==0; continue; end

                Y3 = hyperspherical_harmonic_single(r3,[l3_1,l3],npcf)
                output += clebsch*real(Y1*Y2*Y3)
            end
        end
        return output

    elseif npcf.D==4
        # Cases with a zero are easy!
        if l1==0
            return (l2==l3)*basis_3pcf(l2,r2,r3,npcf)/(sqrt(2)pi)
        elseif l2==0
            return (l1==l3)*basis_3pcf(l1,r1,r3,npcf)/(sqrt(2)pi)
        elseif l3==0
            return (l1==l2)*basis_3pcf(l1,r1,r2,npcf)/(sqrt(2)pi)
        end

        output = 0.
        for l1_2 in 0:l1
            for l1_1  in -l1_2:l1_2
                Y1 = hyperspherical_harmonic_single(r1,[l1_1,l1_2,l1],npcf)
                for l2_2 in 0:l2
                    for l2_1 in -l2_2:l2_2
                        Y2 = hyperspherical_harmonic_single(r2,[l2_1,l2_2,l2],npcf)
                        for l3_2 in 0:l3
                            l3_1 = -l1_1-l2_1
                            if abs(l3_1)>l3_2; continue; end

                            clebsch = clebsch_gordan([l1_1,l1_2,l1],[l2_1,l2_2,l2],[-l3_1,l3_2,l3],npcf.D)
                            clebsch *= (-1)^(l3_2-l3_1)/(l3+1) # final CG coefficient
                            if clebsch==0.; continue; end

                            Y3 = hyperspherical_harmonic_single(r3,[l3_1,l3_2,l3],npcf)
                            output += clebsch*real(Y1*Y2*Y3)
                        end
                    end
                end
            end
        end
        return output
    end
end

"""
    basis_5pcf(l1, l2, l12, l3, l4, r1, r2, r3, r4, npcf)

Return isotropic basis function of `r1`, `r2`, `r3` and `r4` with primary indices `l1`, `l2`, `l12`, `l3`, `l4`.
For simplicity, we generate these from the isotropic basis functions, though representations can be found in terms of Cartesian dot products.
"""
function basis_5pcf(l1,l2,l12,l3,l4,r1,r2,r3,r4,npcf::NPCF)

    if npcf.D==2
        # Cases with a zero are easy!
        if l1==0
            return (l2+l3+l4==0)*basis_4pcf(l2,l3,l4,r2,r3,r4,npcf)/sqrt(2pi)
        elseif l2==0
            return (l1+l3+l4==0)*basis_4pcf(l1,l3,l4,r1,r3,r4,npcf)/sqrt(2pi)
        elseif l3==0
            return (l1+l2+l4==0)*basis_4pcf(l1,l2,l4,r1,r2,r4,npcf)/sqrt(2pi)
        elseif l4==0
            return (l1+l2+l3==0)*basis_4pcf(l1,l2,l3,r1,r2,r3,npcf)/sqrt(2pi)
        end

        if npcf.coords=="cartesian"
            theta1 = atan(r1[2],r1[1])
            theta2 = atan(r2[2],r2[1])
            theta3 = atan(r3[2],r3[1])
            theta4 = atan(r4[2],r4[1])
        elseif npcf.coords=="spherical"
            theta1 = r1[1]
            theta2 = r2[1]
            theta3 = r3[1]
            theta4 = r4[1]
        end
        return cos(l1*theta1+l2*theta2+l3*theta3+l4*theta4)/(2pi)^2
    elseif npcf.D==3
        # Cases with a zero are easy!
        if l1==0
            return (l2==l12)*basis_4pcf(l2,l3,l4,r2,r3,r4,npcf)/sqrt(4pi)
        elseif l2==0
            return (l1==l12)*basis_4pcf(l1,l3,l4,r1,r3,r4,npcf)/sqrt(4pi)
        elseif l3==0
            return (l4==l12)*basis_4pcf(l1,l2,l4,r1,r2,r4,npcf)/sqrt(4pi)
        elseif l4==0
            return (l3==l12)*basis_4pcf(l1,l2,l3,r1,r2,r3,npcf)/sqrt(4pi)
        end

        output = 0.
        for l1_1 in -l1:l1
            Y1 = hyperspherical_harmonic_single(r1,[l1_1,l1],npcf)
            for l2_1 in -l2:l2
                Y2 = hyperspherical_harmonic_single(r2,[l2_1,l2],npcf)
                l12_1 = l1_1+l2_1
                if abs(l12_1)>l12
                    continue
                end
                clebsch1 = clebsch_gordan([l1_1,l1],[l2_1,l2],[l12_1,l12],npcf.D)
                if clebsch1==0
                    continue
                end
                for l3_1 in -l3:l3
                    Y3 = hyperspherical_harmonic_single(r3,[l3_1,l3],npcf)
                    l4_1 = -l12_1-l3_1
                    if abs(l4_1)>l4
                        continue
                    end
                    Y4 = hyperspherical_harmonic_single(r4,[l4_1,l4],npcf)
                    clebsch2 = clebsch_gordan([l12_1,l12],[l3_1,l3],[-l4_1,l4],npcf.D)
                    if clebsch2==0
                        continue
                    end
                    clebsch3 = (-1)^(l4-l4_1)/sqrt(2l4+1) # this one is easy
                    output += clebsch1*clebsch2*clebsch3*real(Y1*Y2*Y3*Y4)
                end
            end
        end
        return output
    elseif npcf.D==4
        # Cases with a zero are easy!
        if l1==0
            return (l2==l12)*basis_4pcf(l2,l3,l4,r2,r3,r4,npcf)/(sqrt(2)pi)
        elseif l2==0
            return (l1==l12)*basis_4pcf(l1,l3,l4,r1,r3,r4,npcf)/(sqrt(2)pi)
        elseif l3==0
            return (l4==l12)*basis_4pcf(l1,l2,l4,r1,r2,r4,npcf)/(sqrt(2)pi)
        elseif l4==0
            return (l3==l12)*basis_4pcf(l1,l2,l3,r1,r2,r3,npcf)/(sqrt(2)pi)
        end

        output = 0.
        for l1_2 in 0:l1
            for l1_1  in -l1_2:l1_2
                Y1 = hyperspherical_harmonic_single(r1,[l1_1,l1_2,l1],npcf)
                for l2_2 in 0:l2
                    for l2_1 in -l2_2:l2_2
                        Y2 = hyperspherical_harmonic_single(r2,[l2_1,l2_2,l2],npcf)

                        for l12_2 in 0:l12
                            l12_1 = l1_1+l2_1
                            if abs(l12_1)>l12_2
                                continue
                            end

                            clebsch1 = clebsch_gordan([l1_1,l1_2,l1],[l2_1,l2_2,l2],[l12_1,l12_2,l12],npcf.D)
                            if clebsch1==0
                                continue
                            end

                            for l3_2 in 0:l3
                                for l3_1 in -l3_2:l3_2
                                    Y3 = hyperspherical_harmonic_single(r3,[l3_1,l3_2,l3],npcf)

                                    for l4_2 in 0:l4
                                        l4_1 = -l12_1-l3_1
                                        if abs(l4_1)>l4_2
                                            continue
                                        end
                                        Y4 = hyperspherical_harmonic_single(r4,[l4_1,l4_2,l4],npcf)
                                        clebsch2 = clebsch_gordan([l12_1,l12_2,l12],[l3_1,l3_2,l3],[-l4_1,l4_2,l4],npcf.D)
                                        if clebsch2==0
                                            continue
                                        end
                                        clebsch3 = (-1)^(l4_2-l4_1)/(l4+1) # this one is easy
                                        output += clebsch1*clebsch2*clebsch3*real(Y1*Y2*Y3*Y4)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        return output
    end
end

"""
    create_weight_array(npcf)

Create the array of angular weights for the NPCF calculation, using the Clebsch-Gordan coefficients.
"""
function create_weight_array(npcf::NPCF)
    if npcf.N==3
        if npcf.D==2
            weight_array = zeros(npcf.lmax+1)
            for l1_1 in 0:npcf.lmax
                # assuming l1_1 + l2_1 = 0, and l2_1<=0
                weight_array[l1_1+1] = 1
            end
        elseif npcf.D==3
            weight_array = zeros((npcf.lmax+1)^2)
            l_index = 1
            for l1_2 in 0:npcf.lmax
                for l1_1 in -l1_2:l1_2
                    # assuming l2_2 = l1_2 and l1_1 + l2_1 = 0
                    weight_array[l_index] = (-1)^(l1_2-l1_1)/sqrt(2l1_2+1)
                    l_index += 1
                end
            end
        elseif npcf.D==4
            weight_array = zeros(Int64((1+npcf.lmax)*(2+npcf.lmax)*(3+2npcf.lmax)/6))
            l_index = 1
            for l1_3 in 0:npcf.lmax
                for l1_2 in 0:l1_3
                    for l1_1 in -l1_2:l1_2
                        # assuming l1_3 = l2_3, l1_2 = l2_2 and l1_1 + l2_1 = 0
                        weight_array[l_index] = (-1)^(l1_2-l1_1)/(l1_3+1)
                        l_index += 1
                    end
                end
            end
        else
            return error("not yet implemented!")
        end
    elseif npcf.N==4
        if npcf.D==2
            weight_array = zeros(2npcf.lmax+1,2npcf.lmax+1)
            l1_ind = 1
            for l1_1 in -npcf.lmax:npcf.lmax
                l2_ind = 1
                for l2_1 in -npcf.lmax:npcf.lmax
                    if abs(l1_1+l2_1)>npcf.lmax || l1_1+l2_1<0
                        weight_array[l1_ind,l2_ind] = 0
                    else
                        weight_array[l1_ind,l2_ind] = 1
                    end
                    l2_ind += 1
                end
                l1_ind += 1
            end
        elseif npcf.D==3
            weight_array = zeros((npcf.lmax+1)^2,(npcf.lmax+1)^2,(npcf.lmax+1)^2)
            l1_ind = 1
            for l1_2 in 0:npcf.lmax
                for l1_1 in -l1_2:l1_2
                    l2_ind = 1
                    for l2_2 in 0:npcf.lmax
                        for l2_1 in -l2_2:l2_2
                            l3_ind = 1
                            for l3_2 in 0:npcf.lmax
                                for l3_1 in -l3_2:l3_2
                                    weight_array[l1_ind,l2_ind,l3_ind] = clebsch_gordan([l1_1,l1_2],[l2_1,l2_2],[-l3_1,l3_2],npcf.D)*clebsch_gordan([-l3_1,l3_2],[l3_1,l3_2],[0,0],npcf.D)
                                    l3_ind += 1
                                end
                            end
                            l2_ind += 1
                        end
                    end
                    l1_ind += 1
                end
            end
        elseif npcf.D==4
            ind_tot = Int64((npcf.lmax+1)*(npcf.lmax+2)*(2npcf.lmax+3)/6)
            weight_array = zeros(ind_tot,ind_tot,ind_tot)
            l1_ind = 1
            for l1_3 in 0:npcf.lmax
                for l1_2 in 0:l1_3
                    for l1_1 in -l1_2:l1_2
                        l2_ind = 1
                        for l2_3 in 0:npcf.lmax
                            for l2_2 in 0:l2_3
                                for l2_1 in -l2_2:l2_2
                                    l3_ind = 1
                                    for l3_3 in 0:npcf.lmax
                                        for l3_2 in 0:l3_3
                                            for l3_1 in -l3_2:l3_2
                                                weight_array[l1_ind,l2_ind,l3_ind] = clebsch_gordan([l1_1,l1_2,l1_3],[l2_1,l2_2,l2_3],[-l3_1,l3_2,l3_3],npcf.D)*clebsch_gordan([-l3_1,l3_2,l3_3],[l3_1,l3_2,l3_3],[0,0,0],npcf.D)
                                                l3_ind += 1
                                            end
                                        end
                                    end
                                    l2_ind += 1
                                end
                            end
                        end
                        l1_ind += 1
                    end
                end
            end
        end
    elseif npcf.N.==5
        if npcf.D==2
            weight_array = zeros(2npcf.lmax+1,2npcf.lmax+1,2npcf.lmax+1)
            l1_ind = 1
            for l1_1 in -npcf.lmax:npcf.lmax
                l2_ind = 1
                for l2_1 in -npcf.lmax:npcf.lmax
                    l3_ind = 1
                    for l3_1 in -npcf.lmax:npcf.lmax
                        if abs(l1_1+l2_1+l3_1)>npcf.lmax || l1_1+l2_1+l3_1<0
                            weight_array[l1_ind,l2_ind,l3_ind] = 0
                        else
                            weight_array[l1_ind,l2_ind,l3_ind] = 1
                        end
                        l3_ind += 1
                    end
                    l2_ind += 1
                end
                l1_ind += 1
            end
        elseif npcf.D==3
            weight_array = zeros((npcf.lmax+1)^2,(npcf.lmax+1)^2,2npcf.lmax+1,(npcf.lmax+1)^2,2npcf.lmax+1)
            l1_ind = 1
            for l1_2 in 0:npcf.lmax
                for l1_1 in -l1_2:l1_2
                    l2_ind = 1
                    for l2_2 in 0:npcf.lmax
                        for l2_1 in -l2_2:l2_2
                            l12_ind = 1
                            for l12_2 in 0:l1_2+l2_2
                                l12_1 = l1_1+l2_1
                                if abs(l12_1)>l12_2
                                    cb1 = 0
                                else
                                    cb1 = clebsch_gordan([l1_1,l1_2],[l2_1,l2_2],[l12_1,l12_2],npcf.D)
                                end
                                l3_ind = 1
                                for l3_2 in 0:npcf.lmax
                                    for l3_1 in -l3_2:l3_2
                                        l4_ind = 1
                                        for l4_2 in 0:npcf.lmax
                                            l4_1 = -l12_1-l3_1
                                            if abs(l4_1)>l4_2 || abs(l12_1)>l12_2
                                                cb2 = 0
                                            else
                                                cb2 = clebsch_gordan([l12_1,l12_2],[l3_1,l3_2],[-l4_1,l4_2],npcf.D)*clebsch_gordan([-l4_1,l4_2],[l4_1,l4_2],[0,0],npcf.D)
                                            end
                                            weight_array[l1_ind,l2_ind,l12_ind,l3_ind,l4_ind] = cb1*cb2
                                            l4_ind += 1
                                        end
                                        l3_ind += 1
                                    end
                                end
                                l12_ind += 1
                            end
                            l2_ind +=1
                        end
                    end
                    l1_ind +=1
                end
            end
        elseif npcf.D==4
            ind_tot = Int64((npcf.lmax+1)*(npcf.lmax+2)*(2npcf.lmax+3)/6) # l1, l2, l3 indices
            ind_tot2 = Int64((2npcf.lmax+1)*(2npcf.lmax+2)/2) # l12, l4 indices

            weight_array = zeros(ind_tot,ind_tot,ind_tot2,ind_tot,ind_tot2)
            l1_ind = 1
            for l1_3 in 0:npcf.lmax
                for l1_2 in 0:l1_3
                    for l1_1 in -l1_2:l1_2
                        l2_ind = 1
                        for l2_3 in 0:npcf.lmax
                            for l2_2 in 0:l2_3
                                for l2_1 in -l2_2:l2_2
                                    l12_ind = 1
                                    for l12_3 in 0:l1_3+l2_3
                                        for l12_2 in 0:l12_3
                                            l12_1 = l1_1+l2_1
                                            if abs(l12_1)>l12_2
                                                cb1 = 0
                                            else
                                                cb1 = clebsch_gordan([l1_1,l1_2,l1_3],[l2_1,l2_2,l2_3],[l12_1,l12_2,l12_3],npcf.D)
                                            end
                                            l3_ind = 1
                                            for l3_3 in 0:npcf.lmax
                                                for l3_2 in 0:l3_3
                                                    for l3_1 in -l3_2:l3_2
                                                        l4_ind = 1
                                                        for l4_3 in 0:npcf.lmax
                                                            for l4_2 in 0:l4_3
                                                                l4_1 = -l12_1-l3_1
                                                                if abs(l4_1)>l4_2 || abs(l12_1)>l12_2
                                                                    cb2 = 0
                                                                    cb3 = 0
                                                                else
                                                                    cb2 = clebsch_gordan([l12_1,l12_2,l12_3],[l3_1,l3_2,l3_3],[-l4_1,l4_2,l4_3],npcf.D)
                                                                    cb3 = clebsch_gordan([-l4_1,l4_2,l4_3],[l4_1,l4_2,l4_3],[0,0,0],npcf.D)
                                                                end
                                                                weight_array[l1_ind,l2_ind,l12_ind,l3_ind,l4_ind] = cb1*cb2*cb3
                                                                l4_ind += 1
                                                            end

                                                        end
                                                        l3_ind += 1
                                                    end
                                                end
                                            end
                                            l12_ind += 1
                                        end
                                    end
                                    l2_ind += 1
                                end
                            end
                        end
                        l1_ind += 1
                    end
                end
            end
        end
    end
    return weight_array
end

"""
    create_npcf_array(npcf)

Create the output array to store the NPCF calculation. The size of the array varies with the NPCF order and dimension.
"""
function create_npcf_array(npcf::NPCF)
    if npcf.N==2
        return zeros(Float64,npcf.nbins)
    elseif npcf.N==3
        return zeros(Float64,npcf.nbins,npcf.nbins,npcf.lmax+1)
    elseif npcf.N==4
        if npcf.D==2
            n_l = 0
            for l1 in -npcf.lmax:npcf.lmax
                for l2 in -npcf.lmax:npcf.lmax
                    if abs(l1+l2)>npcf.lmax || l1+l2<0
                        continue
                    end
                    n_l += 1
                end
            end
            return zeros(Float64,npcf.nbins,npcf.nbins,npcf.nbins,n_l)
        elseif npcf.D>=3
            n_l  = 0

            # Iterate over primary indices
            for l1 in 0:npcf.lmax
                for l2 in 0:npcf.lmax
                    for l3 in abs(l1-l2):min(l1+l2,npcf.lmax)
                        if (-1)^(l1+l2+l3)==1
                            # only even-parity NPCF contributions
                            n_l += 1
                        end
                    end
                end
            end
            return zeros(Float64,npcf.nbins,npcf.nbins,npcf.nbins,n_l)
        end
    elseif npcf.N==5
        if npcf.D==2
            n_l = 0
            for l1 in -npcf.lmax:npcf.lmax
                for l2 in -npcf.lmax:npcf.lmax
                    for l3 in -npcf.lmax:npcf.lmax
                        if abs(l1+l2+l3)>npcf.lmax || l1+l2+l3<0
                            continue
                        end
                        n_l += 1
                    end
                end
            end
            return zeros(Float64,npcf.nbins,npcf.nbins,npcf.nbins,npcf.nbins,n_l)
        elseif npcf.D>=3
            n_l  = 0

            # Iterate over primary indices
            for l1 in 0:npcf.lmax
                for l2 in 0:npcf.lmax
                    for l12 in abs(l1-l2):l1+l2
                        for l3 in 0:npcf.lmax
                            for l4 in abs(l12-l3):min(l12+l3,npcf.lmax)
                                if (-1)^(l1+l2+l3+l4)==1
                                    # only even-parity NPCF contributions
                                    n_l += 1
                                end
                            end
                        end
                    end
                end
            end
            return zeros(Float64,npcf.nbins,npcf.nbins,npcf.nbins,npcf.nbins,n_l)
        end
    else
        return error("not yet implemented!")
    end
end

"""
    accumulate_npcf_contribution(bins, Lambda, al, coupling_weights, npcf)

Accumulate the NPCF contribution for a single set of bins (`bins`) and primary ells (`Lambda`), given `al` and `coupling_weights` arrays.
This performs an outer product of (`N-1`) a_l arrays and a sum over intermediate ell (if `D>2`).
The bulk of the computation time is spent in this function for `N>3`.
"""
@inline function accumulate_npcf_contribution(bins,Lambda,al,coupling_weights,npcf::NPCF)
    contribution = 0

    if npcf.N==3
        b1,b2 = bins
        l1 = Lambda
        if npcf.D==2
            al1 = al[b1,npcf.lmax+l1+1]
            if al1==0; return 0.; end
            al2 = al[b2,npcf.lmax-l1+1]
            if al2==0; return 0.; end
            # taking real part and restricting to l1 >= 0
            contribution += coupling_weights[l1+1]*real(al1*al2)
        elseif npcf.D==3
            for l1_1 in -l1:l1
                ind1 = l1*(l1+1)+l1_1+1
                al1 = al[b1,ind1]
                if al1==0; continue; end
                ind2 = l1*(l1+1)-l1_1+1
                al2 = al[b2,ind2]
                if al2==0; continue; end
                contribution += coupling_weights[ind1]*real(al1*al2)
            end
        elseif npcf.D==4
            for l1_2 in 0:l1
                for l1_1 in -l1_2:l1_2
                    ind1 = Int64(1/6*l1*(1+l1)*(1+2l1)+l1_2*(l1_2+1)+l1_1+1)
                    al1 = al[b1,ind1]
                    if al1==0; continue; end
                    ind2 = Int64(1/6*l1*(1+l1)*(1+2l1)+l1_2*(l1_2+1)-l1_1+1)
                    al2 = al[b2,ind2]
                    if al2==0; continue; end
                    contribution += coupling_weights[ind1]*real(al1*al2)
                end
            end
        else
            return error("not yet implemented!")
        end

    elseif npcf.N==4
        b1,b2,b3 = bins
        if npcf.D==2
            l1_1, l2_1 = Lambda

            al1 = al[b1,l1_1+npcf.lmax+1]
            if al1==0; return 0.; end

            # Compute coupling matrix
            coupling = coupling_weights[l1_1+npcf.lmax+1,l2_1+npcf.lmax+1]
            if coupling==0; return 0.; end

            al2 = al[b2,l2_1+npcf.lmax+1]
            if al2==0; return 0.; end
            al3 = al[b3,-l1_1-l2_1+npcf.lmax+1]
            if al3==0; return 0.; end

            # Assemble contribution, taking real part and restricting to l1_1+l2_1>=0 implicitly
            contribution = real(al1*al2*al3)*coupling

        elseif npcf.D==3
            l1,l2,l3 = Lambda

            # Iterate over secondary ells (l3_1 is known)
            for l1_1 in -l1:l1
                al1 = al[b1,l1*(l1+1)+l1_1+1]
                if al1==0; continue; end
                for l2_1 in -l2:l2
                    al2 = al[b2,l2*(l2+1)+l2_1+1]
                    if al2==0; continue; end

                    l3_1 = -l1_1-l2_1
                    if abs(l3_1)>l3; continue; end
                    al3 = al[b3,l3*(l3+1)+l3_1+1]
                    if al3==0; continue; end

                    # Compute coupling matrix
                    coupling = coupling_weights[l1*(l1+1)+l1_1+1,l2*(l2+1)+l2_1+1,l3*(l3+1)+l3_1+1]
                    if coupling==0; continue; end

                    # Assemble contribution
                    contribution += real(al1*al2*al3)*coupling
                end
            end

        elseif npcf.D==4
            l1,l2,l3 = Lambda

            # Iterate over secondary ells (l3_1 is known)
            for l1_2 in 0:l1
                for l1_1 in -l1_2:l1_2
                    ind1 = Int64(1/6*l1*(1+l1)*(1+2l1)+l1_2*(l1_2+1)+l1_1+1)
                    al1 = al[b1,ind1]
                    if al1==0; continue; end

                    for l2_2 in 0:l2
                        for l2_1 in -l2_2:l2_2
                            ind2 = Int64(1/6*l2*(1+l2)*(1+2l2)+l2_2*(l2_2+1)+l2_1+1)
                            al2 = al[b2,ind2]
                            if al2==0; continue; end

                            for l3_2 in 0:l3
                                l3_1 = -l1_1-l2_1
                                if abs(l3_1)>l3_2; continue; end

                                ind3 = Int64(1/6*l3*(1+l3)*(1+2l3)+l3_2*(l3_2+1)+l3_1+1)

                                # Compute coupling matrix
                                coupling = coupling_weights[ind1,ind2,ind3]
                                if coupling==0; continue; end

                                al3 = al[b3,ind3]
                                if al3==0; continue; end

                                # Assemble contribution
                                contribution += real(al1*al2*al3)*coupling
                            end
                        end
                    end
                end
            end
        else
            return error("not yet implemented!")
        end

    elseif npcf.N==5
        b1,b2,b3,b4 = bins
        if npcf.D==2
            l1_1,l2_1,l3_1 = Lambda

            al1 = al[b1,l1_1+npcf.lmax+1]
            if al1==0; return 0.; end
            al2 = al[b2,l2_1+npcf.lmax+1]
            if al2==0; return 0.; end

            # Compute coupling matrix
            coupling = coupling_weights[l1_1+npcf.lmax+1,l2_1+npcf.lmax+1,l3_1+npcf.lmax+1]
            if coupling==0; return 0.; end

            al3 = al[b3,l3_1+npcf.lmax+1]
            if al3==0; return 0.; end
            al4 = al[b4,-l1_1-l2_1-l3_1+npcf.lmax+1]
            if al4==0; return 0.; end

            # Assemble contribution, taking real part and restricting to l1_1+l2_1>=0 implicitly
            contribution = real(al1*al2*al3*al4)*coupling

        elseif npcf.D==3
            l1,l2,l12,l3,l4=Lambda
            # Iterate over secondary ells (i.e. m) (l12_1, l4_1 are known)
            for l1_1 in -l1:l1
                al1 = al[b1,l1*(l1+1)+l1_1+1]
                if al1==0; continue; end
                for l2_1 in -l2:l2
                    al2 = al[b2,l2*(l2+1)+l2_1+1]
                    if al2==0; continue; end

                    l12_1 = l1_1+l2_1
                    if abs(l12_1)>l12; continue; end

                    for l3_1 in -l3:l3
                        al3 = al[b3,l3*(l3+1)+l3_1+1]
                        if al3==0; continue; end

                        l4_1 = -l12_1-l3_1
                        if abs(l4_1)>l4; continue; end

                        al4 = al[b4,l4*(l4+1)+l4_1+1]
                        if al4==0; continue; end

                        # Compute coupling matrix
                        coupling = coupling_weights[l1*(l1+1)+l1_1+1,l2*(l2+1)+l2_1+1,l12+1,l3*(l3+1)+l3_1+1,l4+1]
                        if coupling==0; continue; end

                        # Assemble contribution
                        contribution += real(al1*al2*al3*al4)*coupling
                    end
                end
            end

        elseif npcf.D==4
            l1,l2,l12,l3,l4=Lambda
            # Iterate over secondary ells (l12_1, l4_1 are known)
            for l1_2 in 0:l1
                for l1_1 in -l1_2:l1_2
                    ind1 = Int64(1/6*l1*(1+l1)*(1+2l1)+l1_2*(l1_2+1)+l1_1+1)
                    al1 = al[b1,ind1]
                    if al1==0; continue; end

                    for l2_2 in 0:l2
                        for l2_1 in -l2_2:l2_2
                            ind2 = Int64(1/6*l2*(1+l2)*(1+2l2)+l2_2*(l2_2+1)+l2_1+1)
                            al2 = al[b2,ind2]
                            if al2==0; continue; end

                            for l12_2 in 0:l12
                                l12_1 = l1_1 + l2_1
                                if abs(l12_1)>l12_2
                                    continue
                                end

                                for l3_2 in 0:l3
                                    for l3_1 in -l3_2:l3_2
                                        ind3 = Int64(1/6*l3*(1+l3)*(1+2l3)+l3_2*(l3_2+1)+l3_1+1)
                                        al3 = al[b3,ind3]
                                        if al3==0; continue; end

                                        for l4_2 in 0:l4
                                            l4_1 = -l12_1-l3_1
                                            if abs(l4_1)>l4_2; continue; end

                                            ind4 = Int64(1/6*l4*(1+l4)*(1+2l4)+l4_2*(l4_2+1)+l4_1+1)
                                            al4 = al[b4,ind4]
                                            if al4==0; continue; end

                                            # Compute coupling matrix
                                            coupling = coupling_weights[ind1,ind2,Int64(1/2*l12*(l12+1)+l12_2+1),ind3,Int64(1/2*l4*(l4+1)+l4_2+1)]
                                            if coupling==0; continue; end

                                            # Assemble contribution
                                            contribution += real(al1*al2*al3*al4)*coupling
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        else
            return error("not yet implemented!")
        end
    else
        return error("not yet implemented!")
    end
    return contribution
end;

"""
    normalize_npcf!(output, npcf)

Normalize the output NPCF measurements (`output`), using the binning parameters given in the `npcf` object.
This divides by the bin volume and the total volume of the space.
"""

function normalize_npcf!(output, npcf::NPCF)

    """Define volume of a single radial bin with index `b`."""
    function bin_vol(b)
        if npcf.D==2
            if npcf.coords=="spherical"
                return 2pi*npcf._dr
            else
                return pi*((npcf.r_min+npcf._dr*b)^2-(npcf.r_min+npcf._dr*(b-1))^2)
            end
        elseif npcf.D==3
            return 4pi/3*((npcf.r_min+npcf._dr*b)^3-(npcf.r_min+npcf._dr*(b-1))^3)
        elseif npcf.D==4
            return pi^2/2*((npcf.r_min+npcf._dr*b)^4-(npcf.r_min+npcf._dr*(b-1))^4)
        end
    end

    # Normalize by the bin volumes
    if npcf.N==2
        for b1 in 1:npcf.nbins
            normalization = bin_vol(b1)*npcf.volume
            output[b1]/=normalization
        end
    elseif npcf.N==3
        for b1 in 1:npcf.nbins
            for b2 in b1+1:npcf.nbins
                normalization = bin_vol(b1)bin_vol(b2)*npcf.volume
                output[b1,b2,:]/=normalization
            end
        end
    elseif npcf.N==4
        for b1 in 1:npcf.nbins
            for b2 in b1+1:npcf.nbins
                for b3 in b2+1:npcf.nbins
                    normalization = bin_vol(b1)bin_vol(b2)bin_vol(b3)*npcf.volume
                    output[b1,b2,b3,:]/=normalization
                end
            end
        end
    elseif npcf.N==5
        for b1 in 1:npcf.nbins
            for b2 in b1+1:npcf.nbins
                for b3 in b2+1:npcf.nbins
                    for b4 in b3+1:npcf.nbins
                        normalization = bin_vol(b1)bin_vol(b2)bin_vol(b3)bin_vol(b4)*npcf.volume
                        output[b1,b2,b3,b4,:]/=normalization
                    end
                end
            end
        end
    end
end;
