using Arpack, LinearMaps, Optim, KrylovKit

mutable struct sym_puMPState{T}
    A::SymTensor{T,3}
    N::Int #number of sites
end

full(M::sym_puMPState) = puMPS.puMPState(full(M.A),M.N);

Base.copy(M::sym_puMPState) = sym_puMPState(copy(M.A), M.N)

bond_dim(M::sym_puMPState) = sum(M.A.legs[1].dim_list)
phys_dim(M::sym_puMPState) = sum(M.A.legs[2].dim_list)
mps_tensor(M::sym_puMPState) = M.A
num_sites(M::sym_puMPState) = M.N

function TM_dense(M::sym_puMPState{T}) where T
    A = M.A
    Aconj = conj(M.A)
    TM = contract_index(A,Aconj,[[-1,1,-3],[-2,1,-4]])
end

function applyTM_l(TM::SymTensor{T,4},A::SymTensor{T,3}) where T
    workvec = contract_index(TM,A,[[-1,-2,1,-5],[1,-4,-3]])
    TM_new = contract_index(workvec,conj(A),[[-1,-2,-3,1,2],[2,1,-4]])
    return TM_new
end

function blockTMs(M::sym_puMPState{T}, Ns::Integer=num_sites(M)) where {T}
    A = mps_tensor(M)
    TMs = SymTensor{T,4}[TM_dense(M)]
    for n in 2:Ns
        TM_new = applyTM_l(TMs[end], A)
        push!(TMs, TM_new)
    end
    return TMs
end

function apply_blockTM_l(M::sym_puMPState{T}, TM::SymTensor{T,4}, N::Integer) where {T}
    A = mps_tensor(M)
    TMcp = copy(TM) #Never overwrite TM!
    for i in 1:N
        TMcp = applyTM_l(TMcp, A) #D^5 d
    end
    return TMcp
end

function blockTM_dense(M::sym_puMPState{T}, N::Integer) where {T}
    #TODO: Depending on d vs. D, block the MPS tensors first to form an initial blockTM at cost D^4 d^blocksize.
    D = bond_dim(M)
    TM = apply_blockTM_l(M, TM_dense(M), N-1)
    TM
end

function LinearAlgebra.tr(Ten::SymTensor{T,2}) where T
    return contract_index(Ten, [1], [2])
end

function LinearAlgebra.tr(Ten::SymTensor{T,4}) where T
    return contract_index(Ten, [1,2], [3,4])
end

function LinearAlgebra.tr(Ten::SymTensor{T,6}) where T
    return contract_index(Ten, [1,2,3], [4,5,6])
end

function LinearAlgebra.rmul!(A::SymTensor, x)
    for i in 1:length(A.Tensors)
        rmul!(A.Tensors[i], x)
    end
end

function LinearAlgebra.norm(M::sym_puMPState{T}; TM_N=blockTM_dense(M, num_sites(M))) where {T}
    sqrt(real(tr(TM_N)))
end

function LinearAlgebra.normalize!(M::sym_puMPState{T}; TM_N=blockTM_dense(M, num_sites(M))) where {T}
    for i in 1:length(mps_tensor(M).Tensors)
        rmul!(mps_tensor(M).Tensors[i], T(1.0 / norm(M, TM_N=TM_N)^(1.0/num_sites(M))) )
    end
    M
end

function LinearAlgebra.normalize!(M::sym_puMPState{T}, blkTMs::Vector{SymTensor{T,4}}) where {T}
    N = num_sites(M)
    normM = norm(M, TM_N=blkTMs[N])
    
    rmul!(mps_tensor(M), T(1.0 / normM^(1.0/N)) )
    for n in 1:N
        rmul!(blkTMs[n], T(1.0 / normM^(2.0/n)) ) 
    end
    M, blkTMs
end

function TM_dense_MPO(A::SymTensor{T,3},B::SymTensor{T,3},O::SymTensor{T,4}) where T
    AO = contract_index(A,O,[[-1,1,-3],[-2,1,-4,-5]])
    AOB = contract_index(AO,conj(B),[[-1,-2,-4,-5,1],[-3,1,-6]])
    return AOB
end

function TM_dense_MPO(M::sym_puMPState{T}, O::Vector{SymTensor{T,4}}) where {T}
    A = mps_tensor(M)
    applyTM_MPO_l(M, O[2:end], TM_dense_MPO(A, A, O[1]))
end   

function applyTM_MPO_l(M::sym_puMPState{T}, O::Vector{SymTensor{T,4}}, TM2::SymTensor{T,6}) where {T}
    A = mps_tensor(M)
    
    TM = TM2
    if length(O) > 0
        for n in 1:length(O)
            TMA = contract_index(TM, A, [[-1,-2,-3,1,-6,-7],[1,-5,-4]])
            TMAO = contract_index(TMA, O[n],[[-1,-2,-3,-4,1,2,-7],[2,1,-5,-6]])
            TM = contract_index(TMAO,conj(A), [[-1,-2,-3,-4,-5,1,2],[2,1,-6]])
        end
    end
    return TM
end

function applyTM_MPO_l(M::sym_puMPState{T},O::SymTensor{T,4}, TM2::SymTensor{T,4}) where {T}
    ## one - site application of O to TM2
    A = mps_tensor(M)
    workvec = contract_index(TM2,A,[[-1,-2,1,-5],[1,-4,-3]])
    workO = contract_index(workvec, O, [[-1,-3,-4,1,-7],[-2,1,-5,-6]])
    TM = contract_index(workO,conj(A), [[-1,-2,-3,-4,-5,1,2],[2,1,-6]])
    return TM
end

function applyTM_MPO_l(M::sym_puMPState{T},O::Vector{SymTensor{T,4}}, TM2::SymTensor{T,4}) where {T}
    A = mps_tensor(M)
    if(length(O)>0)
        output_tensor = applyTM_MPO_l(M, O[2:end], applyTM_MPO_l(M,O[1],TM2))
    else
        output_tensor = TM2
    end
    return output_tensor
end   

function expect(M::sym_puMPState{T}, op::Vector{SymTensor{T,4}}; MPS_is_normalized::Bool=true, blkTMs=[]) where {T}
    N = num_sites(M)
    A = mps_tensor(M)
    D = bond_dim(M)
    
    Nop = length(op)
    
    if N == Nop
        TMop = TM_dense_MPO(M, op)
        res = tr(TMop)

        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? tr(blkTMs[N]) : norm(M)^2
            res /= normsq
        end
    else
        TM = length(blkTMs) >= N-Nop ? blkTMs[N-Nop] : blockTMs(M, N-Nop)[end]
        
        TMop = applyTM_MPO_l(M, op, TM)
        
        res = tr(TMop)
        
        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? tr(blkTMs[N]) : tr(apply_blockTM_l(M, TM, Nop))
            res /= normsq
        end
    end
    
    return res
end

function Haar_random_Tensor(T::Type, dims::Vector{Int})
    ten = randn(T, dims...)
    if(length(dims)==1)
        normalize!(ten)
    else
        ten_mat = reshape(ten, (dims[1],prod(dims[2:end])))
        u,s,v = svd(ten_mat)
        if dims[1]>prod(dims[2:end])
            ten = reshape(u, Tuple(dims))
        else
            ten = reshape(v', Tuple(dims))
        end
    end
    return ten
end

function Haar_random_SymTensor(T::Type, legs::Vector{SymLeg}, inds::Vector{Vector{Int}})
    Tensors = Array{T,length(legs)}[]
    for i in 1:length(inds)
        ind = inds[i]
        dims = [legs[k].dim_list[ind[k]] for k in 1:length(ind)]
        push!(Tensors, Haar_random_Tensor(T, dims))
    end
    return SymTensor(legs, inds, Tensors)
end

function random_sym_puMPS(T::Type, Nc::Int, phys_dims::Vector{Int}, bond_dims::Vector{Int}, Ns::Int)
    ## Nc charge finite positive number, the puMPS tensor has total charge 0
    bondL = SymLeg(Nc, collect(0:Nc-1), bond_dims);
    bondR = conj(copy(bondL))
    physL = SymLeg(Nc, collect(0:Nc-1), phys_dims);
    inds = Vector{Int}[]
    for i in 1:Nc
        for j in 1:Nc
            for k in 1:Nc
                if((bondL.charge_list[i]+physL.charge_list[j]+bondR.charge_list[k])%Nc==0)
                    push!(inds,[i,j,k])
                end
            end
        end
    end
    A = Haar_random_SymTensor(T, SymLeg[bondL, physL, bondR], inds)
    return sym_puMPState(A, Ns)
end

function axpy!(a,x::SymTensor,y::SymTensor)
    @assert isequal(x.legs,y.legs)
    for (list_ind,charge_ind) in enumerate(x.existing_charge_inds)
        y_list_ind = findfirst(y.existing_charge_inds.==[charge_ind])
        if(typeof(y_list_ind)!=Nothing)
            BLAS.axpy!(a,x.Tensors[list_ind],y.Tensors[y_list_ind])
        else
            push!(y.existing_charge_inds, charge_ind)
            push!(y.Tensors, -x.Tensors[list_ind])
        end
    end
end

function applyTM_MPO_r(M::sym_puMPState{T}, O::Vector{SymTensor{T,4}}, TM2::SymTensor{T,6}) where {T}
    A = mps_tensor(M)
    
    TM = TM2
    if length(O) > 0
        for n in length(O):-1:1
            TMA = contract_index(A, TM, [[-1,-2,1],[1,-3,-4,-5,-6,-7]])
            TMAO = contract_index(O[n],TMA, [[-2,1,2,-3],[-1,1,2,-4,-5,-6,-7]])
            TM = contract_index(conj(A),TMAO, [[-3,1,2],[-1,-2,1,2,-4,-5,-6]])
        end
    end
    TM
end

function applyTM_MPO_r(M::sym_puMPState{T}, O::Vector{SymTensor{T,4}}, TM2::SymTensor{T,4}) where {T}
    if(length(O)>0)
        output_tensor = applyTM_MPO_r(M, O[1:end-1], applyTM_MPO_r(M,O[end],TM2))
    else
        output_tensor = TM2
    end
    return output_tensor
end

function applyTM_MPO_r(M::sym_puMPState{T}, O::SymTensor{T,4}, TM2::SymTensor{T,4}) where {T}
    A = mps_tensor(M)
    workvec = contract_index(A,TM2,[[-1,-2,1],[1,-3,-4,-5]])
    workO = contract_index(workvec,O, [[-1,1,-4,-5,-7],[-2,1,-6,-3]])
    TM_new = contract_index(workO, conj(A),[[-1,-2,1,2,-4,-5,-6],[-3,1,2]])
    return TM_new
end

function derivatives_1s(M::sym_puMPState{T}, h::Vector{SymTensor{T,4}}; blkTMs=blockTMs(M, num_sites(M)-1), e0::Real=0.0) where {T}
    A = mps_tensor(M)
    N = num_sites(M)
    D = bond_dim(M)
    
    e0 = real(T)(e0) #will be used for scaling, so need it in the working precision

    j = 1
    TM = blkTMs[j]
    
    #Transfer matrix with one H term
    TM_H = contract_index(TM_dense_MPO(M, h),[2],[5])
    
    #Subtract energy density e0 * I.
    #Note: We do this for each h term individually in order to avoid a larger subtraction later.
    #Assumption: This is similarly accurate to subtracting I*e0 from the Hamiltonian itself.
    #The transfer matrices typically have similar norm before and after subtraction, even when
    #the final gradient has small physical norm.
    axpy!(-e0, blkTMs[length(h)], TM_H) 
    
    #TM_H_res = similar(TM_H)    
    
    #work = workvec_applyTM_l(A, A)
    
    #TMMPO_res = res_applyTM_MPO_l(M, h, TM)
    #workMPO = workvec_applyTM_MPO_l(M, h, vcat(MPS_MPO_TM{T}[TM], TMMPO_res[1:end-1]))
    
    for k in length(h)+1:N-1 #leave out one site (where we take the derivative)
        #Extend TM_H
        TM_H = applyTM_l(TM_H, A)
        
        #New H term
        TM_H_add = contract_index(applyTM_MPO_l(M, h, TM),[2],[5])
        axpy!(-e0, blkTMs[j+length(h)], TM_H_add) #Subtract energy density e0 * I
        
        j += 1
        TM = blkTMs[j]
        
        axpy!(real(T)(1.0), TM_H_add, TM_H) #add new H term to TM_H
    end
    
    #effective ham terms that do not act on gradient site
    axpy!(-length(h)*e0, blkTMs[N-1], TM_H) #Subtract energy density for the final terms

    #Add only the A, leaving a conjugate gap.
    #TM_H_r = TM_convert(TM_H)
    #@tensor d_A[l, s, r] := A[k1, s, k2] * TM_H_r[k2,r, k1,l]
    d_A = contract_index(TM_H, A, [[1,-3,2,-1],[2,-2,1]])
    
    #NOTE: TM now has N-length(h) sites
    for n in 1:length(h)
        TM_H = applyTM_MPO_l(M, h[1:n-1], TM)
        TM_H = applyTM_MPO_r(M, h[n+1:end], TM_H)
        hn = h[n]
        #@tensor d_A[l, t, r] += (A[k1, s, k2] * TM_H[k2,m2,r, k1,m1,l]) * hn[m1,s,m2,t] #allocates temporaries
        temp = contract_index(TM_H, A, [[1,-1,-2,2,-3,-4],[2,-5,1]])
        gr = contract_index(temp, hn, [[3,-3,1,-1,2],[1,2,3,-2]])
        axpy!(real(T)(1.0), gr, d_A)
    end
    
    return d_A
end

function gradient_central(M::sym_puMPState{T}, inv_lambda::SymTensor{T,2}, d_A::SymTensor{T,3}; 
        sparse_inverse::Bool=false, pinv_tol::Real=1e-12,blkTMs::Vector{SymTensor{T,4}} = blockTMs(M, num_sites(M)-1)) where {T}
    # Use dense inverese rather than sparse
    # Note: inv_lambda must be diagonal!
    N = num_sites(M)
    Alegs = M.A.legs
    #D = bond_dim(M)
    #d = phys_dim(M)
        
    T1 = length(blkTMs) >= N-1 ? blkTMs[N-1] : blockTM_dense(M, N-1)
    
    #Overlap matrix in central gauge (except for the identity on the physical dimension)
    #@tensor Nc[b2,b1,t2,t1] := inv_lambda[t1,ti] * inv_lambda[b1,bi] * T1[ti,bi,t2,b2]
    #Nc = reshape(Nc, (D^2, D^2))
    T1top = contract_index(inv_lambda, T1, [[1,-4],[1,-2,-3,-1]])
    Nc = contract_index(inv_lambda, T1top, [[-2,1],[-1,1,-3,-4]])
    ## Note that above can also be obtained from the normalization process
    
    #@tensor d_Ac[v1,v2,s] := d_A[v1,s,vi] * inv_lambda[vi,v2] # now size (D,D,d)
    d_Ac = contract_index(d_A, inv_lambda,[[-1,-2,1],[-3,1]]);
    
    #grad_Ac = zero(d_Ac)
    #kill the option of sparse inverse 
    #if sparse_inverse
    #grad_Ac_init = tensorcopy(grad_Ac_init, [:a,:b,:c], [:a,:c,:b]) # now size (D,D,d)
        #Split the inverse problem along the physical dimension, since N acts trivially on that factor. Avoids constructing N x I.
        #for s in 1:d
            #grad_vec = BiCGstab(Nc, vec(view(d_Ac, :,:,s)), vec(view(grad_Ac_init, :,:,s)), tol, max_itr=max_itr)
            #copyto!(view(grad_Ac, :,:,s), grad_vec)
        #end
    #else
        #Dense version  
        #Nc_i = pinv(Nc, pinv_tol)
        #for s in 1:d
            #grad_vec = Nc_i * vec(view(d_Ac, :,:,s))
            #copyto!(view(grad_Ac, :,:,s), grad_vec)
        #end
    #end
    ## dense inverse - symmeteric case
    Nc_mat = merge_index(Nc, [1,3])
    d_Ac_mat = merge_index(permutedims(d_Ac,[1,3,2]),1)
    
    #@tensor grad_A[v1,s,v2] := grad_Ac[v1,vi,s] * inv_lambda[vi,v2] # back to (D,d,D)
    grad_A,norm_grad_A = pinvNd(Nc_mat, d_Ac_mat, pinv_tol=pinv_tol);
    #norm_grad_A = sqrt(abs(dot(vec(grad_A), vec(d_A))))
    grad_A = permutedims(split_index(grad_A,1,Alegs[1],Alegs[3]),[1,3,2])
    #grad_A, norm_grad_A, tensorcopy(grad_Ac, [:a,:b,:c], [:a,:c,:b])
    return grad_A, norm_grad_A
    #return Nc_mat, d_Ac_mat
end

function eye(T::Type,d::Int)
    diagm(0=>ones(T,d))
end

function eye(T::Type,leg::SymLeg)
    legs = [leg, conj(leg)]
    inds = Vector{Int}[Int[i,i] for i in 1:length(leg.charge_list)]
    Ts = Matrix{T}[eye(T,leg.dim_list[i]) for i in 1:length(leg.charge_list)]
    return SymTensor(legs,inds,Ts)
end

function pinvNd(N::SymTensor{T,2},dA::SymTensor{T,2};pinv_tol::Real=1E-12) where T
    ## return N^(-1)*dA and norm of gradient (dA'*N^{-1}*dA)
    Nlegs = N.legs
    @assert isequal(Nlegs[1],dA.legs[1])
    output_legs = [conj(Nlegs[2]),dA.legs[2]]
    output_inds = Vector{Int}[]
    output_tensors = Matrix{T}[]
    normgrad = 0.0
    for (i,charge_ind) in enumerate(N.existing_charge_inds)
        Nsub = N.Tensors[i]
        charge_ind1 = charge_ind[1]
        charge_ind2 = charge_ind[2]
        for (j,charge_ind_dA) in enumerate(dA.existing_charge_inds)
            charge_ind_dA1 = charge_ind_dA[1]
            charge_ind_dA2 = charge_ind_dA[2]
            if(charge_ind_dA1 == charge_ind1)
                dAsub = dA.Tensors[j]
                push!(output_inds, [charge_ind2,charge_ind_dA2])
                grad = pinv(Nsub,pinv_tol)*dAsub
                push!(output_tensors, grad)
                normgrad += real(dot(dAsub,grad))
            end
        end
    end
    return SymTensor(output_legs, output_inds, output_tensors), sqrt(normgrad)           
end

struct EnergyHighException{T<:Real} <: Exception
    stp::T
    En::T
end
struct WolfeAbortException{T<:Real} <: Exception 
    stp::T
    En::T
end

function line_search_energy(M::sym_puMPState{T}, En0::Real, grad::SymTensor{T,3}, grad_normsq::Real, step::Real, hMPO::Vector{SymTensor{T,4}}; 
        itr::Integer=10, rel_tol::Real=1e-1, max_attempts::Integer=3, wolfe_c1::Real=100.0
    ) where {T}
    
    num_calls::Int = 0
    attempt::Int = 0
    
    f = (stp::Float64)->begin
        num_calls += 1
        
        A_cp = copy(M.A)
        axpy!(-real(T)(stp), grad, A_cp)  
        
        M_new = sym_puMPState{T}(A_cp,num_sites(M))
        #set_mps_tensor!(M_new, mps_tensor(M) .- real(T)(stp) .* grad)
        
        En = real(expect(M_new, hMPO, MPS_is_normalized=false)) #computes the norm and energy-density in one step
        
        #println("Linesearch: $stp, $En")

        #Abort the search if the first step already increases the energy compared to the initial state
        num_calls == 1 && En > En0 && throw(EnergyHighException{Float64}(stp, Float64(En)))
        
        #Note: This is the first Wolfe condition, plus a minimum step size, since we don't want to compute the gradient...
        #Probably it effectively only serves to reduce the maximum step size reached, thus we turn it off by setting wolfe_c1=100.
        stp > 1e-2 && En <= En0 - wolfe_c1 * stp * grad_normsq && throw(WolfeAbortException{Float64}(stp, En))
        
        En
    end
    
    step = Float64(step)

    res = nothing
    while attempt <= max_attempts
        try
            attempt += 1
            ores = optimize(f, step/5, step*1.8, Brent(), iterations=itr, rel_tol=Float64(rel_tol), store_trace=false, extended_trace=false)
            res = Optim.minimizer(ores), Optim.minimum(ores)
            break
        catch e
            if isa(e, EnergyHighException)
                if attempt < max_attempts
                    @warn "Linesearch: Initial step was too large. Adjusting!"
                    step *= 0.1
                    num_calls = 0
                else
                    @warn "Linesearch: Initial step was too large. Aborting!"
                    res = e.stp, e.En
                    break
                end
            elseif isa(e, WolfeAbortException)
                @info "Linesearch: Early stop due to good enough step!"
                res = e.stp, e.En
                break
            else
                rethrow(e)
            end
        end
    end
    
    res
end

function minimize_energy_local!(M::sym_puMPState{T}, hMPO::Vector{SymTensor{T,4}}, maxitr::Integer;
        tol::Real=1e-6,
        step::Real=0.001,
        maxstep::Real=1.0) where {T}
    
    blkTMs = blockTMs(M)
    normalize!(M, blkTMs)
    En = real(expect(M, hMPO, blkTMs=blkTMs))
    
    #grad_Ac = rand_MPSTensor(T, phys_dim(M), bond_dim(M)) #Used to initialise the BiCG solver
    #stol = 1e-12
    norm_grad = Inf
    
    for k in 1:maxitr
        M, lambda, lambda_i = canonicalize_left!(M)
        
        blkTMs = blockTMs(M)
        deriv = derivatives_1s(M, hMPO, blkTMs=blkTMs, e0=En)

        #if use_phys_grad
        grad, norm_grad = gradient_central(M, lambda_i, deriv, blkTMs=blkTMs)
        #else
            #grad = deriv
            #bTM_Nm1 = TM_convert(blkTMs[num_sites(M)-1])
            #@tensor ng2[] := deriv[vt1, p, vt2] * (conj(deriv[vb1, p, vb2]) * bTM_Nm1[vt2,vb2, vt1,vb1])
            #norm_grad = sqrt(real(scalar(ng2)))
        #end
        
        if norm_grad < tol
            break
        end

        #stol = min(1e-6, max(norm_grad^2/10, 1e-12))
        En_prev = En

        step_corr = min(max(step, 0.001),maxstep)
        #step_corr = max(step, 0.001)
        step, En = line_search_energy(M, En, grad, norm_grad^2, step_corr, hMPO)
        
        println("$k, $norm_grad, $step, $En, $(En-En_prev)")

        #Anew = mps_tensor(M) .- real(T)(step) .* grad
        #set_mps_tensor!(M, Anew)
        axpy!(-real(T)(step),grad,M.A)
        #M.A = Anew # change the MPSTensor
        normalize!(M)
    end
    
    normalize!(M)
    M, norm_grad
end

function applyTM_l(A::SymTensor{T,3},x::SymTensor{T,2}) where T
    xA = contract_index(x,A,[[-1,1],[1,-2,-3]])
    TMx = contract_index(xA, conj(A),[[1,2,-2],[1,2,-1]])
    return TMx
end

function applyTM_r(A::SymTensor{T,3},x::SymTensor{T,2}) where T
    xA = contract_index(x,A,[[1,-3],[-1,-2,1]])
    TMx = contract_index(xA, conj(A),[[-1,1,2],[-2,1,2]])
    return TMx
end

function TM_dominant_eigs(A::SymTensor{T,3},dir="l") where T
    legs = SymLeg[A.legs[1],conj(A.legs[1])]
    dims = legs[1].dim_list
    dims2 = dims.^2
    charge_inds = Vector{Int}[Int[i,i] for i in 1:length(dims)]
    applyTM = (v::AbstractVector) -> begin
        ## dim(v) = sum(dims2)
        vs = Matrix{T}[reshape(v[1+sum(dims2[1:i-1]):sum(dims2[1:i])],(dims[i],dims[i])) for i in 1:length(dims)] 
        xtensor = SymTensor{T,2}(legs, charge_inds, vs)
        TMx = (dir == "l" ? applyTM_l(A,xtensor) : applyTM_r(A,xtensor))
        ord = sortperm(TMx.existing_charge_inds)
        vout = vcat(vec.(TMx.Tensors[ord])...)
        vout
    end
    fmap = LinearMap{T}(applyTM, sum(dims2))
    evs, eVs = eigs(fmap, nev=6)
    ev = evs[1]
    v = eVs[:,1]
    eV = Matrix{T}[reshape(v[1+sum(dims2[1:i-1]):sum(dims2[1:i])],(dims[i],dims[i])) for i in 1:length(dims)] 
    return ev,eV
end

function canonicalize_left(l::Matrix{T}, r::Matrix{T};tol=1E-8) where T
    tol = sqrt(eps(real(T)))
    l = l/tr(l)
    r = r/tr(r)
    lh = 0.5(l + l')
    rh = 0.5(r + r')
    lh = T <: Real ? real(lh) : lh
    rh = T <: Real ? real(rh) : rh
    evl, Ul = eigen(Hermitian(l))
    norm(Ul * Ul' - I) > tol && warn("Nonunintary eigenvectors.")

    sevl = Diagonal(sqrt.(complex.(evl)))
    g = sevl * Ul'
    gi = Ul * inv(sevl)
    
    r = g * Hermitian(r) * g'
    r = Hermitian(0.5(r + r'))
    
    evr, Ur = eigen(r)
    norm(Ur * Ur' - I) > tol && warn("Nonunintary eigenvectors.")
    
    gi = gi * Ur
    g = Ur' * g
    
    #It would be nice to use special matrix types here, but currently our tm functions can't handle them
    rnew = Diagonal(convert(Vector{T}, evr))
    lnew = I
    
    lnew, rnew, g, gi
end

function gauge_transform(A::SymTensor{T,3},g::SymTensor{T,2},gi::SymTensor{T,2}) where T
    gA = contract_index(g,A,[[-1,1],[1,-2,-3]])
    gAgi = contract_index(gA,gi,[[-1,-2,1],[1,-3]])
    return gAgi
end

function canonicalize_left!(M::sym_puMPState{T};pinv_tol=1E-12) where T
    evL, ls = TM_dominant_eigs(M.A,"l")
    evR, rs = TM_dominant_eigs(M.A,"r")
    if(abs(evL-evR)>1E-7)
        @warn ("Eigenvalues of transfer matrix do not match!");
    end
    legs = SymLeg[M.A.legs[1],M.A.legs[3]]
    charge_inds = Vector{Int}[Int[i,i] for i in 1:length(legs[1].charge_list)]
    x_tens = Matrix{T}[]
    xi_tens = Matrix{T}[]
    lambda_tens = Matrix{T}[]
    lambdai_tens = Matrix{T}[]
    for ind in 1:length(ls)
        l = ls[ind]
        r = rs[ind]        
        lnew, rnew, x, xi = canonicalize_left(l, r)
        lambda = Diagonal(sqrt.(real.(diag(rnew))))
        lambda_i = pinv(lambda, pinv_tol)
        push!(x_tens, x)
        push!(xi_tens, xi)
        push!(lambda_tens, lambda)
        push!(lambdai_tens, lambda_i)
    end
    x = SymTensor(legs, charge_inds, x_tens)
    xi = SymTensor(legs, charge_inds, xi_tens)
    lambda = SymTensor(legs, charge_inds, lambda_tens)
    lambdai = SymTensor(conj.(legs), charge_inds, lambdai_tens)
    
    AL = gauge_transform(M.A, x, xi)
    M.A = AL
    
    return M, lambda, lambdai
end

function extend_T!(t::Array{T,3},D::Int,N::Int;rs::Float64 = 1E-3) where T
    D0 = size(t,1)
    A = rs.*randn(T,D,size(t,2),D)
    A[1:D0,:,1:D0] = t
    return A
end

function extend_sym_puMPS(M::sym_puMPState{T}, D, N) where T
    A = mps_tensor(M)
    Bleg1 = SymLeg(A.legs[1].N, [0,1,2],[div(D,3),div(D,3),div(D,3)])
    Bleg2 = copy(A.legs[2])
    Bleg3 = conj(Bleg1);
    BTensors = Array{T,3}[extend_T!(t,div(D,3),N) for t in A.Tensors]
    B = SymTensor([Bleg1,Bleg2,Bleg3], A.existing_charge_inds, BTensors);
    return sym_puMPState(B,N)
end

##### ----------     Below are dense contractions with symmetric tensors....

function SymTensor_from_dense(T::Type, A::Array, legs::Vector{SymLeg}, subinds::Vector{Vector{Vector{Int}}})
    ## construct SymTensor from dense (only select those with charge zero)
    r = length(legs)
    Nc = legs[1].N
    auxTensor = zeros([length(leg.charge_list) for leg in legs]...)
    allinds = eachindex(view(auxTensor, [1:size(auxTensor,k) for k in 1:r]...))
    existing_charge_inds = Vector{Int}[]
    Tensors = Array{T,r}[]
    for ind in allinds
        charges = [legs[i].charge_list[ind[i]] for i in 1:r]
        if(sum(charges)%Nc == 0) #Charge conservation
            ind_array = collect(Tuple(ind))
            original_inds = [subinds[k][ind[k]] for k in 1:r]
            push!(existing_charge_inds, ind_array)
            push!(Tensors, A[original_inds...])
        end
    end
    return SymTensor{T,r}(legs, existing_charge_inds, Tensors)
end

function sym_inds_to_dense(D::Int, d::Int = 6, tot_c::Int = 0)
    ## D bond dim with multiples of 3, charge inds = [0,1,2]
    ## d phys dim with multiples of 3, charge inds = [0,1,2]
    findcharge(j,dt) = div(j-1,div(dt,3))
    temp = zeros(D,d,D);
    tempinds = CartesianIndices(temp)
    charge0inds = Int[]
    for (i,ind) in enumerate(tempinds)
        ind1,ind2,ind3 = Tuple(ind)
        charge1 = findcharge(ind1, D)
        charge2 = findcharge(ind2, d)
        charge3 = findcharge(ind3, D)
        if((charge1+charge2+3-charge3)%3==tot_c)
            push!(charge0inds, i)
        end
    end
    return charge0inds
end

function sym_puMPS_from_dense(M::puMPS.puMPState{T},legs::Vector{SymLeg}) where T
    A = puMPS.mps_tensor(M)
    D = puMPS.bond_dim(M)
    d = puMPS.phys_dim(M)
    charge_inds(x::Int) = Vector{Int}[collect(1:div(x,3)),collect(div(x,3)+1:2*div(x,3)),collect(2*div(x,3)+1:x)]
    binds = charge_inds(D)
    pinds = charge_inds(d)
    inds = Vector{Vector{Int}}[binds, pinds, binds];
    A_sym = SymTensor_from_dense(T,A,legs,inds);
    return sym_puMPState{T}(A_sym,puMPS.num_sites(M))
end

function canonicalize_left_sym!(M::puMPS.puMPState{T},legs::Vector{SymLeg};pinv_tol=1E-12) where T
    M_sym = sym_puMPS_from_dense(M,legs)
    M_sym, lambda, lambdai = canonicalize_left!(M_sym, pinv_tol = pinv_tol)
    return full(M_sym), full(lambda), full(lambdai)
end

function minimize_energy_local_sym!(M::sym_puMPState{T}, hMPO::Vector{SymTensor{T,4}}, maxitr::Integer;
        tol::Real=1e-6,
        step::Real=0.001,
        maxstep::Real=1.0) where {T}
    ## Hybrid optimization using full contraction and symmetric tensors
    M_full = full(M);
    hMPO_full = full.(hMPO);
    blkTMs = puMPS.blockTMs(M_full)
    puMPS.normalize!(M_full, blkTMs)
    En = real(puMPS.expect(M_full, hMPO_full, blkTMs=blkTMs))
    
    #grad_Ac = rand_MPSTensor(T, phys_dim(M), bond_dim(M)) #Used to initialise the BiCG solver
    #stol = 1e-12
    norm_grad = Inf
    
    for k in 1:maxitr
        M_full, lambda, lambda_i = canonicalize_left_sym!(M_full,M.A.legs)
        
        blkTMs = puMPS.blockTMs(M_full)
        deriv = puMPS.derivatives_1s(M_full, hMPO_full, blkTMs=blkTMs, e0=En)

        #if use_phys_grad
        grad, norm_grad = puMPS.gradient_central(M_full, lambda_i, deriv, blkTMs=blkTMs)
        #else
            #grad = deriv
            #bTM_Nm1 = TM_convert(blkTMs[num_sites(M)-1])
            #@tensor ng2[] := deriv[vt1, p, vt2] * (conj(deriv[vb1, p, vb2]) * bTM_Nm1[vt2,vb2, vt1,vb1])
            #norm_grad = sqrt(real(scalar(ng2)))
        #end
        
        if norm_grad < tol
            break
        end

        #stol = min(1e-6, max(norm_grad^2/10, 1e-12))
        En_prev = En

        step_corr = min(max(step, 0.001),maxstep)
        #step_corr = max(step, 0.001)
        step, En = puMPS.line_search_energy(M_full, En, grad, norm_grad^2, step_corr, hMPO_full)
        
        println("$k, $norm_grad, $step, $En, $(En-En_prev)")

        Anew = puMPS.mps_tensor(M_full) .- real(T)(step) .* grad
        puMPS.set_mps_tensor!(M_full, Anew)
        #axpy!(-real(T)(step),grad,M.A)
        #M.A = Anew # change the MPSTensor
        puMPS.normalize!(M_full)
        flush(stdout)
    end
    
    puMPS.normalize!(M_full)
    M = sym_puMPS_from_dense(M_full,M.A.legs);
    
    M, norm_grad
end

function excitations_hybrid(M, ks, Gs, Heffs, lambda_i, num_states, tot_charge::Int = 0)
    A = puMPS.mps_tensor(M)
    D = puMPS.bond_dim(M)
    d = puMPS.phys_dim(M)
    inds = sym_inds_to_dense(D,d,tot_charge)
    ens = []
    exs = []
    for j in 1:length(ks)
        k = ks[j]
        println("k=$k")
        G = reshape(Gs[j],(D^2*d,D^2*d))
        Heff = reshape(Heffs[j],(D^2*d,D^2*d))
        Gr = G[inds,inds]
        Hr = Heff[inds,inds]
        GiH = pinv(Gr, 1E-12) * Hr
        println(size(GiH))
        ev, eV, info = eigsolve(GiH, vec(A)[inds], num_states[j], :LM)
        info.converged < num_states[j] && @warn "$(num_states[j] - info.converged) eigenvectors did not converge after $(info.numiter) iterations."
        for (ind,v) in enumerate(eV)
            Bnrm = sqrt(dot(v, Gr * v)) #We can use G to compute the norm
            rmul!(v, 1.0/Bnrm)
            Bc_mat = zeros(Complex{Float64},D,d,D)
            syminds = sym_inds_to_dense(D,d,tot_charge)
            Bc_mat[syminds] = v
            @tensor Bl[l,s,r] := Bc_mat[l,s,r0] * lambda_i[r0,r]
            eV[ind] = Bl[syminds]
        end
        push!(ens, ev)
        push!(exs, eV)
    end
    return ens, exs
end