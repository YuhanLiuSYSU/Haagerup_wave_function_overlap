function Haagerup_F()
    A = (sqrt(13)-3)/2
    B = (sqrt(13)-2)/3
    C = (sqrt(13)+1)/6
    Dp = ((5-sqrt(13)+sqrt(6*(sqrt(13)+1))))/12
    Dm = ((5-sqrt(13)-sqrt(6*(sqrt(13)+1))))/12
    F = zeros(6,6,6)
    for j in 4:6
        F[j,j-3,j] = 1.0
        F[j-3,j,j] = 1.0
    end
    F[4,4,[1,4,5,6]] = [sqrt(A),-B,Dp,Dm]
    F[4,5,4:6] = [Dp,Dm,C]
    F[4,6,4:6] = [Dm,C,Dp]
    F[5,4,4:6] = [Dp,Dm,C]
    F[5,5,[2,4,5,6]] = [sqrt(A),Dm,-B,Dp]
    F[5,6,4:6] = [C,Dp,Dm]
    F[6,4,4:6] = [Dm,C,Dp]
    F[6,5,4:6] = [C, Dp, Dm]
    F[6,6,[3,4,5,6]] = [sqrt(A),Dp,Dm,-B]
    return F
end

function Haagerup_local_MPO()::Vector{Array{Float64,4}}
    F = Haagerup_F()
    hL = zeros(Float64, (1,6,6,6))
    hM = zeros(Float64, (6,6,6,6))
    hR = zeros(Float64, (6,6,1,6))
    for i in 1:6
        hL[1,i,i,i] = 1.0
        hR[i,i,1,i] = 1.0
    end
    for i in 1:6
        for k in 1:6
            hM[i,:,k,:] = -F[i,k,:]*F[i,k,:]'
        end
    end
    return Array{Float64,4}[hL,hM,hR]
end

function Haagerup_local_MPO_sym(T::Type = Complex{Float64})
    hL,hM,hR = Haagerup_local_MPO()
    U3 = hcat([[cis(2pi/3*i*j)/sqrt(3) for i in 0:2] for j in 0:2]...);
    U6 = cat(U3,U3,dims=(1,2)) #This transforms the basis into Z3 eigenbasis;
    hLU = ncon([hL,U6,U6',U6],[[-1,1,3,2],[-2,1],[2,-4],[3,-3]]);
    hRU = ncon([hR,U6,U6',U6'],[[3,1,-3,2],[-2,1],[2,-4],[-1,3]]);
    hMU = ncon([hM,U6',U6,U6,U6'],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]]);
    leg1 = SymLeg(3,[0],[1])
    leg2 = SymLeg(3,[0,1,2],[2,2,2])
    leg3 = conj(leg2)
    legsL = [copy(leg1),copy(leg3),copy(leg3),copy(leg2)]
    legsM = [copy(leg2),copy(leg3),copy(leg3),copy(leg2)]
    legsR = [copy(leg2),copy(leg3),copy(leg1),copy(leg2)]
    hLsym = SymTensor_from_dense(T, hLU, legsL, [[[1]],[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]]])
    hMsym = SymTensor_from_dense(T, hMU, legsM, [[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]]])
    hRsym = SymTensor_from_dense(T, hRU, legsR, [[[1,4],[2,5],[3,6]],[[1,4],[2,5],[3,6]],[[1]],[[1,4],[2,5],[3,6]]])
    return Array{T,4}[hLU,hMU,hRU], SymTensor{T,4}[hLsym, hMsym, hRsym]
end 


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

function Haagerup_OBC_MPO(T::Type = Complex{Float64})
    F = Haagerup_F()
    Ps = zeros(6,6,6) # projectors, last index labels projector
    for i in 1:6
        Ps[i,i,i] = 1.0
    end
    E = Matrix{Float64}(I,6,6) #Identity
    hM = zeros(T,6,6,14,14) # MPO tensor
    hM[:,:,1,1] = E
    hM[:,:,14,14] = E
    for i in 2:7
        hM[:,:,i,1] = Ps[:,:,i-1]
        hM[:,:,14,i+6] = Ps[:,:,i-1]
    end
    for i in 2:7
        for j in 2:7
            hM[:,:,i+6,j] = -F[i-1,j-1,:]*F[i-1,j-1,:]'
        end
    end
    hL = zeros(T, 6,6,1,14)
    hR = zeros(T, 6,6,14,1)
    
    hL[:,:,1,:] = hM[:,:,14,:]
    hR[:,:,:,1] = hM[:,:,:,1]
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hL = permutedims(hL, (3,2,4,1)) #[m1,ket,m2,bra]
    hR = permutedims(hR, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hL = U6conj(hL)
    hM = U6conj(hM)
    hR = U6conj(hR)
    
    return (hL, hM, hR)
end

function Haagerup_PBC_MPO_split(T::Type = Complex{Float64})
    hL, hM, hR = Haagerup_OBC_MPO(T)
    
    hLb = hL[:,:,8:14,:]
    hM1b = hM[8:14,:,2:13,:]
    hM2b = hM[2:13,:,1:7,:]
    hRb = hR[1:7,:,:,:]
    
    h_B = Array{T,4}[hLb, hM1b, hM2b, hRb]
    
    ((hL,hM,hR), h_B)
end

function Haagerup_PBC_MPO_sym_split(T::Type = Complex{Float64})
    ## Here we choose not to return a symTensor object, but an ordinary MPO
    hL, hM, hR = Haagerup_OBC_MPO(T)
    
    U3 = hcat([[cis(2pi/3*i*j)/sqrt(3) for i in 0:2] for j in 0:2]...);
    U6 = cat(U3,U3,dims=(1,2))    
    U14 = cat(ones(1,1),U6,U6,ones(1,1),dims=(1,2))
    
    od = Int[1,4,2,5,3,6]
    hL = ncon([hL,U14],[[-1,-2,1,-4],[1,-3]])[:,od,:,od]
    hM = ncon([U14',hM,U14],[[-1,1],[1,-2,2,-4],[2,-3]])[:,od,:,od]
    hR = ncon([hR, U14'],[[1,-2,-3,-4],[-1,1]])[:,od,:,od]
    
    # physical leg goes into ordering [1,4,2,5,3,6]
    
    hLb = hL[:,:,8:14,:]
    hM1b = hM[8:14,:,2:13,:]
    hM2b = hM[2:13,:,1:7,:]
    hRb = hR[1:7,:,:,:]
    
    h_B = Array{T,4}[hLb, hM1b, hM2b, hRb]
    
    
    ((hL,hM,hR), h_B)
end

function Haagerup_Hn_PBC_MPO_sym_split(n::Int, N::Int, T::Type = Complex{Float64})
    ## Here we choose not to return a symTensor object, but an ordinary MPO
    hL, hM, hR = Haagerup_OBC_MPO(T)
    
    U3 = hcat([[cis(2pi/3*i*j)/sqrt(3) for i in 0:2] for j in 0:2]...);
    U6 = cat(U3,U3,dims=(1,2))    
    U14 = cat(ones(1,1),U6,U6,ones(1,1),dims=(1,2))
    
    od = Int[1,4,2,5,3,6]
    hL = ncon([hL,U14],[[-1,-2,1,-4],[1,-3]])[:,od,:,od]
    hM = ncon([U14',hM,U14],[[-1,1],[1,-2,2,-4],[2,-3]])[:,od,:,od]
    hR = ncon([hR, U14'],[[1,-2,-3,-4],[-1,1]])[:,od,:,od]
    
    # physical leg goes into ordering [1,4,2,5,3,6]
    get_hM = (j::Integer)->begin
        hM_j = deepcopy(hM)
        hM_j[8:13,:,2:7,:] *= cis(n*j*2π/N)
        hM_j
    end
    
    Hn_OBC = puMPS.MPOTensor{T}[hL, (get_hM(j) for j in 2:N-1)..., hR]
    
    hLb = hL[:,:,8:14,:]
    hM1b = get_hM(N)[8:14,:,2:13,:]
    hM2b = get_hM(1)[2:13,:,1:7,:]
    hRb = hR[1:7,:,:,:]
    
    h_B = Array{T,4}[hLb, hM1b, hM2b, hRb]
    
    n,(Hn_OBC, h_B)
end

function U6conj(A::Array{T,4}) where T
    U3 = hcat([[cis(2pi/3*i*j)/sqrt(3) for i in 0:2] for j in 0:2]...);
    U6 = cat(U3,U3,dims=(1,2)) #This transforms the basis into Z3 eigenbasis;
    A = ncon([U6,A,U6'],[[-2,1],[-1,1,-3,2],[2,-4]])
    return A;
end