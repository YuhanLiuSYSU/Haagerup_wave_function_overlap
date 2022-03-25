using LinearAlgebra, TensorOperations

struct SymLeg
    N::Int #N = 0 => U(1), N>0 => Z_N
    charge_list::Vector{Int} #list of charges in the leg, e.g. [0,1,2,3] 
    dim_list::Vector{Int}    #dimensiions for each of the charge, e.g. [4,7,2,8]
end

struct SymTensor{T,N}
    legs::Vector{SymLeg} #length equals rank of the tensor
    existing_charge_inds::Vector{Vector{Int}} #Each element has length = rank of tensor, encoding charge info
    Tensors::Vector{<:AbstractArray{T,N}} #Each element is the tensor with fixed charge. length = length(existing_charge_inds)
end

Base.copy(l::SymLeg) = SymLeg(deepcopy(l.N), deepcopy(l.charge_list), deepcopy(l.dim_list))
charges(leg::SymLeg) = leg.charge_list 
tot_dim(leg::SymLeg) = sum(leg.dim_list)

function Base.conj(l::SymLeg) 
    N = l.N
    new_charge_list = N.-copy(l.charge_list)
    if(N!=0)
        new_charge_list = new_charge_list.%N
    end
    return SymLeg(N, new_charge_list, copy(l.dim_list))
end 

function isopposite(a::SymLeg, b::SymLeg)
    N = a.N
    Nb = b.N
    if(N!=Nb)
        output = false
    else
        if(N==0)
            output = isequal(a.charge_list,-b.charge_list) && isequal(a.dim_list,b.dim_list)
        else
            output = isequal((N.+(a.charge_list)).%N,(N.-b.charge_list).%N) && isequal(a.dim_list,b.dim_list)
        end
    end
    return output
end

Base.isequal(a::SymLeg, b::SymLeg) = (a.N == b.N && isequal(a.charge_list,b.charge_list) && isequal(a.dim_list,b.dim_list))

Base.copy(Ten::SymTensor) = SymTensor(copy.(Ten.legs), deepcopy(Ten.existing_charge_inds), deepcopy(Ten.Tensors));

rank(Ten::SymTensor) = length(Ten.legs)
Base.size(Ten::SymTensor) = tot_dim.(Ten.legs)
num_charges(Ten::SymTensor) = length(Ten.existing_charge_inds)

function full(Ten::SymTensor{T}) where T
    Tensize = size(Ten)
    Ncharges = num_charges(Ten)
    Tenfull = zeros(T, Tensize...)
    for j in 1:Ncharges #run over different charge sectors
        charge_inds = Ten.existing_charge_inds[j]
        subranges = [] #This gives the position of the charge sector in the full tensor
        for l in 1:length(charge_inds) #l are indices of tensor (1 to rank of tensor)
            ds = Ten.legs[l].dim_list
            subrange_start = sum(ds[1:charge_inds[l]-1])+1
            subrange_end =  sum(ds[1:charge_inds[l]])
            push!(subranges,subrange_start:subrange_end)
        end
        Tenfull[subranges...] = Ten.Tensors[j]
    end
    return Tenfull
end

function Base.permutedims(Ten::SymTensor,inds::Vector{Int})
    newlegs = Ten.legs[inds]
    new_charge_inds = [t[inds] for t in Ten.existing_charge_inds]
    newTensors = [permutedims(t,inds) for t in Ten.Tensors]
    return SymTensor(newlegs,new_charge_inds,newTensors)
end

function merge_leg(leg1::SymLeg, leg2::SymLeg)
    # combine two or more contiguous indices into one
    # N is the number of possible charges (Z_N case)
    N = leg1.N
    charge_list1 = leg1.charge_list
    charge_list2 = leg2.charge_list
    dim_list1 = leg1.dim_list
    dim_list2 = leg2.dim_list
    new_charge_ten = zeros(Int, length(charge_list1), length(charge_list2))
    new_dim_ten = zeros(Int, length(dim_list1), length(dim_list2))
    for i in 1:length(charge_list1)
        for j in 1:length(charge_list2)
            if(N == 0)
                new_charge_ten[i,j] = (charge_list1[i] + charge_list2[j]) 
            else
                new_charge_ten[i,j] = (charge_list1[i] + charge_list2[j]) % N
            end
            new_dim_ten[i,j] = dim_list1[i] * dim_list2[j]
        end
    end
    new_charge_list = union(new_charge_ten) #disjoint union of all entries
    new_dim_list = zero(new_charge_list)
    new_ind_list = Vector{CartesianIndex{2}}[]
    new_ind_list_inverse = zero(new_dim_ten)
    for ind_c in 1:length(new_charge_list)
        charge = new_charge_list[ind_c]
        charge_ind = findall(new_charge_ten.==charge)
        for ind in charge_ind
            new_ind_list_inverse[ind] = ind_c
        end
        new_dim_list[ind_c] = new_dim_list[ind_c] + sum([new_dim_ten[ind] for ind in charge_ind])
        push!(new_ind_list,charge_ind)
    end
    group_leg = SymLeg(N, new_charge_list,new_dim_list)
    return group_leg, new_ind_list, new_ind_list_inverse
end

function merge_index(Ten::SymTensor{T}, ind1::Int) where T
    ## merge the legs ind1 and ind1+1 of Ten, the tensor should at least be rank two
    N = Ten.legs[1].N
    r = rank(Ten)
    leg1 = Ten.legs[ind1]
    leg2 = Ten.legs[ind1+1]
    group_leg, new_ind_list, new_ind_list_inverse = merge_leg(leg1, leg2)
    new_legs = vcat(Ten.legs[1:ind1-1],group_leg,Ten.legs[ind1+2:end])
    new_charge_inds = Vector{Int}[]
    old_charge_inds = CartesianIndex{2}[]
    new_Tensors = Array{T,r-1}[]
    for i in 1:length(Ten.existing_charge_inds)
        charge_inds = Ten.existing_charge_inds[i]
        old_charge_ind = CartesianIndex(Tuple(charge_inds[[ind1,ind1+1]]))
        new_charge_ind = new_ind_list_inverse[old_charge_ind]
        push!(old_charge_inds,old_charge_ind)
        push!(new_charge_inds,vcat(charge_inds[1:ind1-1],Int[new_charge_ind],charge_inds[ind1+2:end]))
        tensor_local = Ten.Tensors[i]
        size_local = collect(size(Ten.Tensors[i]))
        reshaped_size = vcat(size_local[1:ind1-1],size_local[ind1]*size_local[ind1+1],size_local[ind1+2:end])
        push!(new_Tensors,reshape(tensor_local,Tuple(reshaped_size)))
    end
    ### merge identical charge index, concatenate the tensors in the merging dimension
    dummyind = 1
    while(dummyind<=length(new_charge_inds))
        current_charge_ind = new_charge_inds[dummyind:dummyind]
        identical_inds = findall(new_charge_inds.==current_charge_ind)[2:end] #identical inds to be deleted
        current_old_charge_inds = old_charge_inds[vcat(Int[dummyind],identical_inds)] #old charge inds of the unmerged index
        current_tensors = new_Tensors[vcat(Int[dummyind],identical_inds)] #to be concatenated
        current_tensors_size = collect(size(current_tensors[1]))
        all_old_charge_inds = new_ind_list[current_charge_ind[1][ind1]]
        temp_order = [findall(current_old_charge_inds.==[a]) for a in all_old_charge_inds]
        concat_tensors = []
        for i in 1:length(temp_order)
            od = temp_order[i]
            if(length(od)>1)
                println("leg merging error: identical charge indices")
            end
            if(length(od)==1)
                push!(concat_tensors, current_tensors[od[1]])
            else
                # cannot find an existing old charge indice, padding zero
                current_old_charge_inds = all_old_charge_inds[i]
                old_dim = leg1.dim_list[current_old_charge_inds[1]]*leg2.dim_list[current_old_charge_inds[2]]
                zerosize = vcat(current_tensors_size[1:ind1-1],Int[old_dim],current_tensors_size[ind1+1:end]) 
                push!(concat_tensors, zeros(zerosize...))
            end
        end
        new_Tensors[dummyind] = cat(concat_tensors...,dims=(ind1))
        deleteat!(new_charge_inds, identical_inds)
        deleteat!(new_Tensors, identical_inds)
        deleteat!(old_charge_inds, identical_inds)
        dummyind=dummyind+1
    end
    return SymTensor(new_legs,new_charge_inds,new_Tensors)
end

function merge_index(Ten::SymTensor{T}, inds::Vector{Int}) where T
    new_Ten = copy(Ten);
    for i in 1:length(inds)
        ind = inds[i]
        new_Ten = merge_index(new_Ten, ind)
        inds[i+1:end] = inds[i+1:end].-1
    end
    return new_Ten
end

function contract_index(Ten::SymTensor{T}, contract_leg_inds1::Vector{Int}, contract_leg_inds2::Vector{Int}) where T
    # contraction of two set of legs in one tensor
    # if no leg remains, return a scalar number, otherwise return a SymTensor
    # Note - ncon does not support one tensor, directly use @tensor
    N = Ten.legs[1].N
    if(length(contract_leg_inds1)!=length(contract_leg_inds2))
        throw("contraction order error: dangling contracted legs")
    end
    for i in 1:length(contract_leg_inds1)
        working_leg1 = Ten.legs[contract_leg_inds1[i]]
        working_leg2 = Ten.legs[contract_leg_inds2[i]]
        if(!isopposite(working_leg1,working_leg2))
            throw("contraction error: contracted legs No.$i have different charge or dimensions")
        end
    end
    scalar_flag = (rank(Ten) == 2*length(contract_leg_inds1))
    if(scalar_flag == true)
        output_scalar = 0
    else
        output_legs_inds = deleteat!(collect(1:rank(Ten)),vcat(contract_leg_inds1,contract_leg_inds2))
        output_charge_inds = Vector{Int}[]
        output_tensors = Array{T,length(output_legs_inds)}[]
    end
    IAtrace = collect(1:rank(Ten))
    IAtrace[contract_leg_inds2]=IAtrace[contract_leg_inds1]
    IAtrace = Tuple(IAtrace);
    for i in 1:length(Ten.existing_charge_inds)
        charge_inds = copy(Ten.existing_charge_inds[i])
        if(charge_inds[contract_leg_inds1] == charge_inds[contract_leg_inds2])
            current_tensor = Ten.Tensors[i]
            if(scalar_flag == false)
                output_charge_ind = deleteat!(charge_inds,vcat(contract_leg_inds1,contract_leg_inds2))
                temp_ind = findfirst(output_charge_inds.==[output_charge_ind])
                if(typeof(temp_ind)==Nothing)
                    push!(output_charge_inds, output_charge_ind)
                    push!(output_tensors, tensortrace(current_tensor, IAtrace))
                else
                    output_tensors[temp_ind]+=tensortrace(current_tensor, IAtrace)
                end
            else
                output_scalar+=tensortrace(current_tensor, IAtrace)[1]
            end
        end
    end
    return scalar_flag ? output_scalar : SymTensor(Ten.legs[output_legs_inds],output_charge_inds, output_tensors)
end

function contract_index(Ten1::SymTensor{T1}, Ten2::SymTensor{T2}, ncon_order::Vector{Vector{Int}}) where {T1,T2}
    # contraction of two tensors
    # ncon order - negative index indicate output order, identical positive index indicates contraction
    # Note - do not include contraction within one tensor in ncon_order, first use the other method to eliminate loops.
    if(length(ncon_order)!=2)
        throw("ncon order error: only supports contraction of two tensors")
    end
    if(length(ncon_order[1])!=rank(Ten1) || length(ncon_order[2])!=rank(Ten2))
        throw("ncon order error: missing or excessive number of indices")
    end
    contract_leg_ind1 = sort!(findall(ncon_order[1].>0))
    remaining_leg_ind1 = findall(ncon_order[1].<0)
    contract_leg_ind2 = sort!(findall(ncon_order[2].>0))
    remaining_leg_ind2 = findall(ncon_order[2].<0)
    s1 = sortperm(ncon_order[1][contract_leg_ind1])
    s2 = sortperm(ncon_order[2][contract_leg_ind2])
    contract_leg_ind1 = contract_leg_ind1[s1]
    contract_leg_ind2 = contract_leg_ind2[s2]
    if(ncon_order[1][contract_leg_ind1] != ncon_order[2][contract_leg_ind2])
        throw("ncon order error: dangling contracted legs")
    end
    legs1 = copy.(Ten1.legs);
    legs2 = copy.(Ten2.legs);
    contract_legs1 = Ten1.legs[contract_leg_ind1]
    contract_legs2 = Ten2.legs[contract_leg_ind2]
    for i in 1:length(contract_leg_ind1)
        working_leg1 = contract_legs1[i]
        working_leg2 = contract_legs2[i]
        if(!isopposite(working_leg1,working_leg2))
            throw("contraction error: contracted legs No.$i have different charges or dimensions")
        end
    end
    output_legs = vcat(deleteat!(legs1,sort(contract_leg_ind1)),deleteat!(legs2, sort(contract_leg_ind2)))
    output_order = reverse(sortperm(vcat(ncon_order[1][remaining_leg_ind1],ncon_order[2][remaining_leg_ind2])))
    output_legs = output_legs[output_order]
    output_charge_inds = Vector{Int64}[]
    if(T1<:Float64 && T2<:Float64)
        T = Float64
    else
        T = Complex{Float64}
    end
    output_tensors = Array{T,length(output_legs)}[]
    for temp_ind1 in 1:length(Ten1.existing_charge_inds)
        for temp_ind2 in 1:length(Ten2.existing_charge_inds)
            charge_ind1 = Ten1.existing_charge_inds[temp_ind1]
            ten1 = Ten1.Tensors[temp_ind1]
            charge_ind2 = Ten2.existing_charge_inds[temp_ind2]
            ten2 = Ten2.Tensors[temp_ind2]
            if(charge_ind1[contract_leg_ind1] == charge_ind2[contract_leg_ind2])
                output_charge_ind = vcat(charge_ind1[remaining_leg_ind1],charge_ind2[remaining_leg_ind2])[output_order]
                temp_ind = findfirst(output_charge_inds.==[output_charge_ind])
                if(typeof(temp_ind)==Nothing)
                    push!(output_charge_inds, output_charge_ind)
                    push!(output_tensors, ncon([ten1,ten2], ncon_order))
                else
                    output_tensors[temp_ind] += ncon([ten1,ten2], ncon_order)
                end
            end
        end
    end
    return SymTensor(output_legs,output_charge_inds,output_tensors)
end

function split_index(Ten::SymTensor{T}, ind::Int64, split_leg1::SymLeg, split_leg2::SymLeg) where T
    # split one index (ind) into two SymLegs with charge info and dimensions, maintaining the canonical form.
    group_leg, new_ind_list, new_ind_list_inverse = merge_leg(split_leg1,split_leg2)
    if(!isequal(group_leg, Ten.legs[ind]))
        throw("split leg error: charge info or dim info disagrees!")
    end
    legs = copy(Ten.legs)
    deleteat!(legs, ind)
    insert!(legs, ind, split_leg1);insert!(legs, ind+1, split_leg2)
    new_charge_inds = Vector{Int64}[]
    new_Tensors = Array{T,length(legs)}[]
    for i in 1:length(Ten.existing_charge_inds)
        charge_inds = Ten.existing_charge_inds[i]
        working_charge_ind = charge_inds[ind]
        working_ten = Ten.Tensors[i]
        size_ten = collect(size(working_ten))
        #println(size_ten)
        split_charge_inds = new_ind_list[working_charge_ind]
        working_dim_start = 1;
        for split_charge_ind in split_charge_inds
            #println(split_charge_ind)
            push!(new_charge_inds,vcat(charge_inds[1:ind-1],collect(Tuple(split_charge_ind)),charge_inds[ind+1:end]))
            working_dims = [split_leg1.dim_list[split_charge_ind[1]], split_leg2.dim_list[split_charge_ind[2]]]
            working_dim_length = working_dims[1]*working_dims[2]
            #println(working_dim_length)
            working_index_range = working_dim_start:working_dim_start+working_dim_length-1
            working_index = vcat([Any[:] for l in 1:length(size(working_ten))-1]...)
            insert!(working_index, ind, working_index_range)
            working_dim_tuple = Tuple(vcat(size_ten[1:ind-1],working_dims,size_ten[ind+1:end]))
            #println(working_dim_tuple)
            push!(new_Tensors,reshape(working_ten[working_index...],working_dim_tuple))
            working_dim_start+=working_dim_length
        end
    end
    return SymTensor(legs, new_charge_inds, new_Tensors)
end

function Base.conj(Ten::SymTensor{T}) where T
    legs = Ten.legs
    inds = Ten.existing_charge_inds
    return SymTensor(conj.(legs),inds,conj.(Ten.Tensors))
end

function Sym_split_trunc(Ten::SymTensor{T}, dir = "l";truncation::Bool = false, max_bd::Int = 1024, max_err::Float64 = 1E-8) where T
    ## Input Ten is a rank two SymTensor, Ten = USVt 
    ## if truncation is true, further truncate S with max_bd and max_err
    ## if (dir = "l"), return TenL = US, TenR = Vt (two SymTensor)
    ## if (dir = "r"), return TenL = U, TenR = SVt (two SymTensor)
    Us = Matrix{T}[]
    Ss = Vector{Float64}[]
    Vts = Matrix{T}[]
    ## Step 1 - SVD in each charge sector
    for t in Ten.Tensors
        t_split = nothing
        try
            t_split = svd(t,alg=LinearAlgebra.DivideAndConquer())
        catch
            t_split = svd(t,alg=LinearAlgebra.QRIteration())
        end
        U = t_split.U
        S = t_split.S
        Vt = t_split.Vt
        push!(Us, U)
        push!(Ss, S)
        push!(Vts, Vt)
    end
    ## Step 2 - truncate in each charge sector
    if(truncation == true)
        set_bds, trunc_err = truncate(Ss,max_bd,max_err)
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        for j in 1:length(Ss)
            set_bd = set_bds[j]
            if(typeof(set_bd) == Nothing)
                Ss[j] = Float64[]
            else
                # set_bd is an integer index
                Ss[j] = Ss[j][1:set_bd]
            end
        end
    end 
    ## Step 3 - Reorganize tensors in each charge sector / create dummy SymLeg
    ## Step 4 - enclose results in SymTensor
    TensL = Matrix{T}[]
    TensR = Matrix{T}[]
    dummy_charge_list = Int[]
    dummy_dim_list = Int[]
    new_existing_charge_inds_L = Vector{Int}[]
    new_existing_charge_inds_R = Vector{Int}[]
    dummy_charge_ind = 0
    if(dir == "l")
        for j in 1:length(Ss)
            S = Ss[j]
            lS = length(S)
            original_charge_inds = Ten.existing_charge_inds[j]
            if(lS>0)
                dummy_charge_ind = dummy_charge_ind+1
                push!(dummy_charge_list, Ten.legs[2].charge_list[original_charge_inds[2]])
                push!(dummy_dim_list, lS)
                push!(TensL, Us[j][:,1:lS])
                push!(TensR, diagm(0=>S)*Vts[j][1:lS,:])
                push!(new_existing_charge_inds_L,[original_charge_inds[1],dummy_charge_ind]) 
                push!(new_existing_charge_inds_R,[dummy_charge_ind,original_charge_inds[2]])
            end
        end
        dummy_leg = SymLeg(Ten.legs[1].N,dummy_charge_list,dummy_dim_list)
        TenL = SymTensor([Ten.legs[1],dummy_leg],new_existing_charge_inds_L,TensL)
        TenR = SymTensor([conj(dummy_leg),Ten.legs[2]],new_existing_charge_inds_R,TensR)
    else
        for j in 1:length(Ss)
            S = Ss[j]
            lS = length(S)
            original_charge_inds = Ten.existing_charge_inds[j]
            if(lS>0)
                dummy_charge_ind = dummy_charge_ind+1
                push!(dummy_charge_list, Ten.legs[1].charge_list[original_charge_inds[1]])
                push!(dummy_dim_list, lS)
                push!(TensL, Us[j][:,1:lS]*diagm(0=>S))
                push!(TensR, Vts[j][1:lS,:])
                push!(new_existing_charge_inds_L,[original_charge_inds[1],dummy_charge_ind])
                push!(new_existing_charge_inds_R,[dummy_charge_ind,original_charge_inds[2]])
            end
        end
        dummy_leg = SymLeg(Ten.legs[1].N,dummy_charge_list,dummy_dim_list)
        TenL = SymTensor([Ten.legs[1],conj(dummy_leg)],new_existing_charge_inds_L,TensL)
        TenR = SymTensor([dummy_leg,Ten.legs[2]],new_existing_charge_inds_R,TensR)
    end
    return Ss, TenL, TenR
end 