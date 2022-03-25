using LinearAlgebra

function overlap_3state(ex1::puMPS.puMPSTvec, ex2::puMPS.puMPSTvec, ex3::puMPS.puMPSTvec)
    ## overlap <ex1 ex2|ex3>
    N1 = puMPS.num_sites(ex1)
    N2 = puMPS.num_sites(ex2)
    N3 = puMPS.num_sites(ex3)
    @assert N1+N2==N3
    M1A,M1B = computeMs(ex1,ex3,0)
    M2A,M2B = computeMs(ex2,ex3,N1)
    ov = tr(M1B*M2A+M1A*M2B)
    return ov
end

function overlap_3state(ex1::puMPS.puMPState, ex2::puMPS.puMPState, ex3::puMPS.puMPState)
    ## overlap <ex1 ex2|ex3>
    N1 = puMPS.num_sites(ex1)
    N2 = puMPS.num_sites(ex2)
    N3 = puMPS.num_sites(ex3)
    @assert N1+N2==N3
    M1A = computeMA(ex1,ex3)
    M2A = computeMA(ex2,ex3)
    ov = tr(M1A*M2A)
    return ov
end

function overlap_3state(ex1::puMPS.puMPSTvec, ex2::puMPS.puMPSTvec, ex3::puMPS.puMPState)
    ## overlap <ex1 ex2|ex3>
    N1 = puMPS.num_sites(ex1)
    N2 = puMPS.num_sites(ex2)
    N3 = puMPS.num_sites(ex3)
    @assert N1+N2==N3
    M1A = computeMA(ex1,ex3)
    M2A = computeMA(ex2,ex3)
    ov = tr(M1A*M2A)
    return ov
end

function overlap_3state(ex1::puMPS.puMPState, ex2::puMPS.puMPState, ex3::puMPS.puMPSTvec)
    N1 = puMPS.num_sites(ex1)
    N2 = puMPS.num_sites(ex2)
    N3 = puMPS.num_sites(ex3)
    @assert N1+N2==N3
    M1A,M1B = computeMs(ex1,ex3,0)
    M2A,M2B = computeMs(ex2,ex3,N1)
    ov = tr(M1A*M2B+M1B*M2A)
    return ov
end

function computeMs(exbra::puMPS.puMPSTvec, exket::puMPS.puMPSTvec, pos::Int)
    A_ket, B_ket = puMPS.mps_tensors(exket)
    A_bra, B_bra = puMPS.mps_tensors(exbra)
    p_ket = puMPS.momentum(exket)
    p_bra = puMPS.momentum(exbra)
    N_bra = puMPS.num_sites(exbra) #N_bra<N_ket
    
    TAA = puMPS.TM_dense(A_ket,A_bra)
    TAB = cis(-p_bra).*puMPS.TM_dense(A_ket,B_bra)
    TBA = cis(p_ket).*puMPS.TM_dense(B_ket,A_bra)
    TBB = cis(p_ket-p_bra).*puMPS.TM_dense(B_ket,B_bra)
    
    for j in 2:N_bra
        TBB = puMPS.applyTM_l(A_ket,A_bra,TBB)
        BLAS.axpy!(cis(j*p_ket),puMPS.applyTM_l(B_ket,A_bra,TAB),TBB)
        BLAS.axpy!(cis(-j*p_bra),puMPS.applyTM_l(A_ket,B_bra,TBA),TBB)
        BLAS.axpy!(cis(j*(p_ket-p_bra)),puMPS.applyTM_l(B_ket,B_bra,TAA),TBB)       
        
        TAB = puMPS.applyTM_l(A_ket,A_bra,TAB)
        BLAS.axpy!(cis(-j*p_bra),puMPS.applyTM_l(A_ket,B_bra,TAA),TAB)
              
        TBA = puMPS.applyTM_l(A_ket,A_bra,TBA)
        BLAS.axpy!(cis(j*p_ket),puMPS.applyTM_l(B_ket,A_bra,TAA),TBA)
        
        TAA = puMPS.applyTM_l(A_ket,A_bra,TAA)
    end
    
    @tensor MA[l,r]:= TAB[l,s,b,r,s,b]
    @tensor MB[l,r]:= TBB[l,s,b,r,s,b]
    
    rmul!(MA,cis(pos*p_ket)) ##???
    rmul!(MB,cis(pos*p_ket))
    
    return MA,MB
end

function computeMs(exbra::puMPS.puMPState, exket::puMPS.puMPSTvec,pos::Int)
    #<ep|II>
    A_bra = puMPS.mps_tensor(exbra)
    A_ket, B_ket = puMPS.mps_tensors(exket)
    p_ket = puMPS.momentum(exket)
    N_bra = puMPS.num_sites(exbra) #N_bra<N_ket
    
    TAA = puMPS.TM_dense(A_ket,A_bra)
    TBA = cis(p_ket).*puMPS.TM_dense(B_ket,A_bra)
    
    for j in 2:N_bra          
        TBA = puMPS.applyTM_l(A_ket,A_bra,TBA)
        BLAS.axpy!(cis(j*p_ket),puMPS.applyTM_l(B_ket,A_bra,TAA),TBA)              
        
        TAA = puMPS.applyTM_l(A_ket,A_bra,TAA)
    end
    
    @tensor MB[l,r]:= TBA[l,s,b,r,s,b]
    @tensor MA[l,r]:= TAA[l,s,b,r,s,b]
    
    rmul!(MB,cis(pos*p_ket))
    return MA,MB
end

function computeMA(M1::puMPS.puMPState,M2::puMPS.puMPState)
    A1 = puMPS.mps_tensor(M1)
    A2 = puMPS.mps_tensor(M2)
    N1 = puMPS.num_sites(M1)
    
    T21 = puMPS.TM_dense(A2,A1)
    for j in 2:N1
        T21 = puMPS.applyTM_l(A2,A1,T21)
    end
    
    @tensor MA[l,r]:= T21[l,s,b,r,s,b]
    return MA
end

function computeMA(exbra::puMPS.puMPSTvec, exket::puMPS.puMPState)
    A_ket = puMPS.mps_tensor(exket)
    A_bra, B_bra = puMPS.mps_tensors(exbra)
    p_bra = puMPS.momentum(exbra)
    N_bra = puMPS.num_sites(exbra) #N_bra<N_ket
    
    TAA = puMPS.TM_dense(A_ket,A_bra)
    TAB = cis(-p_bra).*puMPS.TM_dense(A_ket,B_bra)
    
    for j in 2:N_bra          
        TAB = puMPS.applyTM_l(A_ket,A_bra,TAB)
        BLAS.axpy!(cis(-j*p_bra),puMPS.applyTM_l(A_ket,B_bra,TAA),TAB)              
        
        TAA = puMPS.applyTM_l(A_ket,A_bra,TAA)
    end
    
    @tensor MA[l,r]:= TAB[l,s,b,r,s,b]
    
    return MA
end

