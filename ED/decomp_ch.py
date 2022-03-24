# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:37:33 2021

This is a replicate of eig.decomp, for the purpose of using on cluster.
** DO NOT MAKE DIRECT EDIT TO THIS FILE **

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""
import numpy as np
import scipy.linalg as alg
from math import pi
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs as sparse_eigs


ERROR_CUTOFF = 10**(-6)


def check_diag(matr, is_show = 0):
    matr_remove = matr-np.diag(np.diag(matr))
    diag_error = np.sum(abs(matr_remove))
    
    # if (diag_error > 10**(-6) and is_show == 1):
    #     plt.imshow(abs(matr), cmap = 'jet')
    #     plt.colorbar()
    #     # plt.rcParams["figure.figsize"] = (10,10)
    #     # plt.xticks(fontsize=20)
    #     # plt.yticks(fontsize=20)
    #     plt.show()
          
    return diag_error



def sort_real(eigval, eigvecs):
    idx = eigval.real.argsort()[::1]   
    eigval = eigval[idx]
    eigvecs = eigvecs[:,idx]

    return eigval, eigvecs



def sort_block_(regV):
    reg_num = (regV.shape)[1]
    
    overlap = regV.conj().T @ regV
    if np.sum(abs(overlap - np.identity(reg_num)))>10**(-7):
        
        print("Sort manually...")
        eig_ov, vec_ov = alg.eigh(overlap)
        regV = regV @ vec_ov
        vr_norm = np.diag(1/np.sqrt(np.diag(regV.conj().T @ regV)))
        regV = regV @ vr_norm

    return regV



def simult_diag(H, M, knum = -1, is_phase = 0, is_show = 0, is_sort = 1, bands = 1,
                is_zero_sym = 1, N_chain = None):
    
    """
    Simultaneously diagonalize H and M. (or H, M[0], M[1])

    Although Hp is not hermitian, for some reason, the eigvecs are still 
    orthogonal. The error is very small, as long as there is no further
    degeneracy. 
    
    TODO: "is_zero_sym" part only works for two zero states
    """
    
    if isinstance(M, np.ndarray): M = [M]
    if issparse(M): M = [M]

    
    if issparse(H)==1:
        N_dimH = H.get_shape()[0]
        if N_chain == None: N_chain = int(np.log2(N_dimH))
    else:
        if N_chain == None: N_chain = len(H)
    
    
    ep = 10**(-4)
    ep2 = 10**(-5)
    if len(M) == 1:
        Hp = H + ep*M[0]
    elif len(M) == 2:
        Hp = H + ep*M[0] + ep2*M[1]
    
    if knum > 0:
        eigval, eigvecs = sparse_eigs(Hp, k=knum, which='SR')
    else:
        eigval, eigvecs = alg.eig(Hp)
        
    
    if is_sort == 1: 
        eigval, eigvecs = sort_real(eigval, eigvecs)
        
    
    eig_H = eigvecs.conj().T@ H @ eigvecs
    eig_M0 = eigvecs.conj().T@ M[0] @ eigvecs
    if len(M) == 2:
        eig_M1 = eigvecs.conj().T@ M[1] @ eigvecs
    
    
    # Resolve the residual degeneracy manually...
    label = []
    if len(M) == 1:
        for i in range(len(eig_H)-1):
            if abs(eig_H[i,i]-eig_H[i+1,i+1])<ERROR_CUTOFF \
                and abs(eig_M0[i,i]-eig_M0[i+1,i+1])<ERROR_CUTOFF:
                    label.append(i)
    elif len(M) == 2:
        for i in range(len(eig_H)-1):
            if abs(eig_H[i,i]-eig_H[i+1,i+1])<ERROR_CUTOFF \
                and abs(eig_M0[i,i]-eig_M0[i+1,i+1])<ERROR_CUTOFF\
                    and abs(eig_M1[i,i]-eig_M1[i+1,i+1])<ERROR_CUTOFF:
                    label.append(i)
        
   
    if bool(label):
        start = []
        end = []
        
        for ele in label:
            if not ele-1 in label: start.append(ele)
            if not ele+1 in label: end.append(ele+1)
        
        for i in range(len(start)):
            reg = range(start[i], end[i]+1)
            regV = eigvecs[:,reg]
            regV = sort_block_(regV)
            
            eigvecs[:,reg] = regV
            
    if bool(label):
        eig_H = eigvecs.conj().T@ H @ eigvecs
        eig_M0 = eigvecs.conj().T@ M[0] @ eigvecs
       
    print(" [simult_diag] error for orthonormal: %f" 
          % check_diag(eigvecs.conj().T @ eigvecs, is_show = is_show))
    
    print(" [simult_diag] error for H: %f" 
          % check_diag(eig_H, is_show = is_show))
        
    print(" [simult_diag] error for M[0]: %f" 
          % check_diag(eig_M0, is_show = is_show))
    
    if len(M) == 2:
        eig_M1 = eigvecs.conj().T@ M[1] @ eigvecs
        print(" [simult_diag] error for M[1]: %f" 
              % check_diag(eig_M1, is_show = is_show))
    
    

    if is_phase == 1:
        # should be integers
        eig_M0 = np.angle(np.diag(eig_M0))*N_chain/(2*pi)
        eig_M0 = eig_M0/bands
        
        l_brillouin = N_chain/bands
        
        left = -l_brillouin/2 - 10**(-4)
        right = l_brillouin/2 + 10**(-4)
    
        loc_l_out = np.where(eig_M0<left)[0]
        eig_M0[loc_l_out] = eig_M0[loc_l_out] + l_brillouin
        loc_r_out = np.where(eig_M0>right)[0]
        eig_M0[loc_r_out] = eig_M0[loc_r_out] - l_brillouin
        
    else:
        eig_M0 = np.diag(eig_M0)
        
        
    if is_zero_sym == 1:
        eig_Hd = np.diag(eig_H).real
        zero_loc = np.where(abs(eig_Hd)<10**(-8))[0]
        flip_mtr = np.eye(len(eig_H))
        
        if len(zero_loc)>1:
            if (eig_Hd[zero_loc[0]]>0 and eig_Hd[zero_loc[1]]>0) or\
                (eig_Hd[zero_loc[0]]<0 and eig_Hd[zero_loc[1]]<0):
                
                flip_mtr[zero_loc[0],zero_loc[0]] = -1
                eig_H = eig_H @ flip_mtr
       
    eig_H = np.diag(eig_H).real
    
       
    if len(M) == 1: eig_M = eig_M0
    elif len(M) == 2: eig_M = [eig_M0, np.diag(eig_M1)]
    
    return eig_H, eigvecs, eig_M


def sort_P_nonh(E,V,P,H,N):
    
    # TODO: there is a duplicate of this in spin_tool

    R = V+np.zeros(V.shape,dtype=complex)
    L = R.conj()
    # Need to specify V is complex. Otherwise it will take real part of V[:,reg]=regV@Vtrans
    labels=[-1]
    for i in range(len(E)-1):
        if (E[i+1]-E[i]).real>0.0000001:
            labels.append(i)
            
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg=range(labels[i]+1,labels[i+1]+1)
            regV=V[:,reg]
            Peig=regV.T @ P @ regV
            # Sometimes, S obtained from the eigenvalue of Peig is not integer... 
            # This may because of our numerical way to get the eigensystem. 
            # Some states might be failed to be included.
            
            # Peig is not necessarily hermitian! Using eig might not be safe? 
            # I guess it is still safe because Peig = V*P_{diag}*V^{-1} is still valid
            
            S,Vtrans=alg.eig(Peig)
            R[:,reg] = regV@Vtrans
            L[:,reg] = regV.conj()@(alg.inv(Vtrans)).conj().T

            
            # After this, L is not necessary the conjugate of R
    
    P_eig = L.conj().T @ P @ R
    S = np.angle(P_eig.diagonal())*N/(2*pi)
    print("error for orthonormal: %f" % 
      check_diag(L.conj().T @ R, is_show = 1))
    print("error for H: %f" % 
      check_diag(L.conj().T @ H @ R, is_show = 1))
    print("error for P: %f" % 
      check_diag(P_eig, is_show = 1))
               
    
    return R, L, S


def sort_biortho(hamiltonian,knum = -1, eig_which='SR', PT='true'):
    
    # knum is only used for large system
    """
    #--------------------------------------------------------------------------#
    # COMMENT:
    # 1. If H is symmetric, for H|R> = E|R>, H^\dag |L> = E^* |L>, we have:
    #   |L> = |R>^*
    #
    # 2. Here PT = 'true' means both the Hamiltonian and the eigenstates preserve
    #   the PT symmetry. This guarantees all the eigenvalues are real.
    #   (Seems like it does not matter in numerics...)
    #--------------------------------------------------------------------------#
    """
    
    if knum > 0:
        eigval, eigvecs = sparse_eigs(hamiltonian, k=knum, which=eig_which)
    else:    
        eigval, eigvecs = alg.eig(hamiltonian)
        
    eigval, eigvecs = sort_real(eigval, eigvecs)

    if PT!='true':
        if knum > 0:
            eigval_L, eigvecs_L = sparse_eigs(hamiltonian.conj().T, k=knum, which=eig_which)
        else:
            eigval_L, eigvecs_L = alg.eig(hamiltonian.conj().T)
        idx = eigval_L.argsort()[::1]   
        eigval_L = eigval_L[idx]
        eigvecs_L = eigvecs_L[:,idx]
    
    
    V_norm = np.diag(1/np.sqrt(np.diag(eigvecs.conj().T@eigvecs)))
    eigvecs = eigvecs@V_norm
    
    labels=[-1]
    eigvecs_sort = eigvecs+np.zeros(eigvecs.shape,dtype=complex)
    for i in range(len(eigval)-1):
        if abs(eigval[i+1]-eigval[i])>10**(-7):
            labels.append(i)
            
    if (labels.count(len(eigval)-1) == 0):
        labels.append(len(eigval)-1)

        
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg = range(labels[i]+1,labels[i+1]+1)
            regVR = eigvecs[:,reg] 
            
            
            if np.sum(abs(regVR.T@regVR-np.identity(len(reg))))>10**(-7):
                
                V_unnorm = __Takagifac(regVR)
                eig_fac = np.diag(1/np.sqrt(np.diag(V_unnorm.T@V_unnorm)))               
                
                V_norm = V_unnorm@eig_fac
                overlap = V_norm.T @ V_norm
                
                check_diag(overlap)
                
                subreg = []
                for j in range(len(reg)-1):
                    # Sort again
                    
                    if abs(overlap[j,j+1])>0.000001:
                        subreg.extend([j,j+1])
                        
                subreg = list(set(subreg))
                if subreg!=[]:
                    
                    subreg_VR = V_norm[:,subreg]
                    V_unnorm_2 = __Takagifac(subreg_VR)
                    eig_fac_2 = np.diag(1/np.sqrt(np.diag(V_unnorm_2.T@V_unnorm_2)))   
                    V_norm_22 = V_unnorm_2@eig_fac_2
                    V_norm[:,subreg] = V_norm_22
                    
                    # test4 = test
                    # test4[:,subreg] = V_norm_22
                    # test3 = test4.T @ test4
                    # plt.imshow(abs(test3), cmap = 'jet')
                    # plt.colorbar()
                                          
                eigvecs_sort[:,reg] = V_norm
                
    V_norm = np.diag(1/np.sqrt(np.diag(eigvecs_sort.T@eigvecs_sort)))
    eigvecs_sort = eigvecs_sort@V_norm        

    is_show = 0
    print(" [sort_biortho] error for orthonormal: %f" % 
      check_diag(eigvecs_sort.T @ eigvecs_sort,is_show))
    print(" [sort_biortho] error for H: %f" % 
      check_diag(abs(eigvecs_sort.T@ hamiltonian @eigvecs_sort),is_show))
    
    R = eigvecs_sort
    L = eigvecs_sort.conj()
    
    return eigval, R, L


def __Takagifac(R):
    # Autonne-Takagi factorization
    # D = UAU^T where A is a complex symmetric matrix, U is a unitary. D is real non-negative matrix
    
    # https://en.wikipedia.org/wiki/Symmetric_matrix#Complex_symmetric_matrices

    
    A = R.T @ R
    
    if (abs(A-np.diag(np.diag(A))).sum()) > 10**(-6): 
        
        _,V = alg.eigh(A.conj().T @ A)        
        C = V.T @ A @ V

        if (abs(C-np.diag(np.diag(C))).sum()) > 10**(-6):   
            _,W = alg.eigh((V.T @ A @ V).real)
            U = W.T @ V.T
        else:
            U = V.T
            
        Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U    
        
        R = R@Up.T
    
    return R