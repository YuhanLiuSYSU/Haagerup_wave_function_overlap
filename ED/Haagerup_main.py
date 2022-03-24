# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:52:16 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""
import numpy as np
from scipy import sparse
from math import pi
import time

import Haagerup_ch as Hgr

from decomp_ch import simult_diag

try:
    # -----------------------
    # for local machine
    # the source files of the package can be downloaded from:
    #   https://github.com/YuhanLiuSYSU/phys-toolkit/tree/main/phys_python
    # -----------------------
    from toolkit.plot_style import plot_style_s

except ModuleNotFoundError:
    # for cluster without the above package
    pass


CUTOFF = 10**(-6)

def select_Z3(Z3eig, sel=0):
    
    valid_state = []
    for i in range(len(Z3eig)):
        if abs(Z3eig[i]-sel)<10**(-6):
            valid_state.append(i)
            
    return valid_state


def Koo_Saleur(H_gen, KS_val, N_basis):
    
    [x_ind, y_ind, val, site, rm] = H_gen
    val_n = np.array(val)*np.exp(1j*np.array(site)*(2*pi)/N_s*KS_val)
    Hn = (-1)*sparse.csr_matrix((np.array(val_n), (x_ind, y_ind)), 
                          shape=(N_basis, N_basis),dtype=np.complex128)
    
    return Hn


def E_pick(E, V, M, sel = 0):
    
    valid_st = select_Z3(M[1], sel = sel)
    V_valid = V[:,valid_st]
    E_valid = E[valid_st]
    M_valid = M[0][valid_st]
    
    return  E_valid, V_valid, M_valid
    

def anyon_overlap(v1, v2, v3, vb_1, vb_2, vb_3):
    ov = 0
    
    l1 = len(vb_1[0][0])
    l2 = len(vb_3[0][0])

    hbasis1 = vb_1[2]
    hbasis2 = vb_2[2]
    
    for i in range(len(v3)):
        vbs_3 = vb_3[0][i]
        seg_one = vbs_3[0:l1]
        seg_two = vbs_3[l1:l2]

        try:
            loc1 = hbasis1[str(seg_one)]
            
            try:
                loc2 = hbasis2[str(seg_two)]
                
                ov = ov + v1[loc1]*v2[loc2]*v3[i]
                
            except KeyError:
                pass
            
        except KeyError:
            pass
            
                
    return ov



def main_solver(N_s, knum, is_twist = 0):
    """
    Main function for solving the Haagerup cft chain.

    Parameters
    ----------
    N_s : int
        number of sites
    knum : int
        number of eigenstates to solve in sparse matrix
        
    is_twist: 0 or 1
    TODO: the part on twist boundary condition is not finished yet

    Returns
    -------
    E, V, M

    """
    start_time = time.time()
    
    print(" --- Ns: ", N_s)
    print(" --- knum: ", knum)
    
    basis2 = Hgr.basis_seed(twist_dir=-is_twist) # length 2
    basis = Hgr.generate_basis_iterate(basis2) # length 3
    
    generate, ref = Hgr.H_seed(basis)
    
    if N_s>3:
        is_last = 0
        is_twist = 0
        for i in range(N_s-3):
            if i == N_s-4: 
                is_last = 1
                is_twist = is_twist
            basis, generate = Hgr.generate_H_iterate(basis, generate, ref, 
                                is_last = is_last, is_twist = is_twist, is_three = 0)
    else:
        is_last = 1
        basis, generate = Hgr.generate_H_iterate(basis, generate, ref, 
                                is_last = is_last, is_twist = is_twist, is_three = 1)
        
    
    print(" --- Add boundary terms...")
    H, H_gen = Hgr.add_boundary(basis, generate, ref)
    print(" --- Hamiltonian:  %s seconds ---\n" % (time.time() - start_time))  
    
    trans_op, valid_b = Hgr.Trans_op(basis)
    print(" --- Trans_op:  %s seconds ---\n" % (time.time() - start_time))  
    Z3_op = Hgr.Z3_symm(valid_b)
    print(" --- Z3_op:  %s seconds ---\n" % (time.time() - start_time))  
    
    E, V, M = simult_diag(H, [trans_op,Z3_op], knum=knum,is_phase = 1,N_chain = N_s)
    print(" --- Solve:  %s seconds ---\n" % (time.time() - start_time))    
    
    M[1] = np.angle(M[1])/(2*pi/3)

    return E, V, M, H_gen, valid_b



if __name__ == "__main__":
    
    task = 1
    
    if task == 1:
        """
        Compute energy-momentum spectrum for a given size N_s.
        ---
        N_s: system size
        kum: number of lowest eigenstates to solve
        
        For the case N_s, knum = 9, 200, runtime is around 190s. 
        """
        
        N_s, knum = 9, 200
        
        E, V, M, H_gen, valid_b = main_solver(N_s, knum)
              
        E_valid, V_valid, M_valid = E_pick(E, V, M, sel=0)
        E_valid_p, _, M_valid_p = E_pick(E, V, M, sel=1)
        E_valid_m, _, M_valid_m = E_pick(E, V, M, sel=-1)
        
        H2 = Koo_Saleur(H_gen, 2, len(valid_b[0]))
        ov2 = V.conj().T @ H2 @ V[:,0:12]
                
        T_loc = abs(ov2[:,0]).argmax(axis=0)
        norm_fac = 2/(E[T_loc] - E[0])
        E_valid_norm = (E_valid-min(E_valid))*norm_fac 
        
        
        try:
            ax,_ = plot_style_s(M_valid, E_valid_norm, is_line = 0, scatter_size = 25, 
                                x_labels = '$P$', y_labels = '$\Delta$')
            
            
        except: NameError


    elif task == 2:
        """
        Compute wavefunction overlap.
        """
        
        N_s1, N_s2, N_s3 = 6, 6, 12
        knum1, knum2, knum3 = 100, 100, 15
              
        E1, V1, M1, H_gen1, vbasis1 = main_solver(N_s1, knum1)
        E2, V2, M2, H_gen2, vbasis2 = main_solver(N_s2, knum2)
        E3, V3, M3, H_gen3, vbasis3 = main_solver(N_s3, knum3)

        # M[1] is the Z3 eigenvalue
        P = [[1,1,0]]
        Z3 =[[0,0,0]]
        State = [[0,1,0]]
        Ov = np.zeros(len(P), dtype = complex)
        
        for i in range(len(P)):
            p = P[i]
            z3 = Z3[i]
            St = State[i]
        
            loc1 = np.where((abs(M1[0]-p[0])<CUTOFF) & \
                            (abs(M1[1]-z3[0])<CUTOFF))[0][St[0]]
            loc2 = np.where((abs(M2[0]-p[1])<CUTOFF) & \
                            (abs(M2[1]-z3[1])<CUTOFF))[0][St[1]]
            loc3 = np.where((abs(M3[0]-p[2])<CUTOFF) & \
                            (abs(M3[1]-z3[2])<CUTOFF))[0][St[2]]
            
            v1 = V1[:,loc1]
            v2 = V2[:,loc2]
            v3 = V3[:,loc3]
            
            ov = anyon_overlap(v1, v2, v3, vbasis1, vbasis2, vbasis3)
            Ov[i] = ov
            print("loc1: ", loc1)
            print("loc2: ", loc2)
            print("loc3: ", loc3)
            print("p: ", p)
            print("z3: ", z3)
            print("st: ", St)
            print("ov: ", abs(ov))
            print()

