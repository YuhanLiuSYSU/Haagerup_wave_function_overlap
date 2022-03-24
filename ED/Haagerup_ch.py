# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:07:34 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""
import numpy as np
from scipy import sparse


class bss:
    """
    User-defined basis element object.
    
    Members
    -------
    bs, order, parent, child, sibling, mate, cousin
    
    Method
    ------
    birth, set_childsibling, set_mate, query
    
    """
    
    def __init__(self, bs, order, parent = None, child = [], 
                 sibling = [], mate = [], cousin = [], new_order = -1):
        """
        
        Parameters
        ----------
        * bs : list or np.array
            Basis element. Example: [3,0,3,0,3].
            
        * order : int
            Order in this iteration.
            
        * parent : bss, optional
            Parent of this bss instance. The default is None.
        * child : list of bss.order (list of int), optional
            Child of this bss instance, can be more than 1. The default is [].
            
        * sibling: list of bss.order (list of int), optional
            Sibling of this bss instance.
            Siblings have the same parent. 
            Example: 
                [3,0,3,0] and [3,0,3,3] are siblings
                       ^             ^
            -----------------------------------------------------------------
            * Sibling is useful for handling the last projector in PBC chain
            -----------------------------------------------------------------
            
        * mate: list of bss.order (list of int), optional
            Mate is defined such that only the first element of the basis may be 
            different. Example:
                [3,4,3,5] and [4,4,3,5] are mates
                 ^             ^
            -----------------------------------------------------------------
            * Mate is useful for handling the first projector in PBC chain
            -----------------------------------------------------------------
            
        * cousin: list of bss.order (list of int), optional
            Cousin is defined such that their parents are siblings, and the last
            element is the same. Example:
                [3,4,3,5] and [3,4,4,5] are cousins.
                     ^             ^
            -----------------------------------------------------------------
            * Cousin is useful when adding the next projector in the iteration
            -----------------------------------------------------------------
                
                
        Returns
        -------
        None.

        """
        
        self.bs = bs
        self.order = order
        self.parent = parent
        self.child = child
        self.sibling = sibling
        self.mate = mate
        self.cousin = cousin
        
        self.new_order = new_order
        
    def birth(self, child_ele, child_order):
        
        self.child.append(child_order)
        child_bs = np.hstack((np.array(self.bs),[child_ele]))
        child_obj = bss(child_bs, child_order, parent = self, 
                        child = [], sibling = [], mate = [], cousin = [], 
                        new_order = -1)

        return child_obj
    
    def set_childsibling(self, sibling_set,sibling_order):
        
        for s in sibling_set:
            s.sibling = sibling_order
                
    
    def set_mate(self, mate_order):
        self.mate = mate_order
        
        
    def get_parent_mate_child(self, p_basis):
        
        parent_mate = self.parent.mate
        for i_n,p in enumerate(parent_mate):
            c = np.array(p_basis[p].child)
            if i_n == 0:
                c_tot = c
            else:
                c_tot = np.vstack((c_tot, c))
    
        return c_tot
    
    
    def set_mate_mutual(self, mate_order, self_basis):
        for i in mate_order:
            self_basis[i].set_mate(mate_order)
        
            
    def query(self):
        
        print("basis: ")
        print(self.bs)
        print("order: ", self.order)
        print("parent: ", self.parent.bs)
        print("child: ", self.child)
        print("sibling: ", self.sibling)
        print("mate: ", self.mate)
        print("cousin: ", self.cousin)
        
        
        
def bss_toarr(basis):
    """ Data type conversion: list of bss to list of np.array """
    
    bs_arr = []
    
    for bss_item in basis:
        bs_arr.append(bss_item.bs)
        
    return bs_arr


def bss_tolst(basis):
    """ Data type conversion: list of bss to list of list """
    
    bs_lst = []
    
    for bss_item in basis:
        bs_lst.append(bss_item.bs.tolist())
        
    return bs_lst
        

def bs_tostr(basis):
    """ 
    Data type conversion: list of np.array to list of string.
    We need string for hash table.
    This is because list and np.array are both unhashable.    
    """

    
    basis_str = [None]*len(basis)
    for i, item in enumerate(basis):
        basis_str[i] = str(item)
        
    return basis_str
    
    

def basis_seed(twist_dir = 0):
    """
    Generate the seed of basis (length 2)
    The seed basis don't have parent

    Returns
    -------
    basis1 : list of bss

    """
    
    basis0 = [[0], [1], [2], [3], [4], [5]]
    
    basis = []
    for i,bs in enumerate(basis0):
        basis.append(bss(bs, i, parent = None, child = [], 
                         sibling = [], mate = [], cousin = [], new_order = -1))
    
    basis1 = generate_basis_iterate(basis, twist_dir = twist_dir)

    bs_lst = bss_toarr(basis1)
    bs_arr = np.array(bs_lst)
    for bs_ele, bs in zip(bs_lst, basis1):
        # mate_loc is an array
        mate_loc = np.where(bs_arr[:,1] == bs_ele[1])[0]
        bs.set_mate(mate_loc)
    
    return basis1



def F_symbol_sym():
    """
    F symbol of Haagerup fusion category, in the paper by Huang and Lin.

    Returns
    -------
    F : nested list of array
        F symbol

    """
    
    A = 1/2*(np.sqrt(13)-3)
    x = -1/3*(np.sqrt(13)-2)
    z = 1/6*(np.sqrt(13)+1)
    y2 = 1/12*(5-np.sqrt(13)+np.sqrt(6*(np.sqrt(13)+1)))
    y1 = 1/12*(5-np.sqrt(13)-np.sqrt(6*(np.sqrt(13)+1)))


    F_r_r = np.array([[np.sqrt(A), 0,0, x, y1, y2]])
    F_ar_r = np.array([[0,0,0,y1, y2, z]])        # [F^{a\rho}_{\rho}]_{*,p}
    F_asr_r = np.array([[0,0,0,y2, z, y1]])
    
    F_r_ar = np.array([[0,0,0,y1, y2, z]])
    F_ar_ar = np.array([[0,np.sqrt(A),0, y2, x, y1]])
    F_asr_ar = np.array([[0,0,0,z, y1, y2]])
    
    F_r_asr = np.array([[0,0,0,y2, z, y1]])
    F_ar_asr = np.array([[0,0,0,z, y1, y2]])
    F_asr_asr = np.array([[0,0,np.sqrt(A), y1,y2,x]])
    
    
    F = [[F_r_r, F_ar_r, F_asr_r],
         [F_r_ar, F_ar_ar, F_asr_ar],
         [F_r_asr, F_ar_asr, F_asr_asr]]
    
    return F




def H_seed(basis):
    """
    Generate the seed of Hamiltonian for 3 site
    
    Input: list of bss objects.
    """
    
    valid_basis = pbc_remove(basis)[0]
    valid_basis_str = bs_tostr(bss_toarr(valid_basis))
    valid_basis_hash = dict(enumerate(valid_basis_str))
    valid_res_basis = dict((v,k) for k,v in valid_basis_hash.items())
    
    F = F_symbol_sym()
        
    I = [0,1,2]
    
    x_ind = []
    y_ind = []
    val = []
    
    P = np.zeros((len(valid_basis), len(valid_basis)), dtype = float)
    
    for i_0, bss0 in enumerate(valid_basis):
        for i_f, bssf in enumerate(valid_basis):
            
            bs0, bsf = bss0.bs, bssf.bs
            
            try:
                valid_res_basis[str(bs0)]
                valid_res_basis[str(bsf)]
                
                if bs0[0] == bsf[0] and bs0[2] == bsf[2]:
                    bs_1 = bs0[0]
                    bs_3 = bs0[2]
                    bs_2 = bs0[1]
                    bs_2p = bsf[1]
                  
                    if bs_1 in I or bs_3 in I: 
                        ele1, ele2 = 1,1
                       
                    else:
                        ele1 = F[bs_3-3][bs_1-3][0][bs_2]
                        ele2 = F[bs_3-3][bs_1-3][0][bs_2p]
                        
                    P[i_0,i_f] = ele1*ele2
                    x_ind.append(bss0.order)
                    y_ind.append(bssf.order)
                    val.append(ele1*ele2)
                    
            except KeyError:
                pass

                                
    xy_ind = P.nonzero()
    
    # record the site for Koo-Saleur generator
    site = [1]*len(val)
    
    generate = [x_ind, y_ind, val, site]
    ref = [valid_res_basis, P]
    
    return generate, ref



def pbc_remove(basis_all, is_adjust = 0, is_twist = 0, is_print=0):
    """
    Remove invalid basis under PBC. This is only used for 3 site when we generate
    the seed Hamiltonian

    Parameters
    ----------
    basis_all : list of bss
        Valid basis under OBC

    Returns
    -------
    new_basis : list of bss
        Valid basis under PBC

    """
    
    # input is list of bss type
    
    
    if is_twist == 0:
        basis2 = basis_seed()
        basis3 = generate_basis_iterate(basis2)
        basis3 = bss_tolst(basis3)
        basis3_2 = basis3
    else:
        
        basis2 = basis_seed()
        basis3 = generate_basis_iterate(basis2, twist_dir = 1)
        basis3 = bss_tolst(basis3)
        
        basis2_2 = basis_seed(twist_dir = 1)
        basis3_2 = generate_basis_iterate(basis2_2, twist_dir = -1)
        basis3_2 = bss_tolst(basis3_2)
    

    new_basis = []
    
    subtract_val = 0
    
    for i_bs, basis in enumerate(basis_all):

        bsm2 = basis.bs[-2]
        bsm1 = basis.bs[-1]
        bs0 = basis.bs[0]
        bs1 = basis.bs[1]

        seg1 = [bsm2,bsm1,bs0]
        seg2 = [bsm1,bs0,bs1]
        
        if (seg1 in basis3) and (seg2 in basis3_2):
            if is_adjust == 1:
                basis.new_order = basis.order - subtract_val
                new_basis.append(basis)
            else:
                new_basis.append(basis)           
        else:
            if is_adjust == 1:
                new_basis.append(basis)
                subtract_val += 1
                
    if is_print == 1:
        print(" --- nb of valid basis: ", len(basis_all)-subtract_val)
                
    return new_basis, subtract_val



def generate_basis_iterate(old_basis_all, twist_dir = 0, is_init = -1):
    """
    Generate the basis in iterative way. Using the basis for L sites to generate
    the basis for L+1 sites.

    Parameters
    ----------
    old_basis_all : list of bss
        basis for L sites

    Returns
    -------
    new_basis_all : list of bss
        basis for L+1 sites
    """
    
    I = [0,1,2]
    if twist_dir == 1:
        tsi = [1,2,0,4,5,3]
    elif twist_dir == -1:
        tsi = [2,0,1,5,3,4]
    else:
        tsi = range(6)

    order = 0
    new_basis_all = []
    
    for old_basis in old_basis_all:
        last_ele = old_basis.bs[-1]
        
        if last_ele in I:
            
            new_basis = old_basis.birth(tsi[last_ele+3], order)            
            new_basis_all.append(new_basis) 
                          
            order += 1 
            
            old_basis.set_childsibling([new_basis], [new_basis.order])
            
        else:
            new_ele_all = [tsi[last_ele-3], 3, 4, 5]
            sibling_set = []
            sibling_order = []
            
            for new_ele in new_ele_all:
                new_basis = old_basis.birth(new_ele, order)            
                new_basis_all.append(new_basis) 
                
                sibling_set.append(new_basis)
                sibling_order.append(order)
                order += 1
                
            old_basis.set_childsibling(sibling_set, sibling_order)
        
        
    if len(new_basis_all[0].bs)>2:  
        
        for new_basis in new_basis_all:
            
            if new_basis.mate == []:
                pm_child = new_basis.get_parent_mate_child(old_basis_all)
                
                if len(pm_child.shape) == 1:
                    new_basis.mate.append(new_basis.order)          
                    
                else:
                    for i in range(pm_child.shape[1]):
                        mates = pm_child[:,i]
                        new_basis.set_mate_mutual(mates, new_basis_all)

                        
        for old_basis in old_basis_all:
            for sb in old_basis.sibling:
                prt_sib = old_basis_all[sb]
                
                for my_kid_id in old_basis.child:
                    for sib_kid_id in prt_sib.child:
                        my_kid = new_basis_all[my_kid_id]
                        sib_kid = new_basis_all[sib_kid_id]
                        
                        if my_kid.bs[-1] == sib_kid.bs[-1]:
                            my_kid.cousin.append(sib_kid_id)
                    
    return new_basis_all

    

def generate_H_iterate(old_basis, old_generate, ref, is_last = 0, is_twist = 0, 
                       is_three = 0):
    """
    * Generate the OBC Hamiltonian in iterative way. 
      Use the Hamiltonian of L sites to generate the Hamiltonian of L+1 sites
      
    * cousin is used here to add new projector.

    Parameters
    ----------
    old_basis : list of bss
        basis of L sites Hamiltonian
    old_generate : [x_ind, y_ind, val]
        x index, y index, and value of non-zero elements of L sites Hamiltonian
    ref : [val0, valid_res_basis, xy_hash_res] 
        non-zero elements in 3 site case.
        valid_res_basis, xy_hash_res are hash tables. Allow fast query.

    Returns
    -------
    new_basis : list of bss
        basis of L+1 sites Hamiltonian
    generate : [x_ind, y_ind, val]
        x index, y index, and value of non-zero elements of L+1 sites Hamiltonian.
        This can be used to generate sparse matrix

    """
   
       
    [x_ind_old, y_ind_old, val_old, site_old] = old_generate
    [valid_res_basis, P36] = ref
    
    x_ind = []
    y_ind = []
    val = []
    site = []
    
    if is_three == 0:
        new_basis = generate_basis_iterate(old_basis)
    else:
        new_basis = old_basis
        
    
    rm = 0
    if is_last == 1:
        new_basis, rm = pbc_remove(new_basis, is_adjust = 1,
                                   is_twist = is_twist, is_print = 1)
    
    n_site = len(new_basis[0].bs)-2
    
    for x_id, y_id, val_o, site_o in zip(x_ind_old, y_ind_old, val_old, site_old):
        child1 = old_basis[x_id].child
        child2 = old_basis[y_id].child
        
        for c1, c2 in zip(child1, child2):
            if is_last == 0:
                
                x_ind.append(c1)
                y_ind.append(c2)
                val.append(val_o)
                site.append(site_o)
                
            else:
                bss_c1 = new_basis[c1]
                bss_c2 = new_basis[c2]
                
                if bss_c1.new_order>=0 and bss_c2.new_order>=0:
                
                    x_ind.append(bss_c1.new_order)
                    y_ind.append(bss_c2.new_order)
                    val.append(val_o)
                    site.append(site_o)
    

    for bss in new_basis:
        # cousin
        
        seg = str(bss.bs[-3::])
        try:
            loc1 = valid_res_basis[seg]
        
            cousin_od = bss.cousin
            for od in cousin_od:
                cs = new_basis[od]
                seg_cs = str(cs.bs[-3::])
                try:
                    loc2 = valid_res_basis[seg_cs]
                    
                    if is_last == 0:
                        
                        x_ind.append(bss.order)
                        y_ind.append(cs.order)
                        val.append(P36[loc1, loc2])
                        site.append(n_site)
                        
                    else:
                        if bss.new_order>=0 and cs.new_order>=0:
                        
                            x_ind.append(bss.new_order)
                            y_ind.append(cs.new_order)
                            val.append(P36[loc1, loc2])
                            site.append(n_site)
                    
                except KeyError:
                    pass
                
        except KeyError:
            pass
        
    if is_last == 0:
        generate = [x_ind, y_ind, val, site]
    else:
        generate = [x_ind, y_ind, val, rm, site]
    

    return new_basis, generate
    
    
    
def add_boundary(basis, generate, ref):
    """
    * After generating the OBC Hamiltonian, we add the first and last projectors 
      to make it PBC.
      
    * Sibling is used for last projector
    * Mate is used for first projector

    Parameters
    ----------
    basis : list of bss 
        
    generate : [x_ind, y_ind, val]
        x index, y index, and value of non-zero elements of L sites Hamiltonian.
        
    ref : [val0, valid_res_basis, xy_hash_res] 
        non-zero elements in 3 site case.
        valid_res_basis, xy_hash_res are hash tables. Allow fast query.

    Returns
    -------
    H : sparse.csr_matrix
        PBC hamiltonian

    """
    
    [x_ind, y_ind, val, rm, site] = generate
    [valid_res_basis, P36] = ref
    
    n_site = len(basis[0].bs)-1
    
    # After all iteration, handle the boundary...
    # H in sibling and mate
    for bss in basis:
        
        # sibling
        seg = str(np.array([bss.bs[-2], bss.bs[-1], bss.bs[0]]))
        try:
            loc1 = valid_res_basis[seg]
        
            sibling_od = bss.sibling
            for od in sibling_od:
                sb = basis[od]
                seg_sb = str(np.array([sb.bs[-2], sb.bs[-1], sb.bs[0]]))
                try:
                    loc2 = valid_res_basis[seg_sb]
                                       
                    x_ind.append(bss.new_order)
                    y_ind.append(sb.new_order)
                    val.append(P36[loc1, loc2])
                    site.append(n_site)
                    
                except KeyError:
                    pass
                
        except KeyError:
            pass
        
        
        # # mate
        seg = str(np.array([bss.bs[-1], bss.bs[0], bss.bs[1]]))
        try:
            loc1 = valid_res_basis[seg]
        
            mate_od = bss.mate
            for od in mate_od:
                mt = basis[od]
                seg_mt = str(np.array([mt.bs[-1], mt.bs[0], mt.bs[1]]))
                try:
                    loc2 = valid_res_basis[seg_mt]
                    x_ind.append(bss.new_order)
                    y_ind.append(mt.new_order)
                    val.append(P36[loc1,loc2])
                    site.append(0)
                    
                except KeyError:
                    pass
                
        except KeyError:
            pass
        
    H = (-1)*sparse.csr_matrix((np.array(val), (x_ind, y_ind)), 
                         shape=(len(basis)-rm, len(basis)-rm),dtype=float)
    
    print(len(val))
    H_gen = [x_ind, y_ind, val, site, rm]
    
    return H, H_gen


def Trans_op(basis):
    # Translational symmetry
    # TODO: too slow
    
    valid_basis = []
    for bss in basis:
        if bss.new_order>=0:
            valid_basis.append(bss)
    
    
    basis_array = bss_toarr(valid_basis)
    basis_str = bs_tostr(basis_array)
    
    basis_hash = dict(enumerate(basis_str))
    res_basis = dict((v,k) for k,v in basis_hash.items())
    
    x_array = []
    y_array = []
    val_array = []
    
    for i_0, bs0 in enumerate(basis_array):
        x_array.append(i_0)
        bs_shift = bs0[1:]
        bs_shift = np.hstack((bs_shift, bs0[0]))

        shift_loc = res_basis[str(bs_shift)]

        y_array.append(shift_loc)
        val_array.append(1) 
        
    trans = sparse.csr_matrix((np.array(val_array), (x_array, y_array)), 
                              shape=(len(basis_array), len(basis_array)),dtype=float)


    valid_b = [basis_array, basis_hash, res_basis]
    
    return trans, valid_b


def Z3trans(i):
    
    if i<3:
        im = (i+1) % 3
    else:
        im = ((i-3+1) % 3)+3  
        
    return int(im)


def Z3_symm(valid_b):
    # work for three sites. Intrinsic Z3 symmetry
   
    [basis_array, basis_hash, res_basis] = valid_b
    
    x_array = []
    y_array = []
    val_array = []
    
    for i_0, bs0 in enumerate(basis_array):
        x_array.append(i_0)
        
        N_bs = len(bs0)
        bs_Z3 = np.zeros(N_bs, dtype = int)
        for i in range(N_bs):            
            bs_Z3[i] = Z3trans(bs0[i])
            
        Z3_loc = res_basis[str(bs_Z3)]
        y_array.append(Z3_loc)
        val_array.append(1) 
            
    Z3m = sparse.csr_matrix((np.array(val_array), (x_array, y_array)), 
                              shape=(len(basis_array), len(basis_array)),dtype=float)
    
    return Z3m



   