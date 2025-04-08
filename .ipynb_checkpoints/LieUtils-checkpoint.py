import numpy as np
from copy import deepcopy
from functools import reduce
from qiskit.quantum_info import Operator,SparsePauliOp

def inner(O1,O2):
    return 1/len(O1) * np.einsum('ij,ij->',O1.conjugate(),O2)

def gram_schmidt(h_list):
    V0 = h_list[0]
    V0 = 1/(np.sqrt(inner(V0,V0))) * V0 
    out = [V0]
    for i in range(1,len(h_list)):
        temp = h_list[i]
        temp = 1/(np.sqrt(inner(temp,temp))) * temp 
        
        coeffs = [inner(i,temp) for i in out]
        
        if(np.isclose(np.linalg.norm(coeffs),1)):
            break 
        
        for (c,L) in zip(coeffs,out):
            if(not np.isclose(c,0.0)):
                temp += -c * L 
        
        temp = 1/(np.sqrt(inner(temp,temp))) * temp 
        out.append(temp)
        
    return out 

def gen_DLA(h_list,verbose=True,maxiter=10):
    bases  = [gram_schmidt(h_list)]
    iter = 0
    while True: 
        new_basis = deepcopy(bases[-1])
        prev_basis = bases[-1]
        old_basis = []
        if(len(bases)>=2):
            old_basis  = bases[-2]
        
        for H0 in bases[0]:
            for Hk in prev_basis[len(old_basis):]:
                H  = H0 @ Hk - Hk @ H0
                if(not np.isclose(np.linalg.norm(H),0.0)):
                    H = 1/(np.sqrt(inner(H,H))) * H                     
                    coeffs = [inner(i,H) for i in new_basis]
                    if(not np.isclose(np.linalg.norm(coeffs),1.0)):
                        H -= np.einsum('i,ijk->jk',coeffs,new_basis)
                        H = 1/(np.sqrt(inner(H,H))) * H 
                        new_basis.append(H)
        bases.append(new_basis)
        iter+=1  
        if(iter>maxiter or len(new_basis)>=3000): ###TODO We need to make this more efficient somehow 
            break  
        if(verbose):
            print(f"Iteration: {iter}, Dimension: {len(new_basis)}")
        if(len(new_basis) == len(prev_basis)):
            return new_basis
    return new_basis       
