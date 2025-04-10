from qiskit.quantum_info import SparsePauliOp
from WarmStartUtils import *
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from tqdm.contrib import itertools
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import pickle
import os

"""
Hamiltonian from adjacency matrix A
"""
def indexedZ(i,n):
    """
    Returns a SparsePauli Op corresponding to a Z operator on a single qubit

    Parameters:
        i (int): qubit index 
        n (int): number of qubits

    Returns:
        SparsePauliOp: SparsePauli for single Z operator
    """
    
    return SparsePauliOp("I" * (n-i-1) + "Z" + "I" * i)

def getHamiltonian(A):
    """
    Gets a Hamiltonian from a max-cut adjacency matrix

    Parameters:
        A (np.ndarray): max-cut adjacency matrix

    Returns:
        SparsePauliOp: Hamiltonian 
    """
    
    n = len(A)
    H = 0 * SparsePauliOp("I" * n)
    for i in range(n):
        for j in range(n):
            H -= 1/4 * A[i][j] * indexedZ(i,n) @ indexedZ(j,n)
    return H.simplify()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SU2_op(x,y,z,t):
    """
    Get the matrix for a SU2 rotation around an axis 

    Parameters:
        x (float): x-coordinate of rotation axis
        y (float): y-coordinate of rotation axis
        z (float): z-coordinate of rotation axis
        t (float): rotation angle

    Returns:
        np.ndarray: matrix for the SU2 rotation
    """
    
    return np.array([[np.cos(t)-1j * np.sin(t)*z, -np.sin(t) * (y+1j * x)],[ -np.sin(t) * (-y+1j * x),np.cos(t)+1j * np.sin(t)*z]])

def apply_single_qubit_op(psi,U,q):
    """
    Efficiently apply a single qubit operator U on qubit q to statevector psi

    Parameters:
        psi (np.ndarray): Statevector
        U (np.ndarray): Operator 
        q (int): qubit index 

    Returns:
        np.ndarray: New Statevector 
    """
    
    n=int(np.log2(len(psi)))
    axes = [q] + [i for i in range(n) if i != q]
    contract_shape = (2, len(psi)// 2)
    tensor = np.transpose(
        np.reshape(psi, tuple(2 for i in range(n))),
        axes,
    )
    tensor_shape = tensor.shape
    tensor = np.reshape(
        np.dot(U, np.reshape(tensor, contract_shape)),
        tensor_shape,
    )
    return np.reshape(np.transpose(tensor, np.argsort(axes)),len(psi))

def pre_compute(A):
    """
    Pre-compute the diagonal elements of Hamiltonian corresponding to max-cut adjacency matrix

    Parameters:
        A (np.ndarray): adjacency matrix

    Returns:
        np.ndarray: Array containing matrix diagonal  
    """
    
    return np.array(scipy.sparse.csr_matrix.diagonal(getHamiltonian(np.flip(A)).to_matrix(sparse=True))).real

def apply_mixer(psi,U_ops):
    """
    Apply mixer layer to state
    
    Parameters:
        psi (np.ndarray): Original state vector
        U_op (list[np.ndarray]): list of mixer operators

    Returns:
        psi (np.ndarray): New statevector  
    """
    
    for n in range(0,len(U_ops)):
        psi = apply_single_qubit_op(psi, U_ops[n], n)
    return psi

def cost_layer(precomp,psi,t):
    """
    Given a precomputed Hamiltonian diagonal, apply the cost layer

    Parameters:
        precomp (np.ndarray): Precompute diagonal
        psi (np.ndarray): Statevector
        t (float): Rotation angle

    Returns:
        np.ndarray: New statevector
    """
    
    return np.exp(-1j * precomp*t) * psi

def QAOA_eval(precomp,params,mixer_ops=None,init=None):
    """"
    Returns statevector after applying QAOA circuit

    Parameters:
        precomp (np.ndarray): Hamiltonian diagonal
        params (np.ndarray): Array of QAOA circuit parameters
        mixer_ops (list[np.ndarray]): list of mixer parameters
        init (np.ndarray): initial state 

    Returns:
        psi (np.ndarray): new statevector
    """
    
    p = len(params)//2
    gammas = params[p:]
    betas = params[:p]
    
    psi = np.zeros(len(precomp),dtype='complex128')
    if(init is None):
        psi = np.ones(len(psi),dtype='complex128') *  1/np.sqrt(len(psi))
    else:
        psi = init

    if(mixer_ops is None):
        mixer = lambda t: [SU2_op(1,0,0,t) for m in range(int(np.log2(len(psi))))]
    else:
        mixer = mixer_ops
    
    for i in range(p):
        psi = cost_layer(precomp,psi,gammas[i])
        psi = apply_mixer(psi,mixer(betas[i]))
    return psi

def expval(precomp,psi):
    """"
    Compute the expectation value of a diagonal hamiltonian on a state

    Parameters:
        precomp (np.ndarray): Diagonal elements of Hamiltonian 
        psi (np.ndarray): Statevector 

    Returns:
        float: expectation value 
    """
    
    return np.sum(psi.conjugate() * precomp * psi).real

def Q2_data(theta_list,rotation=None):
    """"
    Get warmstart data from polar angles

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in polar (2D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.

    Returns:
        tuple:
            init (np.ndarray): The initial statevector
            mixer_ops (list[np.ndarray]): the mixer operators 
    """
    
    angles = vertex_on_top(theta_list,rotation)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v/2), np.exp(-1j/2 * np.pi)*np.sin(v/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t: [  SU2_op(0,-np.sin(v),np.cos(v),t) for v in angles]
    return init,mixer_ops

def Q3_data(theta_list,rotation=None,z_rot=0):
    """"
    Get warmstart data from spherical angles

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in spherical (3D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.

    Returns:
        tuple:
            init (np.ndarray): The initial statevector
            mixer_ops (list[np.ndarray]): the mixer operators 
    """
    angles = vertex_on_top(theta_list,rotation,z_rot=z_rot)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v[0]/2), np.exp(1j * v[1])*np.sin(v[0]/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t:  [SU2_op(np.sin(v[0])*np.cos(v[1]),np.sin(v[0])*np.sin(v[1]),np.cos(v[0]),t) for v in angles]
    return init,mixer_ops

X = np.array([[0,1],[1,0]],dtype=complex)
Y = np.array([[0,-1j],[1j,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)

def Q2_Hamiltonian(angles):
    
    GW2_HB = np.zeros((2**len(angles),2**len(angles)),dtype=complex)
    for i,t in enumerate(angles):
        op = -np.sin(t) * Y + np.cos(t)* Z
        l = [np.eye(2) for i in range(len(angles))]
        l[i] = op 
        GW2_HB+=reduce(lambda a,b: np.kron(a,b),l)
    
    return GW2_HB

def Q3_Hamiltonian(angles):
    
    GW3_HB = np.zeros((2**len(angles),2**len(angles)),dtype=complex)
    for i,t in enumerate(angles):
        op = np.sin(t[0])*np.cos(t[1]) * X + np.sin(t[0])*np.sin(t[1]) * Y  + np.cos(t[0]) * Z
        l = [np.eye(2) for i in range(len(angles))]
        l[i] = op 
        GW3_HB+=reduce(lambda a,b: np.kron(a,b),l)
    
    return GW3_HB

def default_Hamiltonian(n):
    
    HB = np.zeros((2**(n),2**(n)),dtype=complex)
    for i in range(n):
        op = X
        l = [np.eye(2) for i in range(n)]
        l[i] = op 
        HB+=reduce(lambda a,b: np.kron(a,b),l)
    
    return HB

def mixer_derivative(mixer_ops):
    out  = []
    def derive(t):
        m_plus  = mixer_ops(t+np.pi/2)
        m_minus  = mixer_ops(t-np.pi/2)
        return [(p-m)/2 for (p,m) in zip(m_plus,m_minus)]
    for i in range(len(mixer_ops(0))):
        out.append(lambda t,idx=i: [mixer_ops(t)[j] if j!=idx else derive(t)[idx] for j in range(len(mixer_ops(0)))]) 
        
    return out

def beta_QAOA_state_derivative(idx,precomp,params,mixer_ops=None,init=None):
    
    p = len(params)//2
    gammas = params[p:]
    betas = params[:p]
    
    psi = np.zeros(len(precomp),dtype='complex128')
    if(init is None):
        psi = np.ones(len(psi),dtype='complex128') *  1/np.sqrt(len(psi))
    else:
        psi = init


    if(mixer_ops is None):
        mixer = lambda t: [SU2_op(1,0,0,t) for m in range(int(np.log2(len(psi))))]
    else:
        mixer = mixer_ops
        
    
    d_psi = np.array([psi for i in range(len(mixer(0)))])
    
    mixer_derivative_ops = mixer_derivative(mixer)
    
    for i in range(p):
        for _ in range(len(d_psi)):
            d_psi[_] = cost_layer(precomp,d_psi[_],gammas[i])
        if(i!=idx):
            for _ in range(len(d_psi)):
                d_psi[_] = apply_mixer(d_psi[_],mixer(betas[i]))
        else:
            for _ in range(len(d_psi)):
                d_psi[_] = apply_mixer(d_psi[_],mixer_derivative_ops[_](betas[i]))
    return np.sum(d_psi,axis=0)



def gamma_QAOA_state_derivative(idx,precomp,params,mixer_ops=None,init=None):
    
    p = len(params)//2
    gammas = params[p:]
    betas = params[:p]
    
    psi = np.zeros(len(precomp),dtype='complex128')
    if(init is None):
        psi = np.ones(len(psi),dtype='complex128') *  1/np.sqrt(len(psi))
    else:
        psi = init


    if(mixer_ops is None):
        mixer = lambda t: [SU2_op(1,0,0,t) for m in range(int(np.log2(len(psi))))]
    else:
        mixer = mixer_ops
        

    for i in range(p):
        psi = cost_layer(precomp,psi,gammas[i])
        if(i == idx):
            psi = (-1j * precomp) * psi
        psi = apply_mixer(psi,mixer(betas[i]))
    return psi

def QAOA_gradient_eval(precomp,params,mixer_ops=None,init=None):
    psi = QAOA_eval(precomp,params,mixer_ops=mixer_ops,init=init)
    grads = []
    for i in range(len(params)//2):
        dpsi = beta_QAOA_state_derivative(i,precomp,params,mixer_ops=mixer_ops,init=init)
        grads.append(2 * np.sum(dpsi.conjugate() * precomp * psi).real)
    
    for i in range(len(params)//2):
        dpsi = gamma_QAOA_state_derivative(i,precomp,params,mixer_ops=mixer_ops,init=init)
        grads.append(2 * np.sum(dpsi.conjugate() * precomp * psi).real)
        
    return np.array(grads)

# def variance_sample(f,n_params,shots=100):
#     data = []
#     for i in tqdm(range(shots)):
#         data.append((f(np.random.random(n_params) * 2 * np.pi)))
#     return np.mean(np.var(data,axis=0))

# Paralell Variance

def variance_sample(f, n_params, shots=100, n_jobs=-1, tqdm_enabled=False):
    param_list = [np.random.random(n_params) * 2 * np.pi for _ in range(shots)]
    
    if tqdm_enabled:
        with tqdm_joblib(total=shots):
            results = Parallel(n_jobs=n_jobs)(
                delayed(f)(params) for params in param_list
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(params) for params in param_list
        )
    
    return np.mean(np.var(results, axis=0))

def QAOA_variance_sample(precomp,p,mixer_ops=None,init=None,shots=100):
    f = lambda params : QAOA_gradient_eval(precomp,params,mixer_ops=mixer_ops,init=init)
    return variance_sample(f,2*p,shots=shots)    

def beta_variance_sample(precomp,p,mixer_ops=None,init=None,shots=100):
    f = lambda params : beta_QAOA_state_derivative(p//2,precomp,params,mixer_ops=mixer_ops,init=init)
    psi = lambda params: QAOA_eval(precomp,params,mixer_ops=mixer_ops,init=init)
    grad = lambda params: 2 * np.sum(f(params).conjugate() * precomp * psi(params)).real
    return variance_sample(grad, 2*p,shots=shots)

def all_beta_variance_run(precomp, GW2_circ_data, GW3_circ_data):
    pickle_file = 'beta_variance_runs.pkl'
    
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            all_runs = pickle.load(f)
    else:
        all_runs = []
    
    beta_vars_default = variance_depth(beta_variance_sample, precomp, shots=1000)
    beta_vars_GW2 = variance_depth(beta_variance_sample, precomp, mixer_ops=GW2_circ_data[1], init=GW2_circ_data[0], shots=1000)
    beta_vars_GW3 = variance_depth(beta_variance_sample, precomp, mixer_ops=GW3_circ_data[1], init=GW3_circ_data[0], shots=1000)
    
    all_runs.append({
        'beta_vars_default': beta_vars_default,
        'beta_vars_GW2': beta_vars_GW2,
        'beta_vars_GW3': beta_vars_GW3
    })
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_runs, f)

def expm(A):
    vals,vecs  = np.linalg.eig(A)
    return vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)

def variance_depth(variance_sample, precomp, initial_p=10, max_p=210, step_size=10,mixer_ops=None, init=None, shots=100):
    vars = []
    for i in tqdm(range(initial_p, max_p, step_size), "Individual Progress"):
        p = i
        var = variance_sample(precomp,p,mixer_ops,init,shots)
        vars.append(var)
    return vars

def variance_plot(name, vars, initial_p=10, max_p=210, step_size=10):
    p_values = list(range(initial_p, max_p, step_size))

    plt.figure(figsize=(10, 4))
    plt.plot(p_values, vars, marker='o', linestyle='None')
    plt.xlabel("p")
    plt.ylabel("Variance (log scale)")
    plt.title(name + " Variance vs Depth")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()