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
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import networkx as nx

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

# Parallel Variance

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

def all_beta_variance_run(precomp, GW2_circ_data, GW3_circ_data, A, GW2_angles, GW3_angles):
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
        'beta_vars_GW3': beta_vars_GW3,
        "A": A,
        "GW2_angles" : GW2_angles,
        "GW3_angles": GW3_angles
    })
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_runs, f)

def beta_variance_run_loop(iterations=10):
    for _ in tqdm(range(iterations), "Overall Progress"):
        n = 8
        A= rand_adj(n)
        
        Y = GW(A)  ###calculate the full GW embedding
        _,GW2_angles,_ = GW2(A,GW_Y=Y) ###project to 2d angles using precalculated GW embedding 
        _,GW3_angles,_ = GW3(A,GW_Y=Y) ###project to 2d angles using precalculated GW embedding 
        
        ###Get circuit information for each warmstart. Circuit information consists of the initial state + the mixer operators for each qubit
        GW2_circ_data = Q2_data(GW2_angles,rotation = 0)
        GW3_circ_data = Q3_data(GW3_angles,rotation = 0)
    
        precomp  = pre_compute(A) ###compute the Hamiltonian information for the cost layers (shared for all circuits)
    
        all_beta_variance_run(precomp, GW2_circ_data, GW3_circ_data, A, GW2_angles, GW3_angles)

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

def variance_plot(name, vars1, vars2, vars3, ax, initial_p=10, max_p=210, step_size=10):
    p_values = list(range(initial_p, max_p, step_size))
    
    # Plot on the provided axis
    ax.plot(p_values, vars1, 'o-', label='Default')
    ax.plot(p_values, vars2, 's--', label='GW2')
    ax.plot(p_values, vars3, '^-', label='GW3')
    
    ax.set_xlabel("p")
    ax.set_ylabel("Variance (log scale)")
    ax.set_title(name + " Variance vs Depth")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

def all_variance_plot_together(name, runs, initial_p=10, max_p=210, step_size=10):
    plt.figure(figsize=(12, 4))
    a = 0.4

    p_values = list(range(initial_p, max_p, step_size))
    
    for d in runs:
        plt.scatter(p_values, d['beta_vars_default'],color = "b", label='default', alpha=a)
        plt.scatter(p_values, d['beta_vars_GW2'], color = "r", label='GW2', alpha=a)
        plt.scatter(p_values, d['beta_vars_GW3'], color = "g", label='GW3', alpha=a)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    plt.xlabel("Depth")
    plt.ylabel("Variance (log)")
    plt.yscale("log")
    plt.title("All Runs Variance vs Depth")
    plt.show()

def all_variance_plot_separate(name, runs, initial_p=10, max_p=210, step_size=10):
    num_runs = len(runs)
    ncols = 6
    nrows = (num_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for m in range(num_runs):
        vars1 = runs[m]["beta_vars_default"]
        vars2 = runs[m]["beta_vars_GW2"]
        vars3 = runs[m]["beta_vars_GW3"]
        title = f'Run {m+1}'
        variance_plot(title, vars1, vars2, vars3, axes[m], initial_p, max_p, step_size)

    # Hide unused subplots
    for j in range(num_runs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

"""__________________________________________Must Sort Into Categories Later ____________________________________________ """

def new_lst(runs, lst):
    new_lst = []
    for p in lst:
        new_lst.append(runs[p])
    return new_lst

def GW_angles_2d(runs, title):
    num_runs = len(runs)
    ncols = 3
    nrows = (num_runs + ncols - 1) // ncols  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, r in enumerate(runs):
        angles = np.array(r["GW2_angles"])

        # Calculate rotation to place vertex 0 at the top (pi/2)
        rotation = np.pi / 2 - angles[0]
        rotated_angles = (angles + rotation) % (2 * np.pi)

        radius = 1
        x = radius * np.cos(rotated_angles)
        y = radius * np.sin(rotated_angles)

        circle_angles = np.linspace(0, 2 * np.pi, 200)
        circle_x = radius * np.cos(circle_angles)
        circle_y = radius * np.sin(circle_angles)

        ax = axes[i]
        ax.plot(circle_x, circle_y, 'k--', label='Unit Circle')
        ax.plot(x, y, 'o', label='Rotated Points')

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)

        margin = 0.2
        ax.set_xlim(-radius - margin, radius + margin)
        ax.set_ylim(-radius - margin, radius + margin)

    for j in range(len(runs), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title + "\n", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.tight_layout()
    plt.show()

def variance_plot(name, vars1, vars2, vars3, ax, initial_p=10, max_p=210, step_size=10):
    p_values = list(range(initial_p, max_p, step_size))
    
    # Plot on the provided axis
    ax.plot(p_values, vars1, 'o-', label='Default')
    ax.plot(p_values, vars2, 's--', label='GW2')
    ax.plot(p_values, vars3, '^-', label='GW3')
    
    ax.set_xlabel("p")
    ax.set_ylabel("Variance (log scale)")
    ax.set_title(name + " Variance vs Depth")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    
def all_variance_plot_separate(name, runs, initial_p=10, max_p=210, step_size=10):
    num_runs = len(runs)
    ncols = 6
    nrows = (num_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for m in range(num_runs):
        vars1 = runs[m]["beta_vars_default"]
        vars2 = runs[m]["beta_vars_GW2"]
        vars3 = runs[m]["beta_vars_GW3"]
        title = f'Run {m+1}'
        variance_plot(title, vars1, vars2, vars3, axes[m], initial_p, max_p, step_size)

    # Hide unused subplots
    for j in range(num_runs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def GW_angles_3d(runs, title):
    num_runs = len(runs)
    ncols = 3
    nrows = (num_runs + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 6 * nrows))

    for i, r in enumerate(runs):
        angles = r["GW3_angles"]  # Each angle is [polar, azimuthal]
        radius = 1

        # Convert spherical to Cartesian
        cartesian = np.array([
            [
                radius * np.sin(polar) * np.cos(azim),
                radius * np.sin(polar) * np.sin(azim),
                radius * np.cos(polar)
            ]
            for polar, azim in angles
        ])

        # Compute rotation to bring first vertex to north pole
        v0 = cartesian[0]
        v0_norm = v0 / np.linalg.norm(v0)
        north_pole = np.array([0, 0, 1])

        # Axis of rotation = cross product; angle = arccos of dot product
        axis = np.cross(v0_norm, north_pole)
        angle = np.arccos(np.clip(np.dot(v0_norm, north_pole), -1.0, 1.0))

        if np.linalg.norm(axis) < 1e-8:
            rot = R.identity()
        else:
            axis = axis / np.linalg.norm(axis)
            rot = R.from_rotvec(angle * axis)

        rotated = rot.apply(cartesian)

        x, y, z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        ax.scatter(x, y, z, label=f'Run {i+1}', s=40, c='b')

        # Draw a transparent unit sphere
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        xs = radius * np.cos(u) * np.sin(v)
        ys = radius * np.sin(u) * np.sin(v)
        zs = radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='gray', alpha=0.1, edgecolor='none')

        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Run {i+1}')
        ax.grid(True)

    fig.suptitle(title + "\n", fontsize=16)
    plt.tight_layout()
    plt.show()
    
def gw_angles_graph(runs, title):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns
    axes = axes.flatten()  # Flatten to 1D for easy indexing

    for i, r in enumerate(runs):
        adj_matrix = r["A"]
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.circular_layout(G)

        ax = axes[i]
        nx.draw(
            G, pos, ax=ax,
            node_color='lightblue',
            edge_color='gray',
            node_size=500
        )
        ax.set_title(f"Graph {i+1}")

    # Remove unused axes
    for j in range(len(runs), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title + "\n", fontsize=16)
    plt.tight_layout()
    plt.show()

def spherical_to_cartesian(angles, radius=1.0):
    return np.array([
        [
            radius * np.sin(polar) * np.cos(azim),
            radius * np.sin(polar) * np.sin(azim),
            radius * np.cos(polar)
        ]
        for polar, azim in angles
    ])

def coplanar_points(run, tol=1e-2):
    angles = run["GW3_angles"]
    points = spherical_to_cartesian(angles)  # ✅ Convert to 3D
    if points.shape[0] < 4:
        print("YES")
    else:
        ref = points[0]
        vecs = points[1:] - ref
        rank = np.linalg.matrix_rank(vecs, tol=tol)
        print("YES" if rank <= 2 else "NO")

def coplanar_runs(runs, title):
    print(title) 
    for r in runs:
        coplanar_points(r)

