import numpy as np
import cvxpy as cp
from scipy.stats import ortho_group
import scipy 

def rand_adj(n):
    """
    Returns a random symmetric adjacency matrix drawn uniformly .
    
        Parameters:
            n (int): The size of the matrix.
        Returns:
            np.ndarray: A symmetric (n x n) matrix with zero diagonal
    """
    
    Q= np.random.randint(low=0, high=2, size=(n,n))
    return np.tril(Q,-1)+np.tril(Q,-1).T


def get_cost(A,w):
    """
    Returns the quadratic cost function. This is agnostic as to whether the problem is 0-1 or pm formulated (hence w).

    Parameters:
        A (np.ndarray): A square matrix representing cost coefficients.
        w (np.ndarray): A vector representing decision variables.
    Returns:
        float: Quadratic cost value.
    """
    
    return w.dot(A).dot(w)


def brute_force_maxpm(Q):
    """
    Returns a solved pm QUBO problem using brute force, checks all possible binary solutions (of length n) to find the one that maxamizes the quadratic cost function.
    
    Parameters:
        Q (np.ndarray): A square matrix representing the QUBO problem.
    Returns:
        tuple:
            max_x (np.ndarray): The pm vector (of length n) that maxamizes the cost function.
            max_f (float): The max cost value.
    """
    
    n=len(Q)
    max_y = []
    max_f = -np.inf
    for k in range(2**n):
        x = np.array([i for i in '0'*(n-len(bin(k)[2:]))+bin(k)[2:]],dtype=int)
        y = 2*x-1
        f = get_cost(Q,y)
        if(f>max_f):
            max_y = y
            max_f = f
    return max_y,max_f


def GW(A):
    """
    Returns the approximation to the max-cut problem using the Goemans-Williamson (GW) semidefinite relaxation.
    
    Parameters:
        A: (np.ndarray): A ymmetric adjacency matrix of the graph.
        
    Returns:
        np.ndarray: A set of column vectors representing the graph in a higher dimensional space.
    """
    
    n=len(A)
    M=cp.Variable((n,n),PSD=True)
    constraints = [M >> 0]
    constraints += [
        M[i,i] == 1 for i in range(n)
    ]
    objective = cp.trace(-1/4 * A @ M)
    prob = cp.Problem(cp.Maximize(objective),constraints)
    prob.solve()
    
    L,d,_ = scipy.linalg.ldl(M.value)
    d = np.diag(d).copy()
    d = d*(d>0)
    
    d = np.sqrt(d)
    d.shape = (-1,1)
    return d * L.T



def GW2(A,reps=50,GW_Y=None):
    """
    The projected GW2 relaxation for adjaceny matrix A.
    
    Parameters:
        A (np.ndarray): A symmetric adjacency matrix of the graph.
        reps (int): Number of repititions, default is 50.
        GW_Y (np.ndarray): An initial embedding of the graph, default is GW(A)

    Returns:
        tuple:
            max_Y (np.ndarray): The best 2D embedding of matrix Y.
            np.ndarray: The set of angles representing Y.
            max (float): The maximum computed cost value.    
    """
    
    if(GW_Y is None):
        GW_Y = GW(A)
    max  = -np.inf
    max_Y = None
    for i in range(reps):
        ortho = ortho_group.rvs(len(A))
        basis = ortho.T[:2]
        Y = basis.dot(GW_Y)
        Y=Y/np.linalg.norm(Y,axis=0)
        if(max < np.trace(-1/4 * A @ Y.T.dot(Y))):
            max=np.trace(-1/4 * A @ Y.T.dot(Y))
            max_Y=Y
        max_Y = np.minimum(np.maximum(-1,max_Y),1)
    return max_Y,get_angle(max_Y),max

def GW3(A,reps=50,GW_Y=None):
    """
    The projected GW3 relaxation for adjaceny matrix A.
    
    Parameters:
        A (np.ndarray): A symmetric adjacency matrix of the graph.
        reps (int): Number of repititions, default is 50.
        GW_Y (np.ndarray): An initial embedding of the graph, default is GW(A)

    Returns:
        tuple:
            max_Y (np.ndarray): The best 2D embedding of matrix Y.
            np.ndarray: The set of angles representing Y.
            max (float): The maximum computed cost value.    
    """
    
    if(GW_Y is None):
        GW_Y = GW(A)
    max  = -np.inf
    max_Y = None
    for i in range(reps):
        ortho = ortho_group.rvs(len(A))
        basis = ortho.T[:3]
        Y = basis.dot(GW_Y)
        Y=Y/np.linalg.norm(Y,axis=0)
        if(max < np.trace(-1/4 * A @ Y.T.dot(Y))):
            max=np.trace(-1/4 * A @ Y.T.dot(Y))
            max_Y=Y
        max_Y = np.minimum(np.maximum(-1,max_Y),1)
    return max_Y,get_angle(max_Y), max


def random_round(Y,A,reps=50):
    """
    Generates random hyperplanes and finds the best partitioning.

    Parameters:
        Y (np.ndarray): 
        A (np.ndarray): 
        reps (int): The number of repititions (defaults to 50).

    Returns:
        tuple:
            max_y (np.ndarray):
            max (float): The highest cost achieved by the best hyperplane.
            max_u(np.ndarray):
        
    """
    k = Y.shape[0]
    max_y = None
    max = -np.inf
    max_u = None
    n=len(A)
    for i in range(reps):
        u = 2* np.random.random(k)-1
        y=np.array(np.sign(Y.T @ u),dtype=int)
        l=get_cost(-1/4 * A,y)
        if(l>max):
            max = l
            max_y = y
            max_u=u
    return max_y,max,max_u

def quality(Y,A):
    """
    Compute the quality of the given embedding Y in terms of A.
    
    Parameters:
        Y (np.ndarray): The embedding matrix.
        A (np.ndarray): The adjaceny matrix.

    Returns:
        float: The quality of Y.
    """
    M = Y.T.dot(Y)
    M = np.maximum(-1,np.minimum(1,M))
    return 1/(2*np.pi) * np.trace(A @ np.arccos(M)).real

def fix(theta_list):
    """
    Normalizes the angles to [0, 2pi] for 1D and 2D arrays.

    Paremters:
        theta_list (np.ndarray): A list of angles.

    Returns:
        The normalized/adjusted list of angles.
    """
    if(theta_list.ndim ==1):
        return np.mod(theta_list, 2*np.pi)
    if(theta_list.ndim ==2):
        new_list = []
        for i in range(len(theta_list)):
            theta,phi = theta_list[i]
            theta %= 2*np.pi
            if(theta > np.pi):
                theta = 2*np.pi-theta
                phi = phi + np.pi
            phi %= 2*np.pi
            new_list.append([theta,phi])
        return np.array(new_list)

def get_angle(Y):
    """
    Computes the angles for a 2D or 3D embedding matrix Y.
    
    Parameters:
        Y (np.ndarray): The embedding matrix.

    Returns:
        np.ndarray: A list of angles.
    """
    if(Y.shape[0] == 3):
        theta_list = []
        for i in Y.T:
            theta = np.arccos(i[2])
            phi  = np.arctan2(i[1],i[0]) % (2*np.pi)
            theta_list.append([theta,phi])
        return np.array(theta_list)
    if(Y.shape[0] == 2):
        theta_list = np.array([np.arctan2(*Y.T[i][::-1]) for i in range(len(Y.T))])
        return theta_list % (2*np.pi)

def vertex_on_top(theta_list,rotation = None,z_rot=0):
    """
    

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in polar (2D) or spherical (3D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.
        z_rot (float): Angle for an z-axis rotation, defaults to None.

    Returns:
        np.ndarray: The rotated vertices in spherical coordinates.

    """
    if(rotation is None):
        return theta_list
    else:
        if(theta_list.ndim == 1):
            return fix(theta_list - theta_list[rotation])
        else:
            theta, phi = theta_list[rotation]
            temp_thetas = fix(theta_list - np.stack([np.zeros(len(theta_list)),np.ones(len(theta_list)) * phi]).T)
            Y = np.array([[np.sin(t[0])*np.cos(t[1]),np.sin(t[0])*np.sin(t[1]),np.cos(t[0])] for t in temp_thetas]).T
            R_y = np.array([[np.cos(theta), 0 , -np.sin(theta)],[0,1,0],[np.sin(theta), 0 , np.cos(theta)]])
            Y = np.dot(R_y,Y)
            if z_rot is None:
                mu=np.random.random() * 2 * np.pi
            else:
                mu = z_rot
            Y = np.minimum(np.maximum(-1,Y),1)
            return fix(get_angle(Y) + np.stack([np.zeros(len(theta_list)),np.ones(len(theta_list)) * mu]).T)
        