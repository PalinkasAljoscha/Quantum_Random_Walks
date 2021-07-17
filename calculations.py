import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TABLEAU_COLORS


def classic_walk(n, go_left_prob, circle=True):
    """
    create transition matrix and initial state for classic random walk
    args: 
    n: number of points on line, use even number to get nice animations, midpoint is adapted to be even point
    this is done in light of frame_option='even' in animation methods, see below
    circle: if True will glue end points together to obatin circle graph
    """
    go_right_prob = 1 - go_left_prob 
    # prepare transition matrix
    T = np.diag(go_left_prob*np.ones(n-1),k=-1) + np.diag(go_right_prob*np.ones(n-1),k=1)
    # for circle, glue opposite border nodes by adding an edge between
    if circle:
        T = T + np.diag(go_left_prob*np.ones(1),k=(n-1)) + np.diag(go_right_prob*np.ones(1),k=-(n-1))
    #create start state with all probability in one mid point
    state_0 = np.zeros(n)
    state_0[n//2 - (n//2)%2] = 1 
    return T, state_0

def anim_classic_walk(transition, projection, start_pd, n_steps=10, show_equil=False, frame_option='pairsmean'):
    
    """
    calculate the evolution of random walk and prepare dataframe df of the evolutionvof probability distribution
    the probability distribution at start is given as input start_pd. i
    in each loop through the n_steps frames for the animation, the transition matrix in applied
    
    projection: matrix would allow projection of whole graph for animation, not really useful for circle 
    show_equil: equilibrium is the uniform distribution, which can be plotted so that conversion can be seen   
    frame_option: three options of animation frames are possible:
    plain: notrecommended, just plots in each frame the actual probability distribution of that time step of the walk
           this gives a jumpy animation, as in every second step any fixed node has probability zero, 
    therefore two variants are recommended:
    pairsmean: show in each animation frame the average of two consecutive evolution steps of probability distribtions
                (except first step, to show true start state)
    even: like a zoom out, it shows only even numbered nodes and only even time steps, 
          this gives the smoothest animation, but might be losing some detail
          Note: this is only recommended for even number of nodes, otherwise even nodes become uneven nodes 
          as they pass the glued border and parts of dynamic are then invisible in the animation
    """
    
    T = csr_matrix(transition)
    n = T.shape[0]
        
    if projection is None:
        P = np.diag(np.ones(n))
    else:
        P = projection
    P = csr_matrix(P)
    n_proj = P.shape[0]
    
    if frame_option in ['pairsmean', 'plain']:
        n_frames = n_steps-1
        pos_select_stepsize = 1
    elif  frame_option=='even':
        if (n%2)!=0: 
            print("NOTE: animation with frame_option='even' and uneven number "
                  +"of nodes not recommended (part of dynamic will be hidden)")
        n_frames = n_steps//2
        pos_select_stepsize = 2
        T = T*T
        
    # create anim df
    T0 = csr_matrix(np.diag(np.ones(n))) 
    T1 = T0
    equi_y = P.dot((1/n)*np.ones(n))*pos_select_stepsize
    
    df = pd.DataFrame({'x': [], 'y': []})
    for i in range(n_frames):
        T1 = T0*T
        if frame_option=='pairsmean':
            x = np.arange(n_proj) 
            if i == 0: 
                y = P*(T0*start_pd)
            else:
                y = 0.5*P*(T0*start_pd + T1*start_pd)
        elif  frame_option=='even':
            x = np.arange(0,n_proj+1,2)[:n_proj//2+n_proj%2]   
            y = (P*(T0*start_pd))[::pos_select_stepsize] 
        elif  frame_option=="plain":
            x = np.arange(n_proj) 
            y = P*(T0*start_pd) 
        T0 = T1
            
        df = df.append(pd.DataFrame({'x': x, 'y': y, 'frame': i, 'info': 'evol'}))
        if show_equil:            
            df = df.append(pd.DataFrame({'x': x, 'y': equi_y[::pos_select_stepsize], 'frame': i, 'info': 'equi'}))        
    
    return df


### adjusted function for marked state with extra coin
def run_state_evol(graph:dict, coin_mat:np.array, start_state:np.array , n_steps:int, probsum_tol=1e-5,
                  marked_state=None,mrk_state_coin=None, verbose=False):
    """
    run number of steps of quantum random walk on given graph with edge labelling,
    which related edges to dimensions in the coin space
    
    INPUT: graph = {'edges': (n,2) array , 'fw_edge_labels': (n,) array ,'bw_edge_labels': (n,) array}
    each edge has a label for forward and one for backward direction, the edge array defines what is fw
    edge labels refer to dimensions of the coin space and determine which coinspace
    coefficients are passed along the edge to neighbor node
    coin_mat= (d,d) array, the coin space operator applied before each step on the coin space at each node
    start_state= (n,d) array defines the start state of the coin space - graph - tensor product  
    a marked state can be passed with separate coin
    OUT: stacked states for each time step
    """
    n,d = start_state.shape 
    # get stacked transition operators on all dimensions of coin space
    A = np.zeros((n,n,d)) 
    A[graph['edges'][:,0],graph['edges'][:,1],graph['fw_edge_labels']] = 1
    A[graph['edges'][:,1],graph['edges'][:,0],graph['bw_edge_labels']] = 1
    # initialise all zero state evolution and set start state at index time 0
    state_evol = np.zeros((n,d,n_steps)).astype(np.complex64)   
    state_evol[:,:,0] = start_state
    # run through all steps, update coin space and then apply transition operator
    if verbose: print('-'*(n_steps-1))
    for i in range(1,n_steps): # .. einstein sum is used for stacked matrix multiplications     
        transf_state = coin_mat.dot(state_evol[:,:,i-1].T)
        if marked_state is not None: 
            transf_state[:,marked_state] = mrk_state_coin.dot(state_evol[marked_state,:,i-1].T)
        state_evol[:,:,i] = np.einsum('ijk,kj->ik', A, transf_state)
        if verbose: print('-',end='')   
    # check for consistent prob distribution
    assert(np.abs((1 - (np.linalg.norm(state_evol,ord=2,axis=1)**2).sum(axis=0))).max()<probsum_tol) 
    if verbose: print("\nerror in amplitude sum: ",np.abs((1 - (np.linalg.norm(state_evol,ord=2,axis=1)**2).sum(axis=0))).max()) 
    return state_evol


def quantum_walk_circle(n):
    d = 2
    # this defines a circle graph with 2 dimensional direction space, one for each direction on the circle
    graph = { 
        'edges': np.column_stack((np.arange(0,n),np.arange(1,n+1)%n)).astype(int),
        'fw_edge_labels': 0 * np.ones(n).astype(int) ,
        'bw_edge_labels': 1 * np.ones(n).astype(int)
    }
    # define the state at time 0
    start_state = np.zeros((n,d)).astype(np.complex64)
    start_state[n//2,:] = np.array([1,1])*(1/np.linalg.norm(np.array([1,1])))
    return graph, start_state


def anim_quantum_walk(probs_evol, projection, show_equil=False,show_cesaromean=True,
                       scale_projection=False, frame_option='pairsmean'):
    """
    animation for quantum random walk 
    similar to anim_classic_walk above, only difference is that the probability evolution not calculated but an input
    the cesaro mean (which is just the mean of vurrent evolution at every step) is additionally calculated as y_avg
    """
    n = probs_evol.shape[0]
    
    if projection is None:
        P = np.diag(np.ones(n))
    elif scale_projection:
        P = projection
        P = (np.diag(1/(P.sum(axis=1)))*P)
    else:
        P = projection
    probs_evol = P.dot(probs_evol)
    n_proj = P.shape[0]
    
    equil_state = P.dot((1/P.shape[1])*np.ones(P.shape[1]))
    
    if frame_option in ['pairsmean', 'plain']:
        n_frames = probs_evol.shape[1]-1
        pos_select_stepsize = 1
    elif  frame_option=='even':
        if (n%2)!=0: 
            print("NOTE: animation with frame_option='even' and uneven number "
                  +"of nodes is not recommended (part of dynamic will be hidden)")
        n_frames = probs_evol.shape[1]//2
        probs_evol = probs_evol[::2,:]
        equil_state = equil_state[::2]
        
    df = pd.DataFrame({'x': [], 'y': []})
    for i in range(n_frames):
        if frame_option=='pairsmean':
            x = np.arange(n_proj)
            y = 0.5*(probs_evol[:,max(i-1,0)] + probs_evol[:,i])
            y_avg = probs_evol[:,:max(1,i+1)].mean(axis=1)
        elif  frame_option=='even':
            x = np.arange(0,n_proj+1,2)[:n_proj//2+n_proj%2]
            y = probs_evol[:,2*i]
            y_avg = probs_evol[:,:max(2,2*i+1)].mean(axis=1)
        elif  frame_option=="plain":  
            x = np.arange(n_proj)
            y = probs_evol[:,i]
            y_avg = y_avg = probs_evol[:,:max(2,i+1)].mean(axis=1)   
    
        df = df.append(pd.DataFrame({'x': x, 'y': y, 'frame': i, 'info': 'Probability'}))
        if show_cesaromean:
            df = df.append(pd.DataFrame({'x': x, 'y': y_avg, 
                                         'frame': i, 'info': 'Cesaro mean'}))    
        if show_equil:
            df = df.append(pd.DataFrame({'x': x, 'y': equil_state, 
                                         'frame': i, 'info': 'Equilibrium'}))

    return df