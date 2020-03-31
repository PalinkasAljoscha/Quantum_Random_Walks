import numpy as np

### adjusted function for marked state with extra coin
def run_state_evol(graph:dict, coin_mat:np.array, start_state:np.array , n_steps:int, probsum_tol=1e-5,
                  marked_state=None,mrk_state_coin=None):
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
    print('-'*(n_steps-1))
    for i in range(1,n_steps): # .. einstein sum is used for stacked matrix multiplications     
        transf_state = coin_mat.dot(state_evol[:,:,i-1].T)
        if marked_state is not None: 
            transf_state[:,marked_state] = mrk_state_coin.dot(state_evol[marked_state,:,i-1].T)
        state_evol[:,:,i] = np.einsum('ijk,kj->ik',A, transf_state)
        print('-',end='')   
    # check for consistent prob distribution
    assert(np.abs((1 - (np.linalg.norm(state_evol,ord=2,axis=1)**2).sum(axis=0))).max()<probsum_tol) 
    print(" error in amplitude sum: ",np.abs((1 - (np.linalg.norm(state_evol,ord=2,axis=1)**2).sum(axis=0))).max()) 
    return state_evol

