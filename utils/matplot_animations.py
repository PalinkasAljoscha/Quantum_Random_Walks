

import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def classic_animation( gif_path, transition, projection, start_pd, n_steps=10, show_equil=False, frame_option='pairsmean'):
    fig = plt.figure(figsize=(15,8))
    n = transition.shape[0]

    if projection is None:
        P = np.diag(np.ones(n))
    else:
        P = projection
    P = csr_matrix(P)
    d = P.shape[0]
    
    if frame_option=='pairsmean':
        pos_offset = 0
        left_border = 0
        right_border = d
        n_frames = n_steps-1
        pos_select_stepsize = 1
    elif  frame_option=='even':
        pos_offset = ((n//2)%2) # if n//2 is odd, border nodes have to be cut, since they have zero prob in the plotted time points
        right_border = n//2 - pos_offset
        left_border = -right_border
        n_frames = n_steps//2
        pos_select_stepsize = 2
    ax = plt.axes(xlim=(left_border,right_border),ylim=(0,0.5))
    #plt.xticks(np.arange(0,n,n//10))
    plt.grid()
    line, = ax.plot([], [], color='cornflowerblue', lw=3)
    if show_equil:
        line_equil, = ax.plot([], [], color='tomato', lw=1)
        equil_state = P.dot((1/n)*np.ones(n))

    
    def init():
        global T0
        T0 = csr_matrix(np.diag(np.ones(n)))
        line.set_data([], [])
        if show_equil:
            line_equil.set_data([], [])
        return line,
    def animate(i): 
        global T0
        if frame_option=='pairsmean':
            x = np.arange(d)   #np.arange(-n,n+1
            T1 = T0*transition
            y = 0.5*P*(T0*start_pd + T1*start_pd)
        elif  frame_option=='even':
            x = np.arange(left_border,n//2+1,2)   #np.arange(-n,n+1)
            T1 = T0*transition*transition
            y = (P*(T0*start_pd))[pos_offset::pos_select_stepsize]  
            #print(x.shape,y.shape)
        T0 = T1
        line.set_data(x, y)
        if show_equil:
            line_equil.set_data(x, equil_state[pos_offset::pos_select_stepsize])
        return line,
    # divide n_steps by two, since it is multiplied by two in each animation step to get only even time points
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=450, blit=True)
    anim.save(gif_path+'.gif', writer='imagemagick')
    return None
    
    
# animation for quantum walk
def animate_probs_evol( gif_path, state_evol, projection, show_equil=False,show_cesaromean=True,
                       scale_projection=False, frame_option='pairsmean'):
    P = projection
    if scale_projection:
        P = (np.diag(1/(P.toarray().sum(axis=1)))*P)
        
    
    probs_evol = np.linalg.norm(state_evol,ord=2,axis=1)**2
    
    fig = plt.figure(figsize=(15,8))
    ax = plt.axes(
        xlim=(  # find left most and right most location that is occuring 
            np.where(P.dot((state_evol[:,:,:]!=0).sum(axis=2).sum(axis=1)))[0].min(),
            np.where(P.dot((state_evol[:,:,:]!=0).sum(axis=2).sum(axis=1)))[0].max()
        ),
        ylim=(0,np.quantile(np.max(P.dot(probs_evol),axis=0),0.99))
    )
    plt.grid()
    line_avgp, = ax.plot([], [],color='gold', lw=3.5)  # cesaro limit of probability distributions
    line_p, = ax.plot([], [],color='cornflowerblue', lw=1)  # probability distributions

    if show_equil:
        line_equil, = ax.plot([], [], color='tomato', lw=1)
        equil_state = P.dot((1/P.shape[1])*np.ones(P.shape[1]))
        
    def init():
        line_avgp.set_data([], [])
        line_p.set_data([], [])
        return line_avgp,

    def animate(i):
        x = np.arange(P.shape[0])
        y = 0.5*(probs_evol[:,i] + probs_evol[:,i+1])
        y_avg = probs_evol[:,:max(2,i+1)].mean(axis=1)
        # apply projection
        y = P.dot(y)
        y_avg = P.dot(y_avg)
        if show_cesaromean:
            line_avgp.set_data(x, y_avg)
        line_p.set_data(x, y)
        if show_equil:
            line_equil.set_data(x, equil_state)
            
        return line_p,

    anim = FuncAnimation(fig, animate, init_func=init, frames=state_evol.shape[2]-2, interval=220, blit=True)

    #anim = FuncAnimation(fig, animate, frames=40, interval=400, repeat=True, repeat_delay= 1000, blit=True)

    anim.save(gif_path+'.gif', writer='imagemagick')
    return None
    