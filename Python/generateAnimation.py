import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generateAnimation(jointPositions_opt, dt, strideLength, nCycles=7):
    
    P = jointPositions_opt
    
    Px1_t = np.zeros((P[0].shape[0], 4))
    Px1_t[:,1] = P[0]
    Px1_t[:,2] = P[2]
    Px1_t[:,3] = P[4]
    
    Py1 = np.zeros((P[0].shape[0], 4))
    Py1[:,1] = P[1]
    Py1[:,2] = P[3]
    Py1[:,3] = P[5]   
    
    Px2_t = np.zeros((P[0].shape[0], 3))
    Px2_t[:,0] = P[2]
    Px2_t[:,1] = P[6]
    Px2_t[:,2] = P[8]
    
    Py2 = np.zeros((P[0].shape[0], 3))
    Py2[:,0] = P[3]
    Py2[:,1] = P[7]
    Py2[:,2] = P[9]
    
    Px1 = np.copy(Px1_t)
    Px2 = np.copy(Px2_t)
    
    for nCycle in range(1, nCycles):
        Px1 = np.concatenate((Px1, Px1_t + nCycle*strideLength), axis=0)
        Px2 = np.concatenate((Px2, Px2_t + nCycle*strideLength), axis=0)
        Py1 = np.concatenate((Py1, Py1), axis=0)
        Py2 = np.concatenate((Py2, Py2), axis=0)        
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, 
                         xlim=(-0.6, 3.6), ylim=(-0.1, 1.7))
    
    line, = ax.plot([], [], 'o-', lw=2, color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, color='blue')
    ax.set_xlabel('(m)')
    ax.set_ylabel('(m)')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)    
    
    def init():
        line.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line, line2, time_text
    
    
    def animate(i):
        thisx = Px1[i, :]
        thisy = Py1[i, :]
        
        thisx2 = Px2[i, :]
        thisy2 = Py2[i, :]
    
        line.set_data(thisx, thisy)
        line2.set_data(thisx2, thisy2)
        time_text.set_text(time_template % (i*dt))
        return line, line2, time_text
    
    anim = animation.FuncAnimation(fig, animate, P[0].shape[0]*nCycles,
                            interval=int(1000/(1/dt)), blit=True, 
                            init_func=init)
    plt.draw()
    
    return anim   
