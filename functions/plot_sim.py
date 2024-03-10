import numpy as np
import matplotlib.pyplot as plt

#%% Plot results
def plot_model(environment, measurements, predictions, action_pos, goal, num_steps):
    time = np.arange(num_steps)*environment.dt

    # Get position (s1) variables    
    true_pos = np.array(environment.true_states)[:,0]
    measured_pos = np.array(measurements)[:,0]
    measured_goal_pos = np.array(measurements)[:,1]
    predicted_pos = np.array(predictions)[:,0]  
    control_signal = np.array(action_pos)
    f_pos = np.array(predictions)[:,2]
    g_pos = np.array(predictions)[:,3]

    # Get velocity (s2) variables
    true_vel = np.array(environment.true_states)[:,1]
    predicted_vel = np.array(predictions)[:,1]  
    f_vel = np.diff(f_pos)/environment.dt
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5,5), sharex =True, sharey=False)

    ax[0,0].plot(time, goal, c='blue', linewidth=1)
    ax[0,0].scatter(time, measured_goal_pos, c='blue', alpha=0.3)
    ax[0,0].scatter(time, measured_pos, label='Measurements', color='orange',alpha=0.3)
    ax[0,0].plot(time, predicted_pos, label='KF Prediction', color='r', linewidth=2.5)
    ax[0,0].plot(time, true_pos, label='True Position', color='orange', linewidth=1)
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Position (arb. units)')
    ax[0,0].legend(loc='lower right', prop={'size': 8})
    ax[0,0].set_ylim([-25, 100])

    ax[1,0].plot(time, goal, c='blue', linewidth=1)
    ax[1,0].scatter(time, measured_goal_pos, c='blue', alpha=0.3)
    ax[1,0].plot(time, f_pos, label='2nd order (force)', color='black',linewidth=3.5)
    ax[1,0].plot(time, g_pos, label='1st order', color='grey',linewidth=1.5)
    ax[1,0].plot(time, goal, c='blue', linewidth=1)
    ax[1,0].plot(time, control_signal, label='Control signal', color='g',linewidth=0.5)
    ax[1,0].set_xlabel('Time')
    ax[1,0].set_ylabel('Position (arb. units)')
    ax[1,0].legend(loc='lower right', prop={'size': 8})
    ax[1,0].set_ylim([-25, 100])
    
    ax[0,1].axhline(0, c='blue', linewidth=1)
    ax[0,1].plot(time, predicted_vel, label='KF Prediction', color='r', linewidth=2.5)
    ax[0,1].plot(time, true_vel, label='True Velocity', color='orange', linewidth=1)
    ax[0,1].set_xlabel('Time')
    ax[0,1].set_ylabel('Velocity (arb. units)')
    ax[0,1].legend(loc='upper right', prop={'size': 8})
    ax[0,1].set_ylim([-25, 100])

    ax[1,1].axhline(0, c='blue', linewidth=1)
    ax[1,1].plot(time[:-1], f_vel, label='2nd order', color='black',linewidth=3.5)
    # ax[1,1].plot(time[:-1], g_vel, label='1st order', color='grey',linewidth=1.5)
    # ax[1,1].plot(time[:-1], control_signal_vel, label='Control velocity', color='g',linewidth=0.5)
    ax[1,1].set_xlabel('Time')
    ax[1,1].set_ylabel('Velocity (arb. units)')
    ax[1,1].legend(loc='upper right', prop={'size': 8})
    ax[1,1].set_ylim([-100,200])
    # plt.tight_layout()
    return fig