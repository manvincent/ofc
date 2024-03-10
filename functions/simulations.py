import numpy as np
#%% Import backend
from functions.estimator import LQE
from functions.controller import LQR
from functions.environment import MassSpring, simple_simulation

#%% Define parameters
def simulate_human(target, k, trajectory, dt): 
    # Empirical parameters
    mass = 0.5 # Mass constant 

    # Get time steps
    num_steps = len(trajectory)
    
    # Set up matrices
    transition_matrix = np.array([[1, dt, 0],
                                [dt*-k/mass, 1, 0],
                                [0, 0, 1]]) 

    # Assume 0 process noise following Todorov and Jordan 
    transition_covariance = np.zeros((3,3))

    # Assumes both pos and vel are observed
    observation_matrix = np.array([[1,0,0],
                                [0,0,1]]
                                )
                                
    # From Todorov & Jordan, in m/s units 
    observation_covariance = np.array([[0.00001,0],
                                    [0,0.00001]])

    control_dynamics = np.array([[0], 
                                [dt/mass],
                                [0]])
                                
    # Cost matrices
                        
    # Cost vector for distance from goal
    q = np.array([1,0,-1])
    # Cost vector for object velocity
    v = np.array([0, 0.2, 0])
    # Cost vector for hand velocity
    w = np.array([0, 0, 0])

    # Add to form cost matrix                                
    state_cost_matrix = np.outer(q,q) + np.outer(v,v) 
    
    # Initilaize computational objects 
    # create LQE object
    kf = LQE(dt, transition_matrix, transition_covariance, 
            control_dynamics, 
            observation_matrix, observation_covariance)

    
    environment = MassSpring(dt, transition_matrix, control_dynamics, 
                    observation_matrix, observation_covariance)

    # Simulate a trial
    true_state = [] 
    measurements = [] 
    predictions = [] 
    state_cost_array = []

    goal = np.ones(num_steps) * target
    # goal = np.sin(np.linspace(0, np.pi*3, num_steps)) * 50

    state = np.array([0,0, goal[0]])

    #  Iterate
    for t  in range(num_steps):    
        if t == 0: 
            environment.initialize_state(state)
            kf.initialize_state(mu_init = state, 
                                sigma_init = np.eye(len(state)))
        # Get measurement from system 
        m = environment.step_measurement()
        measurements.append(m)
            
        # Make a prediction about the state given measurement
        # Incorporate measurement
        kf.project(m)
        # # Update to new posterior
        state = kf.update()
        
        # Make the new state have the time-varying goal
        state[-1] = m[1]#goal[t]
        
        predictions.append(state)    
                    
        # Exert control from data
        force = trajectory[t]*100
        
        state_cost = state.T @ state_cost_matrix @ state        
        state_cost_array.append(state_cost)
        
        
        # Update to next iteration's prior
        kf.predict(force) 
        
        # Simulate system
        environment.step_system(force)    
    return environment, measurements, predictions, goal, num_steps, state_cost_array
#%%     
def simulate_model(target = 70, k = 1.0): 
    # Empirical parameters
    dt = 0.0167 # Corresponds to 16.7 msec or 60 Hz 
    mass = 0.5 # Mass constant 

    # Get time steps
    max_RT = 3.5 # in sec
    num_steps = int(np.ceil(max_RT  /  dt) + 1)
    
    # Set up matrices
    muscle_filter_rate = 0.04 # In second
    tau1 = tau2 = dt / muscle_filter_rate
    transition_matrix = np.array([[1, dt, 0, 0, 0],
                                [dt*-k/mass, 1,dt/mass, 0, 0],
                                [0, 0, 1-dt/tau2, dt/tau2, 0],
                                [0, 0, 0, 1-dt/tau1, 0],
                                [0, 0, 0, 0, 1]]) 

    # Assume 0 process noise following Todorov and Jordan 
    transition_covariance = np.zeros((5,5))

    # Assumes both pos and vel are observed
    observation_matrix = np.array([[1,0,0, 0,0],
                                [0,0,0,0,1]]
                                )
                                
    # From Todorov & Jordan, in m/s units 
    observation_covariance = np.array([[3.33,0],
                                    [0,.1]])

    control_dynamics = np.array([[0], 
                                [0],
                                [0],
                                [dt/tau1],
                                [0]])

    # Cost matrices
    action_cost_matrix = 0.05 # Scalar cost of single action
                        
    # Cost vector for distance from goal
    q = np.array([1,0,0,0,-1])
    # Cost vector for object velocity
    v = np.array([0, 0.2, 0,0,0])
    # Cost vector for hand velocity
    w = np.array([0, 0, 0, 0, 0])

    # Add to form cost matrix                                
    state_cost_matrix = np.outer(q,q) + np.outer(v,v) + np.outer(w,w)
    
    # Initilaize computational objects 
    # create LQE object
    kf = LQE(dt, transition_matrix, transition_covariance, 
            control_dynamics, 
            observation_matrix, observation_covariance)

    # create LQR object
    controller = LQR(transition_matrix, transition_covariance, 
            control_dynamics, 
            observation_matrix, observation_covariance,
                state_cost_matrix, action_cost_matrix,
                    num_steps)

    environment = MassSpring(dt, transition_matrix, control_dynamics, 
                    observation_matrix, observation_covariance)

    # Simulate a trial
    true_state = [] 
    measurements = [] 
    predictions = [] 
    action_pos = [] 
    f_ = [] 
    g_ = [] 

    f_manual = [] 
    g_manual = [] 
    state_cost_array = []

    goal = np.ones(num_steps) * target
    # goal = np.sin(np.linspace(0, np.pi*3, num_steps)) * 50

    state = np.array([0,0,0,0, goal[0]])

    #  Iterate
    for t in range(controller.T):    
        if t == 0: 
            environment.initialize_state(state)
            kf.initialize_state(mu_init = state, 
                                sigma_init = np.eye(len(state)))
        # Get measurement from system 
        m = environment.step_measurement()
        measurements.append(m)
            
        # Make a prediction about the state given measurement
        # Incorporate measurement
        kf.project(m)
        # # Update to new posterior
        state = kf.update()
        
        # Make the new state have the time-varying goal
        state[-1] = m[1]#goal[t]
        
        predictions.append(state)    
                    
        # Exert control
        controller.backward_pass()
        u_, state_cost = controller.forward_pass(state)
        # u_ = controller.control_gain(state)
        
        if t < 20: 
            force = 0
        else: 
            force = u_[0]            
        action_pos.append(force)
        f_.append(state[2])
        g_.append(state[3])
        state_cost_array.append(state_cost)
        
        # Transform control into force manually to check
        f_manual.append(environment.f)
        g_manual.append(environment.g)
        environment.transform_control(u_[0], tau1, tau2)
        
        # Update to next iteration's prior
        kf.predict(force) 
        
        # Simulate system
        environment.step_system(force)    
        
    return environment, measurements, predictions, action_pos, goal, num_steps, state_cost_array
    
