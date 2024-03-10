import numpy as np
np.random.seed(1234)

#%% Create a class to simulate the try system 
class MassSpring(object): 
    def __init__(self, dt, 
                 transition_matrix,
                 control_dynamics, 
                 observation_matrix, observation_covariance):
        # Define variables                                   
        self.dt = dt 
        self.D = transition_matrix
        self.B = control_dynamics 
        self.C = observation_matrix
        self.R_e = observation_covariance
        self.true_states = []
        
    def initialize_state(self, starting_state):
        self.state = starting_state
        self.g, self.f = 0, 0 
        
    # Transform control signal to force (sanity check, not incorporated in system)
    def transform_control(self, u, tau1, tau2): 
        new_g = (1 - self.dt/tau1) * self.g + self.dt/tau1 * u
        new_f = (1 - self.dt/tau2) * self.f + self.dt/tau2 * self.g
        self.g, self.f = new_g, new_f 
        
    def step_system(self, u): 
        self.state = self.D @ self.state + self.B @ [u]
        self.true_states.append(self.state)
                    
    def step_measurement(self): 
        measurement = self.C @ self.state + np.sqrt(self.R_e) * np.random.randn()
        return np.diag(measurement)


#%% A simple mass-spring simulator to look at object output of any arbitrary trajectory
def simple_simulation(trajectory, k=1.0, mass=0.5, dt=1/60): 
    pos = [] 
    vel = [] 
    state = np.array([[0],
                      [0]])
    for f in trajectory:     
        dyn = np.array([[1,dt],
                        [dt*-k/mass, 1]])
        B = np.array([[0],
                    [dt/mass]])
        state = dyn @ state + B * f
        pos.append(state[0,0])       
        vel.append(state[1,0])
    return np.array(pos), np.array(vel)    