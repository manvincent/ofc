import numpy as np

class LQR(object): 
    def __init__(self,
                 transition_matrix, transition_covariance,
                 control_dynamics, 
                 observation_matrix, observation_covariance,
                 state_cost_matrix, action_cost_matrix,
                 num_steps):
        # Define variables                                   
        self.D = transition_matrix
        self.Q_e = transition_covariance
        self.B = control_dynamics 
        self.C = observation_matrix
        self.R_e = observation_covariance
        self.Q_r = state_cost_matrix
        self.R_r = action_cost_matrix
        self.T = num_steps
    
        
    def backward_pass(self): 
        # Initialize list of T+1 elements 
        self.P = [None] * (self.T + 1)
        # Initialize last element P_n to be the state cost matrix
        self.P[self.T] = self.Q_r

        # Iterate backwards 
        for i in range(self.T, 0, -1):  
            # Discrete-time Algebraic Riccati equation to get optimal state cost 
            self.P[i-1] = self.Q_r + self.D.T @ self.P[i] @ self.D - \
                            (self.D.T @ self.P[i] @ self.B) @ \
                            np.linalg.pinv(self.R_r + self.B.T @ self.P[i] @ self.B) @ \
                            (self.B.T @ self.P[i] @ self.D)      
            # see automaticaddison's page on LQR
        
          
    def forward_pass(self, state):
        # Print current cost
        # print(state.T @ self.Q_r @ state)
        
        
        # Sweep forward to get feedback (control) gain)        
        # Create a list of N elements
        self.L = [None] * self.T
        
        # Iterate forwards
        for i in range(self.T): 
                        
            # Calculate the optimal feedback gain L
            self.L[i] = -1 * np.linalg.pinv(self.R_r + self.B.T @ self.P[i+1] @ self.B) @ \
                        self.B.T @ self.P[i+1] @ self.D
                                
        # Optimal control input 
        return self.L[0] @ state, state.T @ self.Q_r @ state
