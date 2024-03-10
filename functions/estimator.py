import numpy as np
from collections import namedtuple


# Helper functions 
gaussian = namedtuple('Gaussian', ['mean', 'cov'])
# Define class for predictor: LQE / KF
class LQE(object): 
    def __init__(self, dt, 
                 transition_matrix, transition_covariance,
                 control_dynamics, 
                 observation_matrix, observation_covariance):
        # Define variables                                   
        self.dt = dt 
        self.D = transition_matrix
        self.Q_e = transition_covariance
        self.B = control_dynamics 
        self.C = observation_matrix
        self.R_e = observation_covariance
        
    def initialize_state(self, mu_init, sigma_init):
        self.state = gaussian(mu_init, sigma_init)

    def predict(self, u):
        # Get new prior given dynamics
        # Mean 
        prior_mean = self.D @ self.state.mean + self.B @ [u]
        # Covariance 
        prior_cov = self.D @ self.state.cov @ self.D.T + self.Q_e 
        
        # Assign to state variable 
        self.state = gaussian(prior_mean, prior_cov)        
        
    def project(self, m): 
        # Incorporate measurement 
        self.innovation_estimate = m - self.C @ self.state.mean
        innovation_covariance = self.C @ self.state.cov @ self.C.T + self.R_e
        # Kalman gain 
        self.K = self.state.cov @ self.C.T @ np.linalg.pinv(innovation_covariance)
          
    def update(self): 
        # Update to posterior given measurement
        # Mean
        posterior_mean = self.state.mean + self.K @ self.innovation_estimate
        # Covariance 
        I = np.eye(self.C.shape[1])        
        posterior_cov = (I - self.K @ self.C) @ self.state.cov
        
        # Assign to state variable 
        self.state = gaussian(posterior_mean, posterior_cov)
        return self.state.mean
        