#%% Import packages

from functions.simulations import simulate_human, simulate_model
from functions.plot_sim import plot_model
            
#%% Run model
environment, measurements, predictions, action_pos, goal, num_steps, state_cost_array = simulate_model()
#%% Plot model
fig = plot_model(environment, measurements, predictions, action_pos, goal, num_steps)

# %%

