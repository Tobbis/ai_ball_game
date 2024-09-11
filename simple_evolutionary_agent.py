import numpy as np
import random

class EvolvableAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.randn(state_size, action_size)
    
    def act(self, state):
        # Simple policy: dot product of state and weights
        action_values = np.dot(state, self.weights)
        return np.argmax(action_values)

    def mutate(self, mutation_rate=0.1):
        mutation_matrix = mutation_rate * np.random.randn(*self.weights.shape)
        self.weights += mutation_matrix

    def crossover(self, other_agent):
        new_weights = 0.5 * (self.weights + other_agent.weights)
        child = EvolvableAgent(self.state_size, self.action_size)
        child.weights = new_weights
        return child