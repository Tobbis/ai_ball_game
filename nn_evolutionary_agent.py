import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class EvolvableNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[64, 64], activation_fn=nn.ReLU):
        super(EvolvableNN, self).__init__()
        self.layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(activation_fn())
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

class EvolvableAgent:
    def __init__(self, state_size, action_size, hidden_layers=[64, 64], activation_fn=nn.ReLU):
        self.state_size = state_size
        self.action_size = action_size
        self.model = EvolvableNN(state_size, action_size, hidden_layers, activation_fn)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        action_values = self.model(state).detach().numpy()
        return np.argmax(action_values)

    def mutate(self, mutation_rate=0.1):
        with torch.no_grad():
            for param in self.model.parameters():
                mutation_mask = torch.randn_like(param) * mutation_rate
                param += mutation_mask

    def crossover(self, other_agent):
        child_agent = EvolvableAgent(self.state_size, self.action_size)
        for child_param, parent1_param, parent2_param in zip(child_agent.model.parameters(), self.model.parameters(), other_agent.model.parameters()):
            coin_toss = torch.randint(0, 2, parent1_param.shape).float()
            child_param.data = coin_toss * parent1_param.data + (1 - coin_toss) * parent2_param.data
        
        return child_agent