import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.core import Env
from torch import nn
import matplotlib.pyplot as plt
import gym

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes, activ_func):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        self.activation = activ_func
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers[:-1]:
            x = getattr(F, self.activation)(layer(x))
        x = self.layers[-1](x)
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()



# Function to train model and get run results
def train_model(
  lr = 0.001,
  epsilon = 0.9,
  epsilon_percentage = 0.99,
  network_architecture = [4, 16, 16, 2],
  optimizer_type = "Adam",
  activation_func = "relu",
  batch_size = 20,
  replay_buffer_size = 10000,
  update_target_num = 1
  ):

  NUM_RUNS = 10
  runs_results = []
  epsilon_min = 0.05  # defined a minimum epsilon value 

  env = gym.make('CartPole-v1')
  for run in range(NUM_RUNS):
    print(f"Starting run {run+1} of {NUM_RUNS}")
    policy_net = DQN(network_architecture, activation_func) # changed policy network archi as variable
    target_net = DQN(network_architecture, activation_func) # changed target network archi as variable
    update_target(target_net, policy_net)
    target_net.eval()

    # changed optimizer and lr as a variable
    optimizer = getattr(optim, optimizer_type)(policy_net.parameters(), lr=lr) 

    # Changed replay buffer size as a variable 
    memory = ReplayBuffer(int(replay_buffer_size))

    steps_done = 0

    episode_durations = []

    for i_episode in range(300):
      if (i_episode+1) % 50 == 0:
        print("episode ", i_episode+1, "/", 300)

      observation, info = env.reset()
      state = torch.tensor(observation).float()

      done = False
      terminated = False
      t = 0
      while not (done or terminated):

        # Select and perform an action
        action = epsilon_greedy(epsilon, policy_net, state)
        # Implement epsilon-decay 
        if epsilon > epsilon_min:
          epsilon = epsilon*epsilon_percentage
        else:
          epsilon = epsilon_min

        observation, reward, done, terminated, info = env.step(action)
        reward = torch.tensor([reward])
        action = torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()

        memory.push([state, action, next_state, reward, torch.tensor([done])])

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if not len(memory.buffer) < batch_size:  # implement batch size of buffer to sample
          transitions = memory.sample(batch_size)
          state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
          # Compute loss
          mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
          # Optimize the model
          optimizer.zero_grad()
          mse_loss.backward()
          optimizer.step()

        if done or terminated:
          episode_durations.append(t + 1)
        t += 1
      # Update the target network, copying all weights and biases in DQN
      if i_episode % update_target_num == 0: # set how many epsiode to update as variable
        update_target(target_net, policy_net)

    runs_results.append(episode_durations)

  return runs_results


# Function to select which hyperparameter to test against baselin 
def hyper_param_testing(
  lr_flag=False,
  epsilon_flag=False,
  epsilon_percentage_flag=False,
  network_architecture_flag=False,
  optimizer_type_flag=False,
  activation_func_flag=False,
  batch_size_flag=False,
  replay_buffer_size_flag=False,
  update_target_num_flag=False
  ):

  final_results = []

  # to test learning rate
  if lr_flag:
    for lr in [0.1, 0.01, 0.001, 0.0001]:
      print("for %f : " % lr)
      run_results = train_model(lr=lr)
      final_results.append(run_results)

  # to test epsilon 
  if epsilon_flag:
    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      print("for %f : " % epsilon)
      run_results = train_model(epsilon=epsilon)
      final_results.append(run_results)
      
  # to test epsilon-decay rate
  if epsilon_percentage_flag:
    for epsilon_percentage in [0.9, 0.99, 1.0]:
      print("for %f : " % epsilon_percentage)
      run_results = train_model(epsilon_percentage=epsilon_percentage)
      final_results.append(run_results)

  # to test types of network architecture  
  if network_architecture_flag:
    for network_architecture in [# 1 hidden layer up to 256 as maximum total hidden layer size
                                                   [4, 16, 2],
                                                   [4, 32, 2],
                                                   [4, 64, 2],
                                                   [4, 128, 2],
                                                   [4, 256, 2],
                                                   # 2 hidden layers with 2nd hidden layer size smaller than first to allow better convergence to 2 output size
                                                   [4, 16, 16, 2],
                                                   [4, 32, 16, 2],
                                                   [4, 64, 16, 2],
                                                   [4, 128, 16, 2],
                                                   [4, 32, 32, 2],
                                                   [4, 64, 32, 2],
                                                   [4, 128, 32, 2],
                                                   [4, 64, 64, 2],
                                                   [4, 128, 64, 2],
                                                   [4, 128, 128, 2],
                                                   # 3 hidden layers with pyramid architecture
                                                   [4, 16, 32, 16, 2],
                                                   [4, 32, 64, 32, 2],
                                                   [4, 64, 128, 64, 2]]:
      print("for %s : " % str(network_architecture))
      run_results = train_model(network_architecture=network_architecture)
      final_results.append(run_results)

  # to test optimizer type  
  if optimizer_type_flag:
    for optimizer_type in ["Adam", "RMSprop", "SGD"]:
      print("for %s : " % str(optimizer_type))
      run_results = train_model(optimizer_type=optimizer_type)
      final_results.append(run_results)

  # to test activation function 
  if activation_func_flag:
    for activation_func in ["sigmoid", "tanh", "relu", "leaky_relu"]:
      print("for %s : " % str(activation_func))
      run_results = train_model(activation_func=activation_func)
      final_results.append(run_results)

  # to test batch size 
  if batch_size_flag:
    for batch_size in [1, 5, 10, 15, 20]:
      print("for %f : " % batch_size)
      run_results = train_model(batch_size=batch_size)
      final_results.append(run_results)

  # to test replay buffer size 
  if replay_buffer_size_flag:
    for replay_buffer_size in [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000]:
      print("for %f : " % replay_buffer_size)
      run_results = train_model(replay_buffer_size=replay_buffer_size)
      final_results.append(run_results)

  # to test number of episode before updating target network 
  if update_target_num_flag:
    for update_target_num in [1, 2, 5, 10]:
      print("for %f : " % update_target_num)
      run_results = train_model(update_target_num=update_target_num)
      final_results.append(run_results)

  return final_results


# Function to get retrun result and policy network with defined parameters
def get_results_and_dqn_policy_net(
  lr = 0.001, 
  epsilon = 0.9,
  epsilon_percentage = 0.99,
  network_architecture = [4, 64, 64, 2],
  optimizer_type = "Adam",
  activation_func = "relu",
  batch_size = 20,
  replay_buffer_size = 10000,
  update_target_num = 1
  ):

  runs_results = []
  epsilon_min = 0.05  # defined a minimum epsilon value 
  NUM_RUNS = 10
  
  env = gym.make('CartPole-v1')
  for run in range(NUM_RUNS):
    print(f"Starting run {run+1} of {NUM_RUNS}")
    policy_net = DQN(network_architecture, activation_func) # changed policy network archi as variable
    target_net = DQN(network_architecture, activation_func) # changed target network archi as variable
    update_target(target_net, policy_net)
    target_net.eval()

    # changed optimizer and lr as a variable
    optimizer = getattr(optim, optimizer_type)(policy_net.parameters(), lr=lr)

    # Changed replay buffer size as a variable 
    memory = ReplayBuffer(int(replay_buffer_size))

    steps_done = 0

    episode_durations = []

    for i_episode in range(300):
      if (i_episode+1) % 50 == 0:
        print("episode ", i_episode+1, "/", 300)

      observation, info = env.reset()
      state = torch.tensor(observation).float()

      done = False
      terminated = False
      t = 0
      while not (done or terminated):

        # Select and perform an action
        action = epsilon_greedy(epsilon, policy_net, state)
        # Implement epsilon-decay 
        if epsilon > epsilon_min:
          epsilon = epsilon*epsilon_percentage
        else:
          epsilon = epsilon_min

        observation, reward, done, terminated, info = env.step(action)
        reward = torch.tensor([reward])
        action = torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()

        memory.push([state, action, next_state, reward, torch.tensor([done])])

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if not len(memory.buffer) < batch_size:  # implement batch size of buffer to sample
          transitions = memory.sample(batch_size)
          state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
          # Compute loss
          mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
          # Optimize the model
          optimizer.zero_grad()
          mse_loss.backward()
          optimizer.step()

        if done or terminated:
          episode_durations.append(t + 1)
        t += 1
      # Update the target network, copying all weights and biases in DQN
      if i_episode % update_target_num == 0: # set how many epsiode to update as variable
        update_target(target_net, policy_net)

    runs_results.append(episode_durations)

  return policy_net, runs_results

# defining test function that was used for train_model with episode = 50 for short simulation to ensure code works
def train_model_for_test(
  lr = 0.001,
  epsilon = 0.9,
  epsilon_percentage = 0.99,
  network_architecture = [4, 16, 16, 2],
  optimizer_type = "Adam",
  activation_func = "relu",
  batch_size = 20,
  replay_buffer_size = 10000,
  update_target_num = 1
  ):

  runs_results = []
  epsilon_min = 0.05
  NUM_RUNS = 10

  env = gym.make('CartPole-v1')
  for run in range(NUM_RUNS):
    print(f"Starting run {run+1} of {NUM_RUNS}")
    policy_net = DQN(network_architecture, activation_func) # changed policy network architecture as variable
    target_net = DQN(network_architecture, activation_func) # changed target network architecture as variable
    update_target(target_net, policy_net)
    target_net.eval()

    # changed optimizer and lr as a variable
    optimizer = getattr(optim, optimizer_type)(policy_net.parameters(), lr=lr)

    memory = ReplayBuffer(int(replay_buffer_size))

    steps_done = 0

    episode_durations = []

    for i_episode in range(50):
      if (i_episode+1) % 50 == 0:
        print("episode ", i_episode+1, "/", 300)

      observation, info = env.reset()
      state = torch.tensor(observation).float()

      done = False
      terminated = False
      t = 0
      while not (done or terminated):

        # Select and perform an action
        action = epsilon_greedy(epsilon, policy_net, state)
        if epsilon > epsilon_min:
          epsilon = epsilon*epsilon_percentage
        else:
          epsilon = epsilon_min

        observation, reward, done, terminated, info = env.step(action)
        reward = torch.tensor([reward])
        action = torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()

        memory.push([state, action, next_state, reward, torch.tensor([done])])

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if not len(memory.buffer) < batch_size:
          transitions = memory.sample(batch_size)
          state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
          # Compute loss
          mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
          # Optimize the model
          optimizer.zero_grad()
          mse_loss.backward()
          optimizer.step()

        if done or terminated:
          episode_durations.append(t + 1)
        t += 1
      # Update the target network, copying all weights and biases in DQN
      if i_episode % update_target_num == 0: # can set as variable
        update_target(target_net, policy_net)

    runs_results.append(episode_durations)

  return runs_results


# Visualising the greedy Q-values for a stationary cart in the middle of the track
# 2D plot showing policy as a function of pole angle and angular velocity (omega)

# This plots the policy and Q values according to the network currently
# stored in the variable "policy_net"

def visualise_DQN(DQN_policy_net, velocity, q=False):
    policy_net = DQN_policy_net   # randomly initialised, replace with your trained DQN

    angle_range = (.2095)*1.5 # you may modify this range
    omega_range = 2     # you may modify this range

    angle_samples = 100
    omega_samples = 100
    angles = torch.linspace(angle_range, -angle_range, angle_samples)
    omegas = torch.linspace(-omega_range, omega_range, omega_samples)

    plt.figure(figsize=(5,3))
        
    greedy_q_array = torch.zeros((angle_samples, omega_samples))
    policy_array = torch.zeros((angle_samples, omega_samples))
    for i, angle in enumerate(angles):
        for j, omega in enumerate(omegas):
            state = torch.tensor([0., velocity, angle, omega])
            with torch.no_grad():
                q_vals = policy_net(state)
                greedy_action = q_vals.argmax()
                greedy_q_array[i, j] = q_vals[greedy_action]
                policy_array[i, j] = greedy_action
    if q:
        plt.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
        plt.colorbar()
        plt.title(f"Q function for velocity={velocity}")
    else:
        contour_plot = plt.contourf(angles, omegas, policy_array.T, cmap='cividis')
        # plt.clabel(contour_plot, inline=1, fontsize=10)
        labels = ["0 - move cart to left", "1 - move cart to right"]
        contour_labels = [contour_plot.collections[0],contour_plot.collections[1]]
        colours = ["#002f6d", "#edd54a"]
        contour_colours = [plt.Rectangle((0,0),1,1,fc=pc) for pc in colours]
        # for l in range(len(labels)):
        #     contour_plot.collections[l].set_label(labels[l])
        # print(policy_array)
        plt.legend(contour_colours, labels)
        plt.title(f"Greedy policy action for velocity={velocity}", fontsize=16)
    plt.xlabel("Angle", fontsize=14)
    plt.ylabel("Angular velocity", fontsize=14)
    plt.show()