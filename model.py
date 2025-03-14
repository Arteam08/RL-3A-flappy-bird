import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



#Parameter
BUFFER_SIZE = 757624   # replay buffer size
BATCH_SIZE = 228             # minibatch size
GAMMA = 0.8777525034865641                # discount factor
TAU = 0.005566248249524419                      # for soft update of target parameters
UPDATE_EVERY = 3         # how often to update the network

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class QNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed):
    """
    Initialize parameters and build model.
    Params:
      - state_size (int): Dimension of each state
      - action_size (int): Dimension of each action
      - seed (int): Random seed
      - fc1_unit (int): Number of nodes in first hidden layer
      - fc2_unit (int): Number of nodes in second hidden layer
    """
    super(QNetwork, self).__init__() ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)

    
    fc1_unit=128
    fc2_unit=128
    self.fc1=nn.Linear(state_size,fc1_unit)
    self.fc2=nn.Linear(fc1_unit,fc2_unit)
    self.fc3=nn.Linear(fc2_unit,action_size)
    

  def forward(self, state):
    """
    Build a network that maps state -> action values.
    """
    x=F.relu(self.fc1(state))
    x=F.relu(self.fc2(x))
    return self.fc3(x)
    
class Agent():
  """
  Interacts with and learns form environment.
  """

  def __init__(self, state_size, action_size, seed):
    """
    Initialize an Agent object.
    Params:
      - state_size (int): dimension of each state
      - action_size (int): dimension of each action
      - seed (int): random seed
    """

    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

    self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=0.0001)

    # Replay Memory
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0

  def step(self, state, action, reward, next_step, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_step, done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step+1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > BATCH_SIZE:
        experience = self.memory.sample()
        self.learn(experience, GAMMA)


  def act(self, state, eps = 0):
    """
    Returns action for given state as per current policy.
    Params:
      - state (array_like): current state
      - eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selction
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, gamma):
    """
    Update value parameters using given batch of experience tuples.
    Params:
      - experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
      - gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    
    ## TODO: compute and minimize the loss
    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

    # Compute Q targets for current states
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

  def soft_update(self, local_model, target_model, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params:
      - local model (PyTorch model): weights will be copied from
      - target model (PyTorch model): weights will be copied to
      - tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


class ReplayBuffer:
  """
  Fixed-size buffe to store experience tuples.
  """
  def __init__(self, action_size, buffer_size, batch_size, seed):
    """
    Initialize a ReplayBuffer object.
    Params:
      - action_size (int): dimension of each action
      - buffer_size (int): maximum size of buffer
      - batch_size (int): size of each training batch
      - seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    self.seed = random.seed(seed)

  def add(self,state, action, reward, next_state,done):
    """
    Add a new experience to memory.
    """
    e = self.experiences(state,action,reward,next_state,done)
    self.memory.append(e)

  def sample(self):
    """
    Randomly sample a batch of experiences from memory.
    """
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states,actions,rewards,next_states,dones)

  def __len__(self):
    """
    Return the current size of internal memory.
    """
    return len(self.memory)