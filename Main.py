import gym 
import numpy as np
import Replay
from Model import DQN
from torch.nn import HuberLoss
from torch.optim import Adam
import torch
from Model import DQN, device
import matplotlib.pyplot as plt 
from Plot import plot_scores

# create our environment 
env = gym.make("CartPole-v1")

BATCH_SIZE = 32
GAMMA = 0.95
LR = 1e-3
target_net_update_steps = 50
unique_actions = [0, 1]
min_samples_for_training = 1000 
num_episodes = 1000
show_every = 10

# our input and output size for the network
obs_space = env.observation_space.shape[0]
num_actions = env.action_space.n

# storing all experiences in a replay buffer
replay_buffer = Replay.ReplayMemory(5000)

# two DQNs, the first is the online model, the second is only for defining targets
policy_net = DQN(obs_space, num_actions).to(device)
target_net = DQN(obs_space, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = Adam(policy_net.parameters(), lr=LR)
criterion = HuberLoss()

# exploration vs. exploitation, we want to randomly explore with the probability epsilon 

def epsilon_greedy(net, state, epsilon):
    # epsilon needs to be in the range of [0,1]
    assert epsilon <= 1 and epsilon >= 0 

    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (highest Q value)
    best_action = int(net_out.argmax())

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if np.random.random() < epsilon:
        action = np.random.choice(unique_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()


def update_step(policy_net, target_net, replay_buffer, gamma, optimizer, criterion, batch_size):
        
    # Sample the data from the replay memory
    batch = replay_buffer.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
    rewards = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device=device) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)

    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1).cuda())

    # Compute the value function of the next states using the target network 
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device=device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)# Set the required tensor shape

    # Compute the Huber loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


scores = []

for episode in range(num_episodes):
    # original state 
    state = env.reset()

    # total number of steps before the pole fall 
    score = 0
    done = False

    # Go on until the pole falls off
    while not done:

      # choose action with epsilon greedy - exploration vs. exploitation 
      action, q_values = epsilon_greedy(policy_net, state, max(1 - episode / 500, 0.01))
      
      # apply the action 
      next_state, reward, done, info = env.step(action)

      # update the score with the reward
      score += reward

      if done: # if the pole has fallen down 
          next_state = None
      
      # Update the replay memory
      replay_buffer.push(state, action, next_state, reward)

      # Update the network
      if len(replay_buffer) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
          update_step(policy_net, target_net, replay_buffer, GAMMA, optimizer, criterion, BATCH_SIZE)

      # render the environment
      if episode % show_every == 0:
        env.render()

      # Set the current state for the next iteration
      state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
    scores.append(score)
    # Print the final score
    print(f"episode: {episode + 1} - score: {score}") # Print the final score

env.close()
# plot the scores
plot_scores(scores)








    









        



