#Continuous Control Project

##Actor-Critic Algorithm

For this project the reacher challenge in the unity environment was solved by using a model free actor-critic algorithm.  This was used for its stable learning in continuous action spaces where in the case of the reacher environment is used to reward 20 distinct arms to reach closer to 20 floating green balls.  An actor and a critic network are trained to solve the Reacher challenge in the Unity Environment where the Critic is trained to evaluate actor actions that maximize the reward and the actor is trained to choose actions that the critic believes are best.  The optimal action-value function is calculated using the Bellman equation which is weighed against the critic's evaluation of the actor actions against the environment states.  Effectively comparing the actors actions against the bootstrapped optimal policy or the target action value function.  The agent collects state to action transitions from the environment and execute learning over batches of experiences.  

```
states, actions, rewards, next_states, dones = experiences

# ---------------------------- update critic ---------------------------- #
# Get predicted next-state actions and Q values from target models
actions_next = self.actor_target(next_states)
Q_targets_next = self.critic_target(next_states, actions_next)
# Compute Q targets for current states (y_i)
Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
# Compute critic loss
Q_expected = self.critic_local(states, actions)
critic_loss = F.mse_loss(Q_expected, Q_targets)
```


After the critic loss is calculated using "reverse-mode automatic differentiation" the gradient of the loss with respect to every weight and bias in the critic network is found.  Using an optimzer (torch.optim.Adam) looks at each parameter and its gradient and the weight in each node in the critic network are nudged in the direction that reduces loss.  
```
## Minimize the loss
self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
```

The actor then uses the "deterministic policy gradient theorem" so that the actor updates in the direction that increases the critic's score. 

```
actions_pred = self.actor_local(states)
actor_loss = -self.critic_local(states, actions_pred).mean()

self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()

```

##Parameters
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.        # L2 weight decay
```

##Plot of Rewards
Below is a chart of the rewards over the course of the challenge and was solved in 85 episodes.
![reacher scores plot](/reacher_scores.png)


```
Episode 5	Reward: 0.83	Average Reward: 0.76
Episode 10	Reward: 3.10	Average Reward: 1.28
Episode 15	Reward: 9.02	Average Reward: 2.63
Episode 20	Reward: 27.72	Average Reward: 7.12
Episode 25	Reward: 37.06	Average Reward: 12.88
Episode 30	Reward: 38.06	Average Reward: 17.09
Episode 35	Reward: 37.63	Average Reward: 20.09
Episode 40	Reward: 36.36	Average Reward: 22.22
Episode 45	Reward: 36.86	Average Reward: 23.78
Episode 50	Reward: 37.29	Average Reward: 25.11
Episode 55	Reward: 37.16	Average Reward: 26.21
Episode 60	Reward: 38.02	Average Reward: 27.16
Episode 65	Reward: 36.70	Average Reward: 27.89
Episode 70	Reward: 37.77	Average Reward: 28.60
Episode 75	Reward: 37.09	Average Reward: 29.16
Episode 80	Reward: 37.10	Average Reward: 29.66
Episode 85	Reward: 37.24	Average Reward: 30.12
Episode 90	Reward: 36.31	Average Reward: 30.51
Episode 95	Reward: 38.14	Average Reward: 30.88
Episode 100	Reward: 35.35	Average Reward: 31.15
Episode 105	Reward: 36.73	Average Reward: 32.95
Episode 110	Reward: 37.30	Average Reward: 34.71
Episode 115	Reward: 37.24	Average Reward: 36.27
Episode 120	Reward: 37.25	Average Reward: 37.12
```
Environment considered solved when the average over the last 100 episodes ≥ 30; we reached 30.12 at episode 85.

##Ideas for Future Work
-[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
-[Entropy regularization for deterministic methods](https://www.sciencedirect.com/science/article/abs/pii/S0020025522013901)
-[Twin Delayed DDPG](https://blog.mlq.ai/deep-reinforcement-learning-twin-delayed-ddpg-algorithm/#google_vignette)
