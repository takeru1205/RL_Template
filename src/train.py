import gym
import torch

from agent import Agent 

print(f'CUDA is {"Enable" if torch.cuda.is_available() else "Disable"}')

env = gym.make('CartPole-v1')
agent = Agent(env=env, cuda=False)

for _ in range(10):
    state = env.reset()
    cumulative_reward = 0
    for _ in range(env.spec.max_episode_steps):
        env.render(mode='rgb_array')
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        cumulative_reward += reward
        state = next_state
        if done:
            print(f'Cumulative Reward: {cumulative_reward}')
            break
env.close()

print(f'Number of stored transitions: {len(agent.buffer)}')


