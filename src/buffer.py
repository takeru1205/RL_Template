import numpy as np
import torch


class ReplayBuffer:
    def __init__(
            self,
            state_dim,
            act_dim,
            capacity=int(1e+6),
            cuda=torch.cuda.is_available()
            ):

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.mem_ctrl = 0
        self.cuda = cuda
        self.state_buffer = np.zeros((capacity, state_dim), dtype=float)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=float)
        self.action_buffer = np.zeros((capacity, act_dim), dtype=float)
        self.reward_buffer = np.zeros(capacity, dtype=float)
        self.done_buffer = np.zeros(capacity, dtype=int)

    def store(
            self,
            state,
            action,
            reward,
            next_state,
            done
            ):

        index = self.mem_ctrl % self.capacity
        self.state_buffer[index] = state
        self.next_state_buffer[index] = next_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = np.array([reward]).astype(np.float)
        self.done_buffer[index] = np.array([1 - done]).astype(np.uint8)
        self.mem_ctrl += 1

    def sample(self, batch_size=256):
        mem_size = min(self.mem_ctrl, self.capacity)
        batch = np.random.choice(mem_size, batch_size)

        states = torch.from_numpy(self.state_buffer[batch])
        next_states = torch.from_numpy(self.next_state_buffer[batch])
        actions = torch.from_numpy(self.action_buffer[batch])
        rewards = torch.from_numpy(self.reward_buffer[batch])
        dones = torch.from_numpy(self.done_buffer[batch])

        if self.cuda:
            states = states.cuda()
            next_states = next_states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()

        return states, next_states, actions, rewards, dones

    def __len__(self):
        return self.mem_ctrl


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(
            state_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
            capacity=1000
            )
    state = env.reset()
    for i in range(10):
        for _ in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, reward, next_state, done)
            state = next_state
        print(f'Length of Buffer: {len(buffer)}')

    s, s_, a, r, d = buffer.sample(128)

    assert s.shape == (128, env.observation_space.shape[0])
    assert s_.shape == (128, env.observation_space.shape[0])
    assert a.shape == (128, env.action_space.n)
    assert r.shape == (128,)
    assert d.shape == (128,)

    assert str(s.device) == 'cuda:0'


