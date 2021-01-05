import numpy as np
from numpy.random import randint
import torch
from torch import tensor
import torch.nn as nn
import os
import pickle
import torch.nn.functional as F


class ExpLrDecay:
    def __init__(self, gamma, min_factor):
        self.gamma = gamma
        self.min_factor = min_factor

    def __call__(self, x):
        return max(self.gamma ** x, self.min_factor)


class SumTree:

    def __init__(self, capacity):
        self._num_levels = int(np.ceil(np.log2(capacity)))
        self._num_nodes = int(2 ** self._num_levels - 1)
        self.tree = np.zeros(self._num_nodes + capacity, dtype=np.float64)
        self.capacity = capacity
        self.max_value = 1e-8

    @property
    def total_sum(self):
        return self.tree[0]

    def update(self, i, v):
        i, i_idx = np.unique(i, return_index=True)
        v = v[i_idx]
        i = i + self._num_nodes
        d = v - self.tree[i]
        self.tree[i] = v

        self.max_value = max(self.max_value, np.max(v))

        for _ in range(self._num_levels):
            i = (i - 1) // 2
            np.add.at(self.tree, i, d)

    def sample(self, size):
        r = (np.random.ranf(size) + np.arange(size)) * (self.total_sum / size)
        idx = np.zeros(size, dtype=np.int32)
        for _ in range(self._num_levels):
            idx *= 2
            idx += 1
            b = np.logical_and(r > self.tree[idx], self.tree[idx + 1] != 0)  # this can happen due to imprecisions
            r -= self.tree[idx] * b
            idx += b

        return idx - self._num_nodes, self.tree[idx] / self.total_sum + 1e-16

    def recompute_tree(self):
        a, b, c = self._num_nodes // 2, self._num_nodes, len(self.tree)
        self.tree[a:b] = np.pad(np.sum(self.tree[b:c].reshape(((c - b) // 2, 2)), axis=1), (0, (b - a - (c - b) // 2)), 'constant')
        a, b, c = a // 2, b // 2, self._num_nodes
        while b >= 1:
            self.tree[a:b] = np.sum(self.tree[b:c].reshape((a + 1, 2)), axis=1)
            a, b, c = a//2, b//2, c//2


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        if self.training:
            self.weight_epsilon.normal_()
            self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class FrameBuffer:
    def __init__(self, capacity, frame_size, frame_size_cropped):
        assert np.prod(frame_size) % 8 == 0 and np.prod(frame_size_cropped) % 8 == 0, 'otherwise not implemented'
        self.capacity = capacity
        self.frame_size = frame_size
        self.frame_size_cropped = frame_size_cropped
        self.n_bit_frame_size = np.prod(frame_size) // 8
        self.n_bit_frame_size_cropped = np.prod(frame_size_cropped) // 8
        self.frames = np.zeros((capacity, (np.prod(frame_size) + np.prod(frame_size_cropped)) // 8), dtype=np.uint8)

    def insert(self, index, f, fc):
        self.frames[index, :self.n_bit_frame_size] = np.packbits(f)
        self.frames[index, self.n_bit_frame_size:] = np.packbits(fc)

    def __getitem__(self, i):
        t = self.frames[i]
        f = np.unpackbits(t[..., :self.n_bit_frame_size], axis=-1)
        fc = np.unpackbits(t[..., self.n_bit_frame_size:], axis=-1)
        f = np.reshape(f, f.shape[:-1] + self.frame_size)
        fc = np.reshape(fc, fc.shape[:-1] + self.frame_size_cropped)
        assert f.shape == i.shape + self.frame_size
        return f, fc


class MemoryBuffer:

    def __init__(self, capacity, n_frames, n_steps, frame_size, frame_size_cropped, alpha, beta, gamma, device='cuda'):

        self.frames = FrameBuffer(capacity, frame_size, frame_size_cropped)
        self.actions = np.zeros((capacity,), dtype=np.uint8)
        self.rewards = np.zeros((capacity,), dtype=np.int8)
        self.terminal = np.zeros((capacity,), dtype=np.bool)
        self.index = 0
        self.size = 0
        self.capacity = capacity
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frame_size_cropped = frame_size_cropped
        self.a_n_frames = -np.arange(self.n_frames)
        self.n_steps = n_steps
        self.a_n_steps = np.arange(self.n_steps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.discount = np.power(self.gamma, self.a_n_steps).astype(np.float32)
        self.device = device
        self.last_was_terminal = True
        self.priority_queue = SumTree(capacity)

    def save_warmup(self, save_name):
        assert self.size < self.capacity
        n = self.index
        while n >= 1 and not self.terminal[n-1]:
            n -= 1
        assert n > 1

        np.savez(
            save_name,
            index=n,
            size=n,
            frames=self.frames.frames[:n],
            actions=self.actions[:n],
            rewards=self.rewards[:n],
            terminal=self.terminal[:n],
            tree=self.priority_queue.tree,
            max_value=self.priority_queue.max_value
        )

    def load_warmup(self, save_name):
        warmup = np.load(save_name)
        self.index = warmup['index']
        self.size = warmup['size']
        self.frames.frames[:self.index] = warmup['frames']
        self.actions[:self.index] = warmup['actions']
        self.rewards[:self.index] = warmup['rewards']
        self.terminal[:self.index] = warmup['terminal']
        self.priority_queue.tree = warmup['tree']
        self.priority_queue.max_value = warmup['max_value']

    def update_priorities(self, idx, deltas, last_insertion_was_included):
        if last_insertion_was_included:
            self.priority_queue.update(idx[1:], deltas[1:] ** self.alpha)
        else:
            self.priority_queue.update(idx, deltas ** self.alpha)

    def insert_first(self, f, fc):
        if not self.last_was_terminal:
            raise ValueError('use insert if last wasn\'t terminal')
        self.frames.insert(self.index, f, fc)
        self.last_was_terminal = False

    def insert(self, a, r, t, f1, fc1):
        if self.last_was_terminal:
            raise ValueError('use insert_first after terminal state')
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.terminal[self.index] = t
        self.last_was_terminal = t

        if self.size >= self.n_steps:
            self.priority_queue.update(
                np.array([(self.index + self.capacity - self.n_steps) % self.capacity, (self.index + self.n_frames) % self.capacity]),
                np.array([self.priority_queue.max_value, 0])
            )

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if not t:
            self.frames.insert(self.index, f1, fc1)

    def make_batch(self, batch_size: int, include_last_insertion: bool, idx=None):
        idx, p = self.priority_queue.sample(batch_size) if idx is None else (idx, np.ones(len(idx)) / len(idx))  # sample random
        # idx = np.where(idx < self.index, idx, idx + self.n_frames)  # cant reconstruct state from self.index to self.n_frames - 1
        if include_last_insertion:
            idx = np.insert(idx, 0, self.index)
            p = np.insert(p, 0, 1e8)

        a = self.actions[idx]

        idxn = (idx.reshape((idx.shape[0], 1)) + self.a_n_steps) % self.capacity
        t = np.logical_or.accumulate(self.terminal[idxn], axis=1)

        r = self.rewards[idxn]
        r[:, 1:] *= np.logical_not(t[:, :-1])
        r = r * self.discount
        r = np.sum(r, axis=1)
        t = t[:, -1]

        idxn = idx.reshape((idx.shape[0], 1)) + self.a_n_frames
        f, fc = self.frames[idxn]

        idxn1 = (idxn + self.n_steps) % self.capacity
        f1, fc1 = self.frames[idxn1]

        if self.n_frames > 1:
            # set frames to zero if terminal
            tt = np.logical_not(np.logical_or.accumulate(self.terminal[idxn[:, 1:]], axis=1).reshape((idx.shape[0], self.n_frames - 1, 1, 1)))
            f[:, 1:] *= tt
            fc[:, 1:] *= tt

        if self.n_steps < self.n_frames - 1:
            tt = np.logical_not(np.logical_or.accumulate(self.terminal[idxn1[:, self.n_steps + 1:]], axis=1).reshape((idx.shape[0], self.n_frames - self.n_steps - 1, 1, 1)))
            f[:, self.n_steps + 1:] *= tt
            fc[:, self.n_steps + 1:] *= tt

        w = p ** -self.beta
        w /= np.max(w)

        if include_last_insertion:
            w[0] = 0

        def to_torch(x, dtype):
            return torch.from_numpy(x).to(self.device).to(dtype)

        return to_torch(f, torch.float), \
               to_torch(fc, torch.float), \
               to_torch(a, torch.long), \
               to_torch(r, torch.float), \
               to_torch(t, torch.bool), \
               to_torch(f1, torch.float), \
               to_torch(fc1, torch.float), \
               to_torch(w, torch.float), \
               idx
        # f1 is undefined if s terminal or s is at self.index

    def save(self, file_name='memory_buffer'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name='memory_buffer'):
        with open(file_name, 'rb') as f:
            ret = pickle.load(f)
            ret.last_was_terminal = True
            # ret.terminal[ret.index - 1] = True
            return ret

class NetworkOld(nn.Module):

    def __init__(self, n_frames, n_actions, n_atoms, hidden_size=512):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_frames, 32, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.LeakyReLU(),
        )
        self.conv_fc = nn.Sequential(
            nn.Conv2d(n_frames, 32, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_f_output_size = 9 * 9 * 64
        self.conv_fc_output_size = 9 * 9 * 64

        self.h1 = nn.Sequential(
            NoisyLinear(self.conv_f_output_size + self.conv_fc_output_size, hidden_size),
            nn.LeakyReLU()
        )

        self.v_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_atoms)
        )

        self.a_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_actions * n_atoms)
        )
        # self.cuda_streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

    def forward(self, f, fc, log=False):

        '''
        torch.cuda.synchronize()
        if reset_noise:
            with torch.cuda.stream(self.cuda_streams[0]):
                self.reset_noise()
        with torch.cuda.stream(self.cuda_streams[1]):
            f = self.conv_f(f).view(-1, self.conv_f_output_size)
        with torch.cuda.stream(self.cuda_streams[2]):
            fc = self.conv_fc(fc).view(-1, self.conv_fc_output_size)
        torch.cuda.synchronize()
        #for s in self.cuda_streams:
        #    torch.cuda.default_stream().wait_stream(s)
        '''

        f = self.conv_f(f).view(-1, self.conv_f_output_size)
        fc = self.conv_fc(fc).view(-1, self.conv_fc_output_size)

        h = self.h1(torch.cat((f, fc), dim=1))

        # duelling networks
        v = self.v_stream(h).view(-1, 1, self.n_atoms)
        a = self.a_stream(h).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        if log:
            return F.log_softmax(q, dim=2)
        return F.softmax(q, dim=2)

    def reset_noise(self):
        self.h1[0].reset_noise()
        self.v_stream[0].reset_noise()
        self.v_stream[2].reset_noise()
        self.a_stream[0].reset_noise()
        self.a_stream[2].reset_noise()

    def save(self, file_name='network'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name='network'):
        with open(file_name, 'rb') as f:
            return pickle.load(f)


class Network(nn.Module):

    def __init__(self, n_frames, n_actions, n_atoms, hidden_size=512):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_frames, 32, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_fc = nn.Sequential(
            nn.Conv2d(n_frames, 32, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_f_output_size = 5 * 5 * 64
        self.conv_fc_output_size = 5 * 5 * 64

        self.h1 = nn.Sequential(
            nn.Linear(self.conv_f_output_size + self.conv_fc_output_size, hidden_size),
            nn.LeakyReLU()
        )

        self.v_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_atoms)
        )

        self.a_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_actions * n_atoms)
        )

    def forward(self, f, fc, log=False):
        f = self.conv_f(f).view(-1, self.conv_f_output_size)
        fc = self.conv_fc(fc).view(-1, self.conv_fc_output_size)

        h = self.h1(torch.cat((f, fc), dim=1))

        # duelling networks
        v = self.v_stream(h).view(-1, 1, self.n_atoms)
        a = self.a_stream(h).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        if log:
            return F.log_softmax(q, dim=2)
        return F.softmax(q, dim=2)

    def reset_noise(self):
        self.v_stream[0].reset_noise()
        self.v_stream[2].reset_noise()
        self.a_stream[0].reset_noise()
        self.a_stream[2].reset_noise()

    def save(self, file_name='network'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name='network'):
        with open(file_name, 'rb') as f:
            return pickle.load(f)


class NetworkQR(nn.Module):

    def __init__(self, n_frames, n_actions, n_atoms, hidden_size=512):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_frames, 32, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_fc = nn.Sequential(
            nn.Conv2d(n_frames, 32, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_f_output_size = 5 * 5 * 64
        self.conv_fc_output_size = 5 * 5 * 64

        self.h1 = nn.Sequential(
            nn.Linear(self.conv_f_output_size + self.conv_fc_output_size, hidden_size),
            nn.LeakyReLU()
        )

        self.v_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_atoms)
        )

        self.a_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_actions * n_atoms)
        )

    def forward(self, f, fc):
        f = self.conv_f(f).view(-1, self.conv_f_output_size)
        fc = self.conv_fc(fc).view(-1, self.conv_fc_output_size)

        h = self.h1(torch.cat((f, fc), dim=1))

        # duelling networks
        v = self.v_stream(h).view(-1, 1, self.n_atoms)
        a = self.a_stream(h).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        self.v_stream[0].reset_noise()
        self.v_stream[2].reset_noise()
        self.a_stream[0].reset_noise()
        self.a_stream[2].reset_noise()

    def save(self, file_name='network'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name='network'):
        with open(file_name, 'rb') as f:
            return pickle.load(f)



if __name__ == '__main__':

    from timeit import timeit


    def ws(a, *x, **xx):
        torch.cuda.synchronize()
        a(*x, **xx)
        torch.cuda.synchronize()


    model = Network(4, 3, 21).cuda()
    f, fc = torch.rand((48, 4, 60, 60), device='cuda'), torch.rand((48, 4, 60, 60), device='cuda')
    number = 1000
    timeit(lambda: ws(model, f, fc), number=100)
    print(timeit(lambda: ws(model, f, fc), number=number) / number)


