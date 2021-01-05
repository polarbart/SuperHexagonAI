import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F


class ExpLrDecay:
    def __init__(self, gamma, min_factor):
        self.gamma = gamma
        self.min_factor = min_factor

    def __call__(self, x):
        return max(self.gamma ** x, self.min_factor)


class FrameBuffer:
    """
    compresses and saves the frame and the cropped frame (f and fc)
    """
    def __init__(self, capacity, frame_size, frame_size_cropped):
        assert np.prod(frame_size) % 8 == 0 and np.prod(frame_size_cropped) % 8 == 0, 'otherwise not implemented'
        self.capacity = capacity
        self.frame_size = frame_size
        self.frame_size_cropped = frame_size_cropped
        self.n_bit_frame_size = np.prod(frame_size) // 8
        self.n_bit_frame_size_cropped = np.prod(frame_size_cropped) // 8
        self.frames = np.zeros((capacity, (np.prod(frame_size) + np.prod(frame_size_cropped)) // 8), dtype=np.uint8)

    def insert(self, index, f, fc):
        # f and fc are bit matrices
        # they can be "compressed" with np.packbits
        self.frames[index, :self.n_bit_frame_size] = np.packbits(f)
        self.frames[index, self.n_bit_frame_size:] = np.packbits(fc)

    def __getitem__(self, i):
        t = self.frames[i]
        f = np.unpackbits(t[..., :self.n_bit_frame_size], axis=-1)
        fc = np.unpackbits(t[..., self.n_bit_frame_size:], axis=-1)
        f = np.reshape(f, f.shape[:-1] + self.frame_size)
        fc = np.reshape(fc, fc.shape[:-1] + self.frame_size_cropped)
        return f, fc


class MemoryBuffer:

    def __init__(self, capacity_per_level, n_levels, n_frames, frame_size, frame_size_cropped, gamma, device='cuda'):

        # each of the six levels occupies one sixth of the memory buffer
        total_capacity = capacity_per_level * n_levels
        assert 0 <= total_capacity <= np.iinfo(np.int).max

        self.frames = FrameBuffer(total_capacity, frame_size, frame_size_cropped)
        self.actions = np.zeros((total_capacity,), dtype=np.uint8)
        self.rewards = np.zeros((total_capacity,), dtype=np.int8)
        self.terminal = np.zeros((total_capacity,), dtype=np.bool)

        self.a_n_levels = np.arange(n_levels, dtype=np.int)
        self.index = capacity_per_level * self.a_n_levels
        self.size = np.zeros(n_levels, dtype=np.int)
        self.capacity_per_level = capacity_per_level
        self.n_levels = n_levels
        self.total_capacity = total_capacity
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frame_size_cropped = frame_size_cropped
        self.a_n_frames = -np.arange(self.n_frames, dtype=np.int)
        self.gamma = gamma
        self.device = device
        self.last_was_terminal = True

    def save_warmup(self, save_name):
        assert np.all(self.size < self.capacity_per_level)
        index = self.index.copy()

        # find the last index where the game terminated
        # such that just full episodes are saved
        for i in range(index.shape[0]):
            while index[i] >= 1 and not self.terminal[index[i] - 1]:
                index[i] -= 1

        # helper function
        def pack(a):
            return np.array([a[i * self.capacity_per_level:index[i]] for i in range(index.shape[0])], dtype=np.object)

        np.savez(
            save_name,
            index=index,
            size=index % self.capacity_per_level,
            frames=pack(self.frames.frames),
            actions=pack(self.actions),
            rewards=pack(self.rewards),
            terminal=pack(self.terminal),
        )

    def load_warmup(self, save_name):

        warmup = np.load(save_name, allow_pickle=True)
        self.index = warmup['index']
        self.size = warmup['size']

        # helper function
        def unpack(a, s):
            for i in range(self.index.shape[0]):
                a[i * self.capacity_per_level:self.index[i]] = s[i]

        unpack(self.frames.frames, warmup['frames'])
        unpack(self.actions, warmup['actions'])
        unpack(self.rewards, warmup['rewards'])
        unpack(self.terminal, warmup['terminal'])

    def insert_first(self, level, f, fc):
        # insert the first frame after the agent has died
        if not self.last_was_terminal:
            raise ValueError('use insert if last wasn\'t terminal')
        self.frames.insert(self.index[level], f, fc)
        self.last_was_terminal = False

    def insert(self, level, a, r, t, f1, fc1):
        # insert action, reward, terminal and next frame
        if self.last_was_terminal:
            raise ValueError('use insert_first after terminal state')
        self.actions[self.index[level]] = a
        self.rewards[self.index[level]] = r
        self.terminal[self.index[level]] = t
        self.last_was_terminal = t

        # increment / wrap around the index
        if self.index[level] < (self.capacity_per_level * (level + 1) - 1):
            self.index[level] += 1
        else:
            self.index[level] = self.capacity_per_level * level
        self.size[level] = min(self.size[level] + 1, self.capacity_per_level)

        # insert next frame
        if not t:
            self.frames.insert(self.index[level], f1, fc1)

    def make_batch(self, batch_size: int):

        assert batch_size % self.n_levels == 0, 'batch_size should be multiple of n_levels'

        # randomly draw samples
        # draw the same number of samples for each of the six levels
        sizes = np.repeat(self.size, batch_size // self.n_levels)
        offsets = np.repeat(self.a_n_levels * self.capacity_per_level, batch_size // self.n_levels)
        idx = np.random.randint(sizes, dtype=np.int)
        idxo = idx + offsets

        # extract action, reward and terminal
        a = self.actions[idxo]
        r = self.rewards[idxo]
        t = self.terminal[idxo]

        # extract state and next state taking frame stack into account
        idxf = (idx[:, np.newaxis] + self.a_n_frames) % self.capacity_per_level
        idxfo = idxf + offsets[:, np.newaxis]
        f, fc = self.frames[idxfo]

        idxf1 = (idxf + 1) % self.capacity_per_level
        idxf1o = idxf1 + offsets[:, np.newaxis]
        f1, fc1 = self.frames[idxf1o]

        # mask out the frames if terminal i. e. if they are from the previous episode
        if self.n_frames > 1:
            # set frames to zero if terminal
            tt = np.logical_not(np.logical_or.accumulate(self.terminal[idxfo[:, 1:]], axis=1).reshape((-1, self.n_frames - 1, 1, 1)))
            f[:, 1:] *= tt
            fc[:, 1:] *= tt
            f1[:, 2:] *= tt[:, :-1]
            fc1[:, 2:] *= tt[:, :-1]

        # helper function
        def to_torch(x, dtype):
            return torch.from_numpy(x).to(self.device).to(dtype)

        return to_torch(f, torch.float), \
               to_torch(fc, torch.float), \
               to_torch(a, torch.long), \
               to_torch(r, torch.float), \
               to_torch(t, torch.bool), \
               to_torch(f1, torch.float), \
               to_torch(fc1, torch.float)
        # f1 is undefined if s terminal or s is at self.index


class NoisyLinear(nn.Module):
    """
    NoisyLinear layer with bias
    """
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
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class Network(nn.Module):

    def __init__(self, n_frames, n_actions, n_atoms, hidden_size=1024):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_frames, 64, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_fc = nn.Sequential(
            nn.Conv2d(n_frames, 64, 7, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv_f_output_size = 5 * 5 * 128
        self.conv_fc_output_size = 5 * 5 * 128

        self.h = nn.Sequential(
            nn.Linear(self.conv_f_output_size + self.conv_fc_output_size, hidden_size),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            NoisyLinear(hidden_size // 2, n_actions * n_atoms)
        )

    def forward(self, f, fc, log=False):
        f = self.conv_f(f).view(-1, self.conv_f_output_size)
        fc = self.conv_fc(fc).view(-1, self.conv_fc_output_size)

        o = self.h(torch.cat((f, fc), dim=1)).view(-1, self.n_actions, self.n_atoms)

        return F.log_softmax(o, dim=2) if log else F.softmax(o, dim=2)

    def reset_noise(self):
        self.h[2].reset_noise()
        self.h[4].reset_noise()
