import numpy as np
import torch
import torch.nn.functional as F
from Utils import MemoryBuffer, Network, ExpLrDecay
from SuperHexagon import SuperHexagonInterface
from time import time
import pickle
import os


class Trainer:
    def __init__(
            self,
            capacity_per_level=500000,
            warmup_steps=50000,
            n_frames=4,
            n_atoms=21,
            v_min=-1,
            v_max=0,
            gamma=.99,
            hidden_size=512,
            device='cuda',
            batch_size=48,
            lr=0.0000625 * 2,
            lr_decay=0.985,
            update_target_net_every=16000,
            train_every=4,
            frame_skip=4,
            disable_noisy_after=2000000
    ):

        # training objects
        self.memory_buffer = MemoryBuffer(
            capacity_per_level,
            SuperHexagonInterface.n_levels,
            n_frames,
            SuperHexagonInterface.frame_size,
            SuperHexagonInterface.frame_size_cropped,
            gamma,
            device=device
        )
        self.net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms, hidden_size).to(device)
        self.target_net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms, hidden_size).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1.5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, ExpLrDecay(lr_decay, min_factor=.1))

        # parameters
        self.batch_size = batch_size
        self.update_target_net_every = update_target_net_every
        self.train_every = train_every
        self.frame_skip = frame_skip
        self.disable_noisy_after = disable_noisy_after
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.device = device

        # parameters for distributional
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float, device=device)
        self.offset = torch.arange(0, batch_size * n_atoms, n_atoms, device=device).view(-1, 1)
        self.m = torch.empty((batch_size, n_atoms), device=device)

        # debug and logging stuff
        self.list_steps_alive = [[] for _ in range(SuperHexagonInterface.n_levels)]
        self.longest_run = [(0, 0)] * SuperHexagonInterface.n_levels
        self.total_simulated_steps = [0] * SuperHexagonInterface.n_levels
        self.losses = []
        self.kls = []
        self.times = []
        self.iteration = 0

    def warmup(self, game, log_every):
        t = True
        for i in range(1, self.warmup_steps + 1):
            if i % log_every == 0:
                print('Warmup', i)
            if t:
                self.total_simulated_steps[game.level] += game.simulated_steps
                if self.total_simulated_steps[game.level] > self.total_simulated_steps[game.level - 1]:
                    game.select_level((game.level + 1) % 6)
                f, fc = game.reset()
                self.memory_buffer.insert_first(game.level, f, fc)
            a = np.random.randint(0, 3)
            (f, fc), r, t = game.step(a)
            self.memory_buffer.insert(game.level, a, r, t, f, fc)
        return t

    def train(
            self,
            save_every=50000,
            save_name='trainer',
            log_every=1000,
    ):

        game = SuperHexagonInterface(self.frame_skip)

        # if trainer was loaded, select the level that was played the least
        if any(x != 0 for x in self.total_simulated_steps):
            game.select_level(np.argmin(self.total_simulated_steps).item())

        # init state
        f, fc = np.zeros(game.frame_size, dtype=np.bool), np.zeros(game.frame_size_cropped, dtype=np.bool)
        sf, sfc = torch.zeros((1, 4, *game.frame_size), device=self.device), torch.zeros((1, 4, *game.frame_size_cropped), device=self.device)
        t = True

        # run warmup is necessary
        if self.iteration == 0:
            if os.path.exists('warmup_buffer.npz'):
                self.memory_buffer.load_warmup('warmup_buffer.npz')
            else:
                t = self.warmup(game, log_every)
                self.memory_buffer.save_warmup('warmup_buffer.npz')

        # trainings loop
        last_time = time()
        save_when_terminal = False
        while True:

            self.iteration += 1

            # disable noisy
            if self.iteration == self.disable_noisy_after:
                self.net.eval()
                self.target_net.eval()

            # log
            if self.iteration % log_every == 0 and all(len(l) > 0 for l in self.list_steps_alive):
                print(f'{self.iteration} | '
                      f'{[round(np.mean(np.array(l[-100:])[:, 1]) / 60, 2) for l in self.list_steps_alive]}s | '
                      f'{[round(r[1] / 60, 2) for r in self.longest_run]}s | '
                      f'{self.total_simulated_steps} | '
                      f'{time() - last_time:.2f}s | '
                      f'{np.mean(self.losses[-log_every:])} | '
                      f'{np.mean(self.kls[-log_every:])} | '
                      f'{self.lr_scheduler.get_last_lr()[0]} | '
                      f'{game.level}')

            # indicate that the trainer should be saved the next time the agent dies
            if self.iteration % save_every == 0:
                save_when_terminal = True

            # update target net
            if self.iteration % self.update_target_net_every == 0:
                self.lr_scheduler.step()
                self.target_net.load_state_dict(self.net.state_dict())

            # if terminal
            if t:
                # select next level if this level was played at least as long as the previous level
                if self.total_simulated_steps[game.level] > self.total_simulated_steps[game.level - 1]:
                    game.select_level((game.level + 1) % 6)
                f, fc = game.reset()
                self.memory_buffer.insert_first(game.level, f, fc)
                sf.zero_()
                sfc.zero_()

            # update state
            sf[0, 1:] = sf[0, :-1].clone()
            sfc[0, 1:] = sfc[0, :-1].clone()
            sf[0, 0] = torch.from_numpy(f).to(self.device)
            sfc[0, 0] = torch.from_numpy(fc).to(self.device)

            # train
            if self.iteration % self.train_every == 0:
                loss, kl = self.train_batch()
                self.losses.append(loss)
                self.kls.append(kl)

            # act
            with torch.no_grad():
                self.net.reset_noise()
                a = (self.net(sf, sfc) * self.support).sum(dim=2).argmax(dim=1).item()
            (f, fc), r, t = game.step(a)
            self.memory_buffer.insert(game.level, a, r, t, f, fc)

            # if terminal
            if t:
                if game.steps_alive > self.longest_run[game.level][1]:
                    self.longest_run[game.level] = (self.iteration, game.steps_alive)
                self.list_steps_alive[game.level].append((self.iteration, game.steps_alive))
                self.total_simulated_steps[game.level] += game.simulated_steps
                self.times.append(time() - last_time)

                if save_when_terminal:
                    print('saving...')
                    for _ in range(60):
                        game.game.step(False)
                    self.save(save_name)
                    for _ in range(60):
                        game.game.step(False)
                    save_when_terminal = False

    def train_batch(self):

        # sample minibatch
        f, fc, a, r, t, f1, fc1 = self.memory_buffer.make_batch(self.batch_size)

        # compute target q distribution
        with torch.no_grad():
            self.target_net.reset_noise()
            qdn = self.target_net(f1, fc1)
            an = (qdn * self.support).sum(dim=2).argmax(dim=1)

        Tz = (r.unsqueeze(1) + t.logical_not().unsqueeze(1) * self.gamma * self.support).clamp_(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) & (l == u)] -= 1
        u[(l == u)] += 1

        vdn = qdn.gather(1, an.view(-1, 1, 1).expand(self.batch_size, -1, self.n_atoms)).view(self.batch_size, self.n_atoms)
        self.m.zero_()
        self.m.view(-1).index_add_(0, (l + self.offset).view(-1), (vdn * (u - b)).view(-1))
        self.m.view(-1).index_add_(0, (u + self.offset).view(-1), (vdn * (b - l)).view(-1))

        # forward and backward pass
        qld = self.net(f, fc, log=True)
        vld = qld.gather(1, a.view(-1, 1, 1).expand(self.batch_size, -1, self.n_atoms)).view(self.batch_size, self.n_atoms)
        loss = -torch.sum(self.m * vld, dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        kl = F.kl_div(vld.detach(), self.m, reduction='batchmean')
        return loss.detach().item(), kl.item()

    def save(self, file_name='trainer'):

        # first backup the last save file
        # in case anything goes wrong
        file_name_backup = file_name + '_backup'
        if os.path.exists(file_name):
            os.rename(file_name, file_name_backup)

        # save this object
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

        # remove backup if nothing went wrong
        if os.path.exists(file_name_backup):
            os.remove(file_name_backup)

    @staticmethod
    def load(file_name='trainer'):
        with open(file_name, 'rb') as f:
            ret = pickle.load(f)
            assert ret.memory_buffer.last_was_terminal
            return ret


if __name__ == '__main__':

    save_name = 'trainer_full'
    load = os.path.exists(save_name)

    if load:
        trainer = Trainer.load(save_name)
    else:
        trainer = Trainer(
            capacity_per_level=500000,
            warmup_steps=100000,
            n_frames=4,
            n_atoms=51,
            v_min=-1,
            v_max=0,
            gamma=.99,
            hidden_size=1024,
            device='cuda',
            batch_size=48,
            lr=0.0000625 * 2,
            lr_decay=0.99,
            update_target_net_every=25000,
            train_every=6,
            frame_skip=4,
            disable_noisy_after=2000000
        )

    trainer.train(save_every=200000, save_name=save_name)
