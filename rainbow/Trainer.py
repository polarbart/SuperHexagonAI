import numpy as np
import torch
import torch.nn.functional as F
from Utils import MemoryBuffer, Network, ExpLrDecay
from SuperHexagon import SuperHexagonInterface
import matplotlib.pyplot as plt
from itertools import count
from torch.multiprocessing import Process, Queue, set_start_method
from collections import deque
from time import time
import pickle
import shutil
import os


class Trainer:
    def __init__(
            self,
            capacity=500000,
            warmup_steps=50000,
            n_frames=4,
            n_steps=3,
            n_atoms=21,
            v_min=-1,
            v_max=0,
            alpha=.6,
            beta=.4,
            gamma=.99,
            hidden_size=512,
            device='cuda',
            batch_size=48,
            lr=0.0000625 * 2,
            lr_decay=0.985,
            beta_converged=4000000,
            update_target_net_every=16000,
            train_every=4,
            frame_skip=4
    ):
        self.memory_buffer = MemoryBuffer(capacity, n_frames, n_steps, SuperHexagonInterface.frame_size, SuperHexagonInterface.frame_size_cropped, alpha, beta, gamma, device=device)
        self.net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms, hidden_size).to(device)
        self.target_net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms, hidden_size).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1.5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, ExpLrDecay(lr_decay, min_factor=.1))
        self.batch_size = batch_size
        self.beta_converged = beta_converged
        self.update_target_net_every = update_target_net_every
        self.train_every = train_every
        self.frame_skip = frame_skip
        self.warmup_steps = warmup_steps
        self.n_steps = n_steps
        self.beta = beta
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float, device=device)
        self.iteration = 0
        self.list_steps_alive = []
        self.losses = []
        self.kls = []
        self.times = []
        self._offset = torch.arange(0, batch_size * n_atoms, n_atoms, device=device).view(-1, 1)
        self._m = torch.empty((batch_size, n_atoms), device=device)
        self._longest_run = 0

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reinit_priority_queue(self):
        self.memory_buffer.update_priorities(np.arange(self.memory_buffer.size, dtype=np.int), np.ones(self.memory_buffer.size) * self.memory_buffer.priority_queue.max_value, False)
        self.memory_buffer.priority_queue.recompute_tree()

    def warmup(self, game, log_every):
        t = True
        for i in range(1, self.warmup_steps + 1):
            if i % log_every == 0:
                print('Warmup', i)
            if t:
                f, fc = game.reset()
                self.memory_buffer.insert_first(f, fc)
            a = np.random.randint(0, 3)
            (f, fc), r, t = game.step(a)
            self.memory_buffer.insert(a, r, t, f, fc)
        self.memory_buffer.update_priorities(np.arange(self.warmup_steps - self.n_steps, dtype=np.int), np.ones(self.warmup_steps - self.n_steps) * np.log(self.n_atoms - 1), False)
        return t

    def train(
            self,
            save_every=50000,
            save_name='trainer',
            log_every=1000,
    ):

        save_when_terminal = False
        game = SuperHexagonInterface(self.frame_skip)

        f, fc = np.zeros(game.frame_size, dtype=np.bool), np.zeros(game.frame_size_cropped, dtype=np.bool)
        sf, sfc = np.zeros((1, 4, *game.frame_size), dtype=np.bool), np.zeros((1, 4, *game.frame_size_cropped), dtype=np.bool)
        t = True
        if self.iteration == 0:
            if os.path.exists('warmup_buffer_level3.npz'):
                self.memory_buffer.load_warmup('warmup_buffer_level3.npz')
            else:
                t = self.warmup(game, log_every)
                self.memory_buffer.save_warmup('warmup_buffer_level3.npz')

        last_time = time()

        while True:
            self.iteration += 1
            if self.iteration % log_every == 0 and len(self.list_steps_alive) > 0:
                print(f'{self.iteration} | '
                      f'{np.mean(self.list_steps_alive[-100:]) / 60:.2f}s | '
                      f'{self._longest_run / 60:.2f}s | '
                      f'{time() - last_time:.2f}s | '
                      f'{np.mean(self.losses[-log_every:])} | '
                      f'{np.mean(self.kls[-log_every:])} | '
                      f'{self.lr_scheduler.get_last_lr()[0]}')
                # last_time = time()
            if self.iteration % save_every == 0:
                save_when_terminal = True
            if self.iteration % self.update_target_net_every == 0:
                self.memory_buffer.beta = min(1., self.beta + (1 - self.beta) * self.iteration / self.beta_converged)
                self.lr_scheduler.step()
                self.target_net.load_state_dict(self.net.state_dict())

            if t:
                # game.recorder.start()
                f, fc = game.reset()
                self.memory_buffer.insert_first(f, fc)
                sf[:] = 0
                sfc[:] = 0

            sf[0, 1:] = sf[0, :-1]
            sfc[0, 1:] = sfc[0, :-1]
            sf[0, 0] = f
            sfc[0, 0] = fc

            if self.iteration % self.train_every == 0:
                a, loss, kl = self.train_and_get_action(True)
                self.losses.append(loss)
                self.kls.append(kl)
            else:
                with torch.no_grad():
                    self.net.reset_noise()
                    a = (self.net(torch.from_numpy(sf).cuda().float(), torch.from_numpy(sfc).cuda().float()) * self.support).sum(dim=2).argmax(dim=1).item()

            (f, fc), r, t = game.step(a)
            self.memory_buffer.insert(a, r, t, f, fc)

            if t:
                if game.steps_alive > self._longest_run:
                    self._longest_run = game.steps_alive
                # game.recorder.stop()
                # if game.steps_alive >= 60 * 60 and game.steps_alive > self._longest_run * .7:
                #     game.recorder.save(f'superhexagon_{int(time())}', 60)
                # game.recorder.start()
                self.list_steps_alive.append(game.steps_alive)
                self.times.append(time() - last_time)
                if save_when_terminal:
                    print('saving...')
                    self.memory_buffer.priority_queue.recompute_tree()
                    self.save(save_name)
                    save_when_terminal = False

    def train_and_get_action(self, renew_action):

        f, fc, a, r, t, f1, fc1, w, idx = self.memory_buffer.make_batch(self.batch_size - int(renew_action), include_last_insertion=renew_action)

        with torch.no_grad():
            self.target_net.reset_noise()
            self.net.reset_noise()
            an = (self.net(f1, fc1) * self.support).sum(dim=2).argmax(dim=1)
            qdn = self.target_net(f1, fc1)
        '''
        with torch.no_grad():
            torch.cuda.synchronize()
            with torch.cuda.stream(self.cuda_stream_1):
                self.target_net.reset_noise()
                qdn = self.target_net(f1, fc1)
            with torch.cuda.stream(self.cuda_stream_2):
                self.net.reset_noise()
                an = (self.net(f1, fc1) * self.support).sum(dim=2).argmax(dim=1)
            torch.cuda.synchronize()
        '''
        #if self.iteration <= self.update_target_net_every:
        #    Tz = r.unsqueeze(1).expand((self.batch_size, self.n_atoms)).clamp(self.v_min, self.v_max)
        #else:
        Tz = (r.unsqueeze(1) + t.logical_not().unsqueeze(1) * self.gamma**self.n_steps * self.support).clamp_(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) & (l == u)] -= 1
        u[(l == u)] += 1

        vdn = qdn.gather(1, an.view(-1, 1, 1).expand(self.batch_size, -1, self.n_atoms)).view(self.batch_size, self.n_atoms)
        self._m.zero_()
        self._m.view(-1).index_add_(0, (l + self._offset).view(-1), (vdn * (u - b)).view(-1))
        self._m.view(-1).index_add_(0, (u + self._offset).view(-1), (vdn * (b - l)).view(-1))

        qld = self.net(f, fc, log=True)
        vld = qld.gather(1, a.view(-1, 1, 1).expand(self.batch_size, -1, self.n_atoms)).view(self.batch_size, self.n_atoms)
        ce = -torch.sum(self._m * vld, dim=1)
        loss = (w * ce).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        kl = F.kl_div(vld.detach(), self._m, reduction='none').sum(dim=1).clamp(min=.001).cpu().numpy()
        self.memory_buffer.update_priorities(idx, kl, renew_action)

        return (qld[0].detach().exp() * self.support).sum(dim=1).argmax().item(), loss.detach().item(), kl.mean().item()

    def save(self, file_name='trainer'):

        file_name_backup = file_name + '_backup'
        if os.path.exists(file_name):
            os.rename(file_name, file_name_backup)

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

        if os.path.exists(file_name_backup):
            os.remove(file_name_backup)

    @staticmethod
    def load(file_name='trainer'):
        with open(file_name, 'rb') as f:
            ret = pickle.load(f)
            assert ret.memory_buffer.last_was_terminal
            return ret


if __name__ == '__main__':
    load = False

    '''
    TODO
    -Better network + zeitmessung () check. more layers bigger filters and faster 
    -is binarization of frame correct? check
    super hexagon ai googlen
    -data augmentation check 
    test dataaugmentation and new net
    load save warmup 
    
    maybe later
    visualization of feature maps
    - does it detect the player and the barriers
    '''

    if load:
        trainer = Trainer.load('trainer')
    else:
        # trainer = Trainer(capacity=10000, lr_decay=.999, beta_converged=8000000, warmup_steps=100, update_target_net_every=1000)
        trainer = Trainer(
            capacity=1000000,
            warmup_steps=100000,
            n_frames=4,
            n_steps=1,
            n_atoms=51,
            v_min=-1,
            v_max=0,
            alpha=.5,
            beta=.4,
            gamma=.99,
            hidden_size=512,
            device='cuda',
            batch_size=48,
            lr=0.0000625 * 2,
            lr_decay=0.99,
            beta_converged=2000000,
            update_target_net_every=16000,
            train_every=4,
            frame_skip=4
        )

    trainer.train(save_every=200000, save_name='trainer_3')
