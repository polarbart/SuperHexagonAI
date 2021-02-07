"""
slightly more advanced eval script used for the video https://youtu.be/gjqSZ_4mQjg
1. it uses three networks which vote on the best action
2. it computes the best action at every step instead of every four steps
"""

import numpy as np
import torch
from utils import Network
from superhexagon import SuperHexagonInterface
from itertools import count
from time import time


@torch.no_grad()
def get_action(f, fc):

    def to_torch_tensor(x):
        return torch.from_numpy(x).to(device).float()

    f, fc = to_torch_tensor(f), to_torch_tensor(fc)

    a = [0, 0, 0]
    for n in nets:
        q = n(f, fc).cpu().squeeze().numpy()
        action = np.sum((q * support), axis=1).argmax()
        a[action] += 1
    return np.argmax(a)


n_frames = 4
log_every = 1000
n_atoms = 51

device = 'cuda'

fp, fcp = np.zeros((4, 1, 4, *SuperHexagonInterface.frame_size), dtype=np.bool), np.zeros((4, 1, 4, *SuperHexagonInterface.frame_size_cropped), dtype=np.bool)
support = np.linspace(-1, 0, n_atoms)

nets = []
for i in range(3):
    net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms).to(device)
    net.load_state_dict(torch.load(f'net{i+1}', map_location=device))
    net.eval()
    nets.append(net)

get_action(fp[0], fcp[0])

game = SuperHexagonInterface(1, run_afap=False, allow_game_restart=True)
game.select_level(0)

list_times_alive = []
f, fc = game.reset()

last_action = 0
for i in count(1):
    last_time = time()
    if i % log_every == 0 and list_times_alive:
        print(f'{i} {np.mean(list_times_alive[-100:]) / 60:.2f}s {np.max(list_times_alive) / 60:.2f}s')

    n = i % 4

    fp[n, 0, 1:] = fp[n, 0, :3]
    fp[n, 0, 0] = f
    fcp[n, 0, 1:] = fcp[n, 0, :3]
    fcp[n, 0, 0] = fc

    action = get_action(fp[n], fcp[n])
    (f, fc), _, t = game.step(action)

    if t:
        list_times_alive.append(game.steps_alive)
        fp[:] = 0
        fcp[:] = 0
        for _ in range(240):
            game.game.step(False)
        game._restart_game()
        for _ in range(120):
            game.game.step(False)
        f, fc = game.reset()
