import numpy as np
import torch
from utils import Network
from superhexagon import SuperHexagonInterface
from itertools import count


if __name__ == '__main__':

    # parameters
    level = 0
    device = 'cuda'
    net_path = 'super_hexagon_net'

    n_frames = 4
    frame_skip = 4
    log_every = 1000
    n_atoms = 51

    # setup
    fp, fcp = np.zeros((1, n_frames, *SuperHexagonInterface.frame_size), dtype=np.bool), np.zeros((1, n_frames, *SuperHexagonInterface.frame_size_cropped), dtype=np.bool)
    support = np.linspace(-1, 0, n_atoms)

    net = Network(n_frames, SuperHexagonInterface.n_actions, n_atoms).to(device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()

    game = SuperHexagonInterface(frame_skip=frame_skip, run_afap=False)
    game.select_level(level)

    list_times_alive = []
    f, fc = game.reset()

    # helper function
    def to_torch_tensor(x):
        return torch.from_numpy(x).to(device).float()

    # global no_grad
    torch.set_grad_enabled(False)

    # run actor
    for i in count(1):

        # log
        if i % log_every == 0 and list_times_alive:
            print(f'{i} {np.mean(list_times_alive[-100:]) / 60:.2f}s {np.max(list_times_alive) / 60:.2f}s')

        # update state
        fp[0, 1:] = fp[0, :3]
        fp[0, 0] = f
        fcp[0, 1:] = fcp[0, :3]
        fcp[0, 0] = fc

        # act
        action = np.sum((net(to_torch_tensor(fp), to_torch_tensor(fcp)).cpu().squeeze().numpy() * support), axis=1).argmax()
        (f, fc), _, terminal = game.step(action)

        if terminal:
            list_times_alive.append(game.steps_alive)
            fp[:] = 0
            fcp[:] = 0
            f, fc = game.reset()
