""""""

import numpy as np
import time
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp
import warnings

warnings.filterwarnings("ignore")
assert pyro.__version__.startswith("1.7.0")
torch.set_printoptions(precision=6, linewidth=120)

# Why is uniform sampling for candidate better than normal sampling? (probs not true)
# Normal sampling better for pre selected data
# Gradient based optimization with random starting point runs into borders, if not yet
# tested. GP needs data in that area. Pre selected data outside range might(?) help.
# Idea is to use appr. distribution -> draw many samples in aquisition func
# time: 20-40s for 1 map eval, 200-300s for 1 GP update, 1-3s for 1 GP eval

# TODO: Add error handling (RuntimeError) to remove last datapoint of adam fails
# TODO: None fit handling: adjust start mu val, stop entire map if one is None
# TODO: end early if point repeated twice (if random component, maybe not)
# TODO: try different kernels
# TODO: acquisition function too simple
# TODO: EI gets stuck, util value is -0.?
# TODO: Check GP train loss (how good is approx?)
# TODO: check noise EI - averages over lowest known value?
# TODO: MC integrate EI?


def const_weights(weights: np.array):
    w_list = [1.0] * 16

    if weights.shape[0] == 4:
        for i in range(4):
            w_list[(i * 2)] = weights[i]
            w_list[(i * 2 + 1)] = weights[i]
    elif weights.shape[0] == 8:
        for i in range(8):
            w_list[i] = weights[i]
    else:
        assert False, "ERROR: Invalid weight array length. Needs to be 4 or 8."

    return np.array(w_list)


def main(
    n_start_data: int = 20,
    n_opt_steps: int = 10,
    n_candidates: int = 10,
    dim: int = 8,
    w_range: np.array = np.array([0.9, 1.1]),
    b_dummy_val: bool = False,
):
    pos, evs = scan_200117()
    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        detector=0,
    )

    pyro.clear_param_store()
    # Initialize with random data points
    x = []
    y = []
    for i in range(n_start_data):
        # rng_weights = np.random.rand(4) * (w_range[1] - w_range[0]) + w_range[0]
        rng_weights = 0.05 * np.random.randn(dim) + 1.0
        weights = const_weights(rng_weights)

        if b_dummy_val:
            losses = [0.0, np.random.rand(1) * 30000 + 5000]
        else:
            smc.calc_peak_positions(weights=weights)
            losses = smc.calc_loss()

        if losses[1] is not None:
            x.append(rng_weights)
            y.append(losses[1])

    x = torch.tensor(x)
    y = torch.flatten(torch.tensor(y))

    print(x)
    print(y)

    gpmodel = gp.models.GPRegression(
        x, y, gp.kernels.Matern52(input_dim=dim), noise=torch.tensor(0.1), jitter=1.0e-4
    )

    def update_posterior(x_new):
        if b_dummy_val:
            losses = [0.0, np.random.rand(1) * 30000 + 5000]
        else:
            new_weights = const_weights(torch.flatten(x_new))
            smc.calc_peak_positions(weights=np.array(new_weights))
            losses = smc.calc_loss()

        if losses[1] is not None:
            y = torch.tensor([losses[1]])
            print("Curr losses ", losses)

            x = torch.cat([gpmodel.X, x_new])  # incorporate new evaluation
            y = torch.cat([gpmodel.y, y])

            print("last 5 x ", x[-5:])
            print("last 5 y ", y[-5:])

            gpmodel.set_data(x, y)
            # optimize the GP hyperparameters using Adam with lr=0.001
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
            gp.util.train(gpmodel, optimizer)

    def lower_confidence_bound(x_in, kappa=3.0):
        mu, variance = gpmodel(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def prob_of_improvement(x_in, kappa):
        mu, variance = gpmodel(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        argmin = torch.min(gpmodel.y, dim=0)[1].item()
        mu_min = gpmodel.y[argmin]
        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        return n_dist.cdf((mu - mu_min - kappa) / sigma)

    def expected_improvement(x_in, kappa=1.0):
        """"""
        mu, variance = gpmodel(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        argmin = torch.min(gpmodel.y, dim=0)[1].item()
        mu_min = gpmodel.y[argmin]
        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        # gamma = (mu - mu_min - kappa) / sigma
        gamma = (mu_min - mu + kappa) / sigma
        return -(
            sigma * (gamma * n_dist.cdf(gamma) + torch.exp(n_dist.log_prob(gamma)))
        )

    def find_a_candidate(x_init, lower_bound=w_range[0], upper_bound=w_range[1]):
        # transform x to an unconstrained domain
        constraint = constraints.interval(lower_bound, upper_bound)
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn="strong_wolfe")

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = expected_improvement(x)
            # y = lower_confidence_bound(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x2 = transform_to(constraint)(unconstrained_x)
        return x2.detach()

    def next_x(
        lower_bound=w_range[0], upper_bound=w_range[1], num_candidates=n_candidates
    ):
        candidates = []
        values = []

        # Start with best candidate x and sample rest random
        argmin = torch.min(gpmodel.y, dim=0)[1].item()
        x_init = torch.unsqueeze(gpmodel.X[argmin], 0)
        print("Min x / y \t", gpmodel.X[argmin], gpmodel.y[argmin])
        for i in range(num_candidates):
            x = find_a_candidate(x_init, lower_bound, upper_bound)
            y = expected_improvement(x)
            # y = lower_confidence_bound(x)
            candidates.append(x)
            values.append(y)
            # x_init = x.new_empty((1, dim)).uniform_(lower_bound, upper_bound)
            x_init = x.new_empty((1, dim)).normal_(1.0, 0.05)
        # Use minimum (best) result
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        print(f"winner: {candidates[argmin]}, LBO: {values[argmin]}")
        return candidates[argmin]

    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)
    for i in range(n_opt_steps):
        print(f"Current optimizer step:\t{i + 1}/{n_opt_steps}")
        start_time = time.time()
        x_min = next_x()
        end_time = time.time()
        print(f"Avg time per candidate: {(end_time - start_time)/n_candidates:0.3f}")
        start_time = time.time()
        update_posterior(x_min)
        end_time = time.time()
        print(f"Time for update: {(end_time - start_time):0.3f}")


if __name__ == "__main__":
    main(
        b_dummy_val=False,
        n_start_data=50,
        n_opt_steps=70,
        n_candidates=50,
        dim=8,
    )

# [1.0040, 0.9734, 0.9903, 0.9905] (6567, 7839)
# [1.0113, 0.9991, 0.9559, 0.9794, 0.9962, 0.9883, 1.0002, 0.9860] (6359.8, 7066.2)
# [0.9979, 0.9948, 0.9510, 0.9732, 1.0052, 0.9930, 1.0038, 0.9920] (6318.1, 6945.0)

# EI runs
# kappa 0
# [0.992950, 0.955659, 1.000981, 1.003407] (6464, 7499)
# [1.010134, 1.002948, 0.964882, 0.988906, 0.990236, 0.990926, 0.992734, 0.978815] (6408, 7226)
# kappa 1 / 50 50 100
# [0.995389, 0.957945, 0.999247, 1.001300] (6457, 7439)
# [0.989336, 0.983152, 0.936838, 0.963596, 1.012520, 1.003641, 1.013249, 1.003278] (6293, 7156)
# kappa 1 / 20 50(45) 100
# [0.993375, 0.956549, 1.000868, 1.002985] (6455, 7483)
# [0.998838, 0.959562, 0.997056, 0.999847] (6457, 7434)
# [0.992441, 0.954160, 1.001888, 1.005544] (6474, 7478)
# does not suffice for 8 dim! (reaches about 11k)
# kappa 1 / 50 50 20
# [1.007533, 1.003166, 0.959988, 0.980420, 0.991973, 0.991907, 0.998605, 0.983629] (6371, 7022)
# [0.960330, 1.010716, 0.959088, 0.963435, 1.030832, 0.989064, 0.986494, 1.004305] (7211, 9411)
# kappa 1 / 50 70 20
# [1.030843, 1.006677, 0.961154, 0.999453, 0.987405, 0.977974, 0.986383, 0.977584] (6476, 7256)
# [0.972688, 0.952187, 0.900349, 0.954270, 1.057235, 1.008276, 1.012375, 1.040411] (7202, 8681)
# kappa 1 / 50 70 50
# [0.998394, 0.991246, 0.948916, 0.972422, 1.007636, 0.993730, 1.003973, 0.994475] (6310, 6961)
# [0.989509, 0.991025, 0.949082, 0.966628, 1.013632, 0.999015, 1.009884, 0.993862] (6457, 7169)
