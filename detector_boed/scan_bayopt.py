""""""
# TODO: Dummy loss vals
# TODO: add exception: RuntimeError: torch.linalg.cholesky: U(70,70) is zero, singular U.
# TODO: do time.time() eval

import warnings
import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200118


warnings.filterwarnings("ignore")
assert pyro.__version__.startswith("1.7.0")
torch.set_printoptions(precision=6, linewidth=120)


class ScanBayOpt:
    """"""

    def __init__(
        self,
        scan_map_class: ScanMapClass,
        weight_dim: int = 4,
        weight_range: np.array = np.array([0.9, 1.1]),
        detector: int = 0,
    ):
        self._w_dim = weight_dim
        self._w_range = weight_range
        self._smc = scan_map_class
        self._detector = detector

        self._gp_model = None
        self.optimum = None

    def __repr__(self):
        return f"BayOpt_det{self._detector}_dim{self._w_dim}_{self._smc.label}"

    def __str__(self):
        return f"BayOpt_det{self._detector}_dim{self._w_dim}_{self._smc.label}"

    def optimize(
        self,
        n_start_data: int = 20,
        n_opt_steps: int = 10,
        n_candidates: int = 10,
    ):
        """"""

        pyro.clear_param_store()
        x_init_, y_init = self._init_w_rndm_data(n_start_data)

        self._gp_model = gp.models.GPRegression(
            x_init_,
            y_init,
            gp.kernels.Matern52(input_dim=self._w_dim),
            noise=torch.tensor(0.1),
            jitter=1.0e-4,
        )

        self._update_posterior()
        for i in range(n_opt_steps):
            print(f"Current optimizer step:\t{i + 1}/{n_opt_steps}")
            x_min = self._next_x(n_candidates)
            self._update_posterior(x_min)

        return self.optimum

    def _update_posterior(self, x_new: torch.tensor = None):
        """"""

        if x_new is not None:
            new_weights = self.construct_weights(torch.flatten(x_new), self._detector)
            self._smc.calc_peak_positions(weights=np.array(new_weights))
            losses = self._smc.calc_loss()

            if losses[1] is not None:
                y = torch.tensor([losses[1]])
                print("Curr losses ", losses)

                x = torch.cat([self._gp_model.X, x_new])
                y = torch.cat([self._gp_model.y, y])
                self._gp_model.set_data(x, y)
                print("last 5 x ", x[-5:])
                print("last 5 y ", y[-5:])

        if x_new is None or losses[1] is not None:
            optimizer = torch.optim.Adam(self._gp_model.parameters(), lr=0.001)
            gp.util.train(self._gp_model, optimizer)

        argmin = torch.min(self._gp_model.y, dim=0)[1].item()
        self.optimum = {
            "x_opt": self._gp_model.X[argmin],
            "y_opt": self._gp_model.y[argmin],
        }
        print(f"Curr optimum: {self.optimum}")

    def _next_x(self, n_candidates: int):
        """"""

        candidates = []
        values = []

        # Start with best candidate x and sample rest random
        x_seed = torch.unsqueeze(self.optimum["x_opt"], 0)
        for i in range(n_candidates):
            x = self._find_candidate(x_seed, self._expected_improvement)
            y = self._expected_improvement(self._gp_model, x)
            candidates.append(x)
            values.append(y)
            # x_init = x.new_empty((1, dim)).uniform_(lower_bound, upper_bound)
            x_seed = x.new_empty((1, self._w_dim)).normal_(1.0, 0.05)

        # Use minimum (best) result
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        print(f"x_new: {candidates[argmin]}, Util_val: {values[argmin]}")

        if values[argmin] > -(10 ** -5):
            candidates = []
            values = []
            print(f"Using lower confidence bound instead")
            x_seed = torch.unsqueeze(self.optimum["x_opt"], 0)
            for i in range(10):
                x = self._find_candidate(x_seed, self._lower_confidence_bound)
                y = self._lower_confidence_bound(self._gp_model, x)
                candidates.append(x)
                values.append(y)
                x_seed = x.new_empty((1, self._w_dim)).normal_(1.0, 0.05)
            argmin = torch.min(torch.cat(values), dim=0)[1].item()
            print(f"x_new: {candidates[argmin]}, Util_val: {values[argmin]}")

        return candidates[argmin]

    def _find_candidate(self, x_seed, acqu_func):
        """"""

        # transform x to an unconstrained domain
        constraint = constraints.interval(self._w_range[0], self._w_range[1])
        unconstrained_x_init = transform_to(constraint).inv(x_seed)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn="strong_wolfe")

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = acqu_func(self._gp_model, x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # convert it back to original domain.
        x2 = transform_to(constraint)(unconstrained_x)
        return x2.detach()

    def _init_w_rndm_data(self, n_start_data: int):
        """"""

        x = []
        y = []
        for i in range(n_start_data):
            # rng_weights = np.random.rand(4) * (w_range[1] - w_range[0]) + w_range[0]
            rng_weights = 0.05 * np.random.randn(self._w_dim) + 1.0
            weights = self.construct_weights(rng_weights, self._detector)

            self._smc.calc_peak_positions(weights=weights)
            losses = self._smc.calc_loss()

            if losses[1] is not None:
                x.append(rng_weights)
                y.append(losses[1])

        x = torch.tensor(x)
        y = torch.flatten(torch.tensor(y))

        return x, y

    @staticmethod
    def construct_weights(weights: np.array, detector: int = 0):
        """"""

        w_list = [1.0] * 16

        if weights.shape[0] == 4:
            for i in range(4):
                w_list[(8 * detector + i * 2)] = weights[i]
                w_list[(8 * detector + i * 2 + 1)] = weights[i]
        elif weights.shape[0] == 8:
            for i in range(8):
                w_list[(8 * detector + i)] = weights[i]
        else:
            assert False, "ERROR: Invalid weight array length. Needs to be 4 or 8."

        return np.array(w_list)

    @staticmethod
    def _lower_confidence_bound(gp_model, x_in, kappa=3.0):
        mu, variance = gp_model(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    @staticmethod
    def _prob_of_improvement(gp_model, x_in, kappa):
        mu, variance = gp_model(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        argmin = torch.min(gp_model.y, dim=0)[1].item()
        mu_min = gp_model.y[argmin]
        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        return n_dist.cdf((mu - mu_min - kappa) / sigma)

    @staticmethod
    def _expected_improvement(gp_model, x_in, kappa=1.0):
        """"""
        mu, variance = gp_model(x_in, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        argmin = torch.min(gp_model.y, dim=0)[1].item()
        mu_min = gp_model.y[argmin]
        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        # gamma = (mu - mu_min - kappa) / sigma
        gamma = (mu_min - mu + kappa) / sigma
        return -(
            sigma * (gamma * n_dist.cdf(gamma) + torch.exp(n_dist.log_prob(gamma)))
        )


def main():
    pos, evs = scan_200118()
    # pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200118.label,
        detector=1,
    )

    sbo = ScanBayOpt(scan_map_class=smc, weight_dim=8, detector=smc.detector)
    result = sbo.optimize(
        n_start_data=50,
        n_opt_steps=80,
        n_candidates=50,
    )

    print("Best optimization result: ", result)
    best_weights = sbo.construct_weights(result["x_opt"], smc.detector)
    smc.calc_peak_positions(best_weights)
    smc.plot_scanmap()


if __name__ == "__main__":
    main()

# EI + LCB
# 17.01.20 kappa 1 + 3 / 20 50 50
# [0.988395, 0.952541, 1.004483, 1.006872] (6465, 7532)
# kappa 1 + 3 / 20 70 50
# [0.992061, 0.949986, 1.003158, 1.009104] (6494, 7607)
# [0.996902, 0.959449, 0.998016, 1.000479] (6468, 7472) [20 25 50]
# kappa 1 + 3 / 50 70 50
# [0.987705, 0.983427, 0.944135, 0.963306, 1.016174, 0.999343, 1.011421, 1.000975] (6294, 7014)
# [1.004935, 0.996130, 0.953226, 0.974597, 1.001476, 0.991785, 1.000493, 0.992223] (6322, 6995)
# det1
# [0.933810, 0.911781, 0.948395, 0.943526, 1.039428, 1.002044, 1.015895, 1.021173] (12823, 13297)
# [0.902457, 0.920994, 0.945084, 0.928535, 1.038595, 1.013404, 1.023363, 1.025052] (12910, 13559)

# 19.01.20 kappa 1 + 3 / 50 70 50
# det0
# [1.016379, 0.992181, 0.952013, 0.973492, 0.991813, 0.993097, 0.997753, 0.994426] (6887, 7334)
# [0.989955, 0.980589, 0.941429, 0.959186, 1.007393, 1.005110, 1.008838, 1.005848] (6808, 7381)
# det1
# [0.900495, 0.908756, 0.932351, 0.924984, 1.043766, 1.015699, 1.036748, 1.016810] (13182, 13667) 50 46 50
# [0.901998, 0.904251, 0.935012, 0.915880, 1.055232, 1.008977, 1.032006, 1.026725] (13001, 13302)
# [0.925773, 0.920650, 0.950969, 0.945447, 1.042692, 0.998446, 1.017786, 1.014351] (13024, 13273)
# [0.922398, 0.901055, 0.930234, 0.942136, 1.054460, 1.000960, 1.020886, 1.025909] (13086, 13387)

# 14.01.20 kappa 1 + 3 / 50 70 50
# det0
# [1.029311, 1.016668, 0.967947, 0.989383, 0.974225, 0.994178, 0.993779, 0.987306] (6438, 7867)
# [1.020390, 1.001549, 0.951786, 0.980865, 0.988492, 1.000348, 0.996493, 1.003737] (6264, 7768)
# det1
# does not work!

# 16.01.20 kappa 1 + 3 / 50 70 50
# det 0
# [0.986570, 1.003502, 0.946171, 0.984044, 0.999282, 1.002169, 0.998284, 0.987721] (6821, 7492)
# [0.965442, 0.992094, 0.944052, 0.963702, 1.014197, 1.010516, 1.010200, 0.996825] (6954, 7736)
# det 1
# [0.906239, 0.900649, 0.929862, 0.912442, 1.052825, 1.010970, 1.025709, 1.036958] (12732, 13484)
# [0.959239, 0.943674, 0.968899, 0.953212, 1.018763, 0.987511, 1.006586, 1.005496] (12804, 13081)

# 20.01.20 kappa 1 + 3 / 50 70 50
# det 0
# [0.981491, 1.000165, 0.940416, 0.972391, 1.009283, 0.998123, 1.007426, 1.001705] (6006, 6680)
# [1.052837, 1.029538, 0.969125, 1.023406, 0.964535, 0.953276, 0.971233, 0.973692] (6926, 82619)
# det 1
# [0.903944, 0.902108, 0.934250, 0.924144, 1.045416, 1.008397, 1.025817, 1.026226] (13065, 13815)
# [0.929096, 0.915618, 0.948542, 0.943670, 1.033680, 0.997584, 1.016141, 1.015991] (13149, 13650)

# 21.01.20 kappa 1 + 3 / 50 70 50
# det 0
# [0.984937, 0.994228, 0.950598, 0.972297, 1.009479, 0.996995, 1.000693, 0.996841] (6729, 7385) 50 52 50
# [1.007895, 1.003504, 0.959532, 0.988985, 0.998333, 0.982249, 0.989087, 0.986529] (6782, 7465)
# det 1
# [0.900256, 0.900088, 0.926992, 0.919984, 1.053791, 1.008974, 1.027240, 1.031226] (13107, 13848)
# [0.922084, 0.931532, 0.951370, 0.942273, 1.036510, 0.996205, 1.014771, 1.017764] (13159, 13523)

# 22.01.20 kappa 1 + 3 / 50 80 50
# det 0
# [0.954109, 0.975352, 0.928182, 0.954187, 1.030113, 1.011730, 1.019102, 1.007643] (6666, 7548) 50 47 50
# [0.977252, 1.000225, 0.951456, 0.966188, 1.002773, 0.998498, 1.008772, 0.986460] (6794, 7396)
# 22.01.20 kappa 1 + 3 / 50 80 50
# det 1
# [0.929315, 0.927203, 0.948378, 0.956256, 1.031227, 0.996117, 1.011635, 1.015677] (13219, 13876)
# [0.935565, 0.916760, 0.939337, 0.961515, 1.033559, 0.998639, 1.013228, 1.018845] (13234, 13933)

# 18.01.20 kappa 1 + 3 / 50 80 50
# det 0
# [1.001412, 1.003598, 0.955507, 0.964964, 0.989094, 0.996220, 1.004455, 0.988604] (5814, 6399)
# [1.013583, 1.004575, 0.958183, 0.972812, 0.986177, 0.989460, 0.997237, 0.986930] (5811, 6385)
# det 1
#
#
