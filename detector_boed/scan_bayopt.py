""""""

# TODO: add exception: RuntimeError: torch.linalg.cholesky: U(70,70) is zero, singular U.

import warnings
import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

from base_scanopt import BaseScanOpt

warnings.filterwarnings("ignore")
assert pyro.__version__.startswith("1.7.0")
torch.set_printoptions(precision=6, linewidth=120)


class ScanBayOpt(BaseScanOpt):
    """"""

    def __init__(
        self,
        scan_map_class: ScanMapClass,
        weight_dim: int = 4,
        weight_range: np.array = np.array([0.9, 1.1]),
        detector: int = 0,
    ):
        super().__init__(
            scan_map_class=scan_map_class,
            weight_dim=weight_dim,
            weight_range=weight_range,
            detector=detector,
            opt_label="BayOpt",
        )

        self._gp_model = None

    def optimize(
        self,
        n_start_data: int = 50,
        n_opt_steps: int = 80,
        n_candidates: int = 50,
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
            losses = self.calc_losses(torch.flatten(x_new))

            if losses[1] is not None:
                y = torch.tensor([losses[1]])
                x = torch.cat([self._gp_model.X, x_new])
                y = torch.cat([self._gp_model.y, y])
                self._gp_model.set_data(x, y)

        if x_new is None or losses[1] is not None:
            optimizer = torch.optim.Adam(self._gp_model.parameters(), lr=0.001)
            gp.util.train(self._gp_model, optimizer)

    def _next_x(self, n_candidates: int):
        """"""

        candidates = []
        values = []

        # Start with best candidate x and sample rest random
        x_seed = torch.unsqueeze(torch.tensor(self.optimum["x_opt"]), 0)
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
            x_seed = torch.unsqueeze(torch.tensor(self.optimum["x_opt"]), 0)
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
            rng_weights = self._get_random_weights()

            losses = self.calc_losses(rng_weights)
            if losses[1] is not None:
                x.append(rng_weights)
                y.append(losses[1])

        x = torch.tensor(x)
        y = torch.flatten(torch.tensor(y))

        return x, y

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
    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=0,
    )

    sbo = ScanBayOpt(scan_map_class=smc, weight_dim=8, detector=smc.detector)
    result = sbo.optimize(
        n_start_data=(2*8),
        n_opt_steps=(500 - 2*8),
        n_candidates=50,
    )
    sbo.plot_history()
    sbo.save_history()

    print("Best optimization result: ", result)
    best_weights = sbo.construct_weights(result["x_opt"], smc.detector)
    smc.calc_peak_positions(best_weights)
    smc.calc_loss()
    smc.plot_scanmap()


if __name__ == "__main__":
    main()
