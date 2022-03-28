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
from panter.config.filesScanMaps import scan_200117, scan_200118

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
        bdummy_values: bool = False,
    ):
        super().__init__(
            scan_map_class=scan_map_class,
            weight_dim=weight_dim,
            weight_range=weight_range,
            detector=detector,
            opt_label="BayOpt",
            bdummy_values=bdummy_values,
        )

        self._gp_model = None

    def optimize(
        self,
        n_start_data: int = 50,
        n_opt_steps: int = 80,
        n_candidates: int = 50,
        start_x: torch.tensor = None,
        start_y: torch.tensor = None,
    ):
        """"""

        pyro.clear_param_store()
        x_init, y_init = self._init_w_rndm_data(n_start_data)
        # x_imp = torch.load("gpr_xdata.pt")
        # y_imp = torch.load("gpr_ydata.pt")
        #
        # x_init = torch.cat([x_init, x_imp])
        # y_init = torch.cat([y_init, y_imp])

        self._gp_model = gp.models.GPRegression(
            x_init,
            y_init,
            gp.kernels.Matern52(input_dim=self._w_dim),
            noise=torch.tensor(0.1),
            jitter=1.0e-4,
        )

        if start_x is not None and start_y is not None:
            self.add_data(start_x, start_y)

        self._update_posterior()
        for i in range(n_opt_steps):
            print(f"Current optimizer step:\t{i + 1}/{n_opt_steps}")
            x_min = self._next_x(n_candidates)
            self._update_posterior(x_min)

        return self.optimum

    def _update_posterior(self, x_new: torch.tensor = None):
        """"""

        try:
            if x_new is not None:
                losses = self.calc_losses(torch.flatten(x_new))

                if losses[1] is not None:
                    y = torch.tensor([losses[1]])
                    x = torch.cat([self._gp_model.X, x_new])
                    y = torch.cat([self._gp_model.y, y])
                    self._gp_model.set_data(x, y)

                    # print("Storing data")
                    # torch.save(x, "gpr_xdata.pt")
                    # torch.save(y, "gpr_ydata.pt")

            if x_new is None or losses[1] is not None:
                # try RuntimeError
                optimizer = torch.optim.Adam(self._gp_model.parameters(), lr=0.001)
                gp.util.train(self._gp_model, optimizer)
        except RuntimeError:
            print("Cholesky Error. Re-instantiating GP")
            # rand_ind = torch.randint(0, self._gp_model.X.shape[0])
            x = self._gp_model.X
            y = self._gp_model.y
            self._gp_model = gp.models.GPRegression(
                x,
                y,
                gp.kernels.Matern52(input_dim=self._w_dim),
                noise=torch.tensor(0.1),
                jitter=1.0e-4,
            )
            self._update_posterior()

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

    def add_data(self, x_in, y_in):
        """"""

        x = torch.cat([self._gp_model.X, x_in])
        y = torch.cat([self._gp_model.y, y_in])
        self._gp_model.set_data(x, y)



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

    def _acqu_decorator(aqu_func):
        def prep(*args, **kwargs):
            gpm = args[0]
            x = args[1]
            mu, variance = gpm(x, full_cov=False, noiseless=False)
            sigma = variance.sqrt()
            argmin = torch.min(gpm.y, dim=0)[1].item()
            mu_min = gpm.y[argmin]
            return aqu_func(*args, **kwargs, mu=mu, sigma=sigma, mu_min=mu_min)

        return prep

    @staticmethod
    @_acqu_decorator
    def _lower_confidence_bound(gp_model, x_in, mu, sigma, mu_min=None, kappa=3.0):
        """"""
        return mu - kappa * sigma

    @staticmethod
    @_acqu_decorator
    def _prob_of_improvement(gp_model, x_in, mu, sigma, mu_min, kappa):
        """"""
        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        return n_dist.cdf((mu - mu_min - kappa) / sigma)

    @staticmethod
    @_acqu_decorator
    def _expected_improvement(gp_model, x_in, mu, sigma, mu_min, kappa=1.0):
        """"""

        n_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        # gamma = (mu - mu_min - kappa) / sigma
        gamma = (mu_min - mu + kappa) / sigma
        return -(
            sigma * (gamma * n_dist.cdf(gamma) + torch.exp(n_dist.log_prob(gamma)))
        )


def main():

    # start_x = torch.tensor(
    #     [
    #         [0.971824, 0.947031, 0.919466, 0.912347, 1.013404, 1.037333, 1.1, 1.017283],
    #         [0.855084, 1.007138, 0.88301, 1.009697, 1.104118, 1.024755, 1.042399,
    #          0.992933],
    #         [0.963974, 0.979819, 0.917613, 0.957483, 1.011557, 1.03814, 1.066389,
    #          1.005038],
    #         [0.923633, 0.954014, 0.910152, 0.969097, 1.061233, 1.061396, 1.026949,
    #          1.018377],
    #         [0.997179, 1.011509, 0.914966, 0.954833, 1.001574, 1.014533, 1.030541,
    #          1.026458],
    #         [0.998522, 1.021792, 0.949858, 0.919018, 1.011224, 0.991306, 1.030706,
    #          1.034205],
    #         [0.974622, 0.977984, 0.904965, 0.95958, 0.99712, 1.044262, 1.058482,
    #          1.019763],
    #         [0.972942, 0.99892, 0.920493, 0.944149, 1.015637, 1.027048, 1.051522,
    #          1.034644],
    #         [0.951984, 0.974236, 0.905264, 0.949583, 1.020036, 1.055654, 1.060328,
    #          1.02599],
    #         [0.963756, 0.989239, 0.915349, 0.948434, 1.015645, 1.036962, 1.056991,
    #          1.026281],
    #         [1.01852, 0.992802, 0.952419, 0.950101, 0.997164, 0.995349, 1.042086,
    #          1.004908],
    #         [1.061688, 1.00633, 0.963394, 0.951178, 0.957908, 0.986341, 1.032956,
    #          1.007515],
    #         [1.013405, 1.003819, 0.933614, 0.938359, 0.997237, 1.00559, 1.046283,
    #          1.015609],
    #         [1.053141, 1.000046, 0.959638, 0.936576, 0.967539, 0.993334, 1.046417,
    #          1.003358],
    #         [1.019171, 1.005466, 0.941418, 0.933785, 0.992763, 0.998959, 1.046592,
    #          1.012433],
    #         [0.990130, 1.005051, 0.919249, 0.940087, 1.005116, 1.018327, 1.044417,
    #          1.016039],
    #         [0.986912, 1.020239, 0.946554, 0.955842, 0.994931, 1.004388, 1.038196,
    #          1.000945],
    #         [1.018768, 1.034834, 0.964188, 0.961707, 0.965186, 0.997062, 1.020644,
    #          0.997114],
    #         [1.002249, 1.016963, 0.938807, 0.945932, 0.997153, 1.003167, 1.039167,
    #          1.007208],
    #         [1.004579, 1.013662, 0.941514, 0.947196, 0.993279, 1.003073, 1.038321,
    #          1.006907],
    #     ]
    # )
    # start_y = torch.tensor(
    #     [
    #         4075.9394634595665,
    #         5774.826759144951,
    #         3285.4448003377656,
    #         4481.049430870333,
    #         3005.287809050258,
    #         3268.150676664713,
    #         3557.8216208759527,
    #         4056.885121814777,
    #         3675.9679754378158,
    #         3491.595511649137,
    #         2762.2964614052116,
    #         3063.4246399926224,
    #         2824.0531987495237,
    #         3117.5341544698617,
    #         2760.169897484338,
    #         2964.504712055655,
    #         2504.8213482557885,
    #         2526.788822861071,
    #         2640.9642359081136,
    #         2602.9634294334714,
    #     ]
    # )

    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=1,
    )

    DIM = 8
    sbo = ScanBayOpt(
        scan_map_class=smc,
        weight_dim=DIM,
        detector=smc.detector,
        bdummy_values=False,
        # weight_range=np.array([0.85, 1.15]),
    )
    result = sbo.optimize(
        n_start_data=(2 * DIM),
        # n_start_data=(2),
        n_opt_steps=(250 - 2 * DIM),
        # n_start_data=(1),
        # n_opt_steps=(10),
        n_candidates=50,
        # start_x=start_x,
        # start_y=start_y,
    )
    sbo.plot_history(bsave_fig=True)
    # sbo.plot_history_order(bsave_fig=False)
    # sbo.save_history()
    print("Best optimization result: ", result)

    best_weights = sbo.construct_weights(result["x_opt"], smc.detector)
    smc.calc_peak_positions(best_weights)
    smc.calc_loss()
    smc.plot_scanmap()


if __name__ == "__main__":
    main()

# sim loss det 0
# {'x_opt': array([0.971824, 0.947031, 0.919466, 0.912347, 1.013404, 1.037333, 1.1     , 1.017283]), 'y_opt': (4075.9394634595665, 4075.9394634595665)}
# {'x_opt': array([0.855084, 1.007138, 0.88301 , 1.009697, 1.104118, 1.024755, 1.042399, 0.992933]), 'y_opt': (5774.826759144951, 5774.826759144951)}
# {'x_opt': array([0.963974, 0.979819, 0.917613, 0.957483, 1.011557, 1.03814 , 1.066389, 1.005038]), 'y_opt': (3285.4448003377656, 3285.4448003377656)}
# {'x_opt': array([0.923633, 0.954014, 0.910152, 0.969097, 1.061233, 1.061396, 1.026949, 1.018377]), 'y_opt': (4481.049430870333, 4481.049430870333)}
# {'x_opt': array([0.997179, 1.011509, 0.914966, 0.954833, 1.001574, 1.014533, 1.030541, 1.026458]), 'y_opt': (3005.287809050258, 3005.287809050258)}
# {'x_opt': array([0.998522, 1.021792, 0.949858, 0.919018, 1.011224, 0.991306, 1.030706, 1.034205]), 'y_opt': (3268.150676664713, 3268.150676664713)}
# {'x_opt': array([0.974622, 0.977984, 0.904965, 0.95958 , 0.99712 , 1.044262, 1.058482, 1.019763]), 'y_opt': (3557.8216208759527, 3557.8216208759527)}
# {'x_opt': array([0.972942, 0.99892 , 0.920493, 0.944149, 1.015637, 1.027048, 1.051522, 1.034644]), 'y_opt': (4056.885121814777, 4056.885121814777)}
# {'x_opt': array([0.951984, 0.974236, 0.905264, 0.949583, 1.020036, 1.055654, 1.060328, 1.02599 ]), 'y_opt': (3675.9679754378158, 3675.9679754378158)}
# {'x_opt': array([0.963756, 0.989239, 0.915349, 0.948434, 1.015645, 1.036962, 1.056991, 1.026281]), 'y_opt': (3491.595511649137, 3491.595511649137)}
# {'x_opt': array([1.01852 , 0.992802, 0.952419, 0.950101, 0.997164, 0.995349, 1.042086, 1.004908]), 'y_opt': (2762.2964614052116, 2762.2964614052116)}
# {'x_opt': array([1.061688, 1.00633 , 0.963394, 0.951178, 0.957908, 0.986341, 1.032956, 1.007515]), 'y_opt': (3063.4246399926224, 3063.4246399926224)}
# {'x_opt': array([1.013405, 1.003819, 0.933614, 0.938359, 0.997237, 1.00559 , 1.046283, 1.015609]), 'y_opt': (2824.0531987495237, 2824.0531987495237)}
# {'x_opt': array([1.053141, 1.000046, 0.959638, 0.936576, 0.967539, 0.993334, 1.046417, 1.003358]), 'y_opt': (3117.5341544698617, 3117.5341544698617)}
# {'x_opt': array([1.019171, 1.005466, 0.941418, 0.933785, 0.992763, 0.998959, 1.046592, 1.012433]), 'y_opt': (2760.169897484338, 2760.169897484338)}
# {'x_opt': array([0.986912, 1.020239, 0.946554, 0.955842, 0.994931, 1.004388, 1.038196, 1.000945]), 'y_opt': (2504.8213482557885, 2504.8213482557885)}
# {'x_opt': array([1.018768, 1.034834, 0.964188, 0.961707, 0.965186, 0.997062, 1.020644, 0.997114]), 'y_opt': (2526.788822861071, 2526.788822861071)}
# {'x_opt': array([1.002249, 1.016963, 0.938807, 0.945932, 0.997153, 1.003167, 1.039167, 1.007208]), 'y_opt': (2640.9642359081136, 2640.9642359081136)}
# {'x_opt': array([1.004579, 1.013662, 0.941514, 0.947196, 0.993279, 1.003073, 1.038321, 1.006907]), 'y_opt': (2602.9634294334714, 2602.9634294334714)}