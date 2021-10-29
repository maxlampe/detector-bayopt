""""""
# TODO: Dummy loss vals
# TODO: do time.time() eval

import warnings
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from panter.map.scan2DMap import ScanMapClass

plt.rcParams.update({"font.size": 12})
warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, linewidth=120)

CLRS = {
    "blue": "#0c0887",
    "purple-blue": "#4b03a1",
    "dark-purple": "#7d03a8",
    "purple": "#a82296",
    "magenta": "#cb4679",
    "red": "#e56b5d",
    "dark-orange": "#f89441",
    "orange": "#fdc328",
    "yellow": "#f0f921",
}


class BaseScanOpt:
    """"""

    def __init__(
        self,
        scan_map_class: ScanMapClass,
        weight_dim: int = 4,
        weight_range: np.array = np.array([0.9, 1.1]),
        detector: int = 0,
        opt_label: str = "Base",
        bdummy_values: bool = False,
    ):
        self._smc = scan_map_class
        self._w_dim = weight_dim
        self._w_range = weight_range
        self._det = detector
        self._opt_label = opt_label
        self._use_dummys = bdummy_values

        self.optimum = None
        self.weight_hist = []
        self.loss_hist = []

    def __repr__(self):
        return f"{self._opt_label}_det{self._det}_dim{self._w_dim}_{self._smc.label}"

    def __str__(self):
        return f"{self._opt_label}_det{self._det}_dim{self._w_dim}_{self._smc.label}"

    def optimize(self, n_opt_steps: int = 10):
        """Optimize the panter ScanMap for a given number of optimization steps."""
        pass

    def calc_losses(self, in_weights: np.array, badd_to_history: bool = True):
        """Calcute peak positions and losses for a given set of weights."""

        if not self._use_dummys:
            new_weights = self.construct_weights(in_weights, self._det)
            self._smc.calc_peak_positions(weights=np.array(new_weights))
            losses = self._smc.calc_loss()
        else:
            losses = self._calc_dummy_losses()
        print(f"\tLatest weights: {in_weights} \t losses: {losses}")

        if badd_to_history:
            if torch.is_tensor(in_weights):
                in_weights = in_weights.cpu().detach().numpy()
            self.weight_hist.append(in_weights)
            self.loss_hist.append(losses)

            loss_hist_array = np.array(self.loss_hist)
            argmin = np.argmin(loss_hist_array.T[1])
            self.optimum = {
                "x_opt": self.weight_hist[argmin],
                "y_opt": self.loss_hist[argmin],
            }
            print(f"\tCurr optimum @step{argmin + 1}: {self.optimum}")

        return losses

    def _get_random_weights(self):
        """"""

        # rng_weights = np.random.rand(4) * (w_range[1] - w_range[0]) + w_range[0]
        a = (self._w_range[1] + self._w_range[0]) * 0.5
        b = (self._w_range[1] - self._w_range[0]) * 0.25
        assert b > 0, f"Lower range bound smaller than upper bound: {self._w_range}"
        rng_weights = b * np.random.randn(self._w_dim) + a

        return rng_weights

    def plot_history(self, y=None, bsave_fig: bool = False):
        """"""

        if y is not None:
            loss_plot = np.array(y)
        else:
            loss_plot = np.array(self.loss_hist)

        opt_curve = [loss_plot.T[1][0]]
        for i in range(1, loss_plot.T[1].shape[0]):
            opt_curve.append(loss_plot.T[1][: (i + 1)].min())
        opt_curve = np.array(opt_curve)

        plt.figure(figsize=(8, 8))
        plt.plot(opt_curve, label="Best value", c=CLRS["blue"], linewidth=2.0)
        plt.plot(loss_plot.T[0], "-x", label="Uniformity", c=CLRS["orange"], alpha=0.2)
        plt.plot(
            loss_plot.T[1],
            "-o",
            label="Uniformity + Symm.-Loss",
            c=CLRS["red"],
            alpha=0.2,
        )
        plt.plot(
            [24e3] * loss_plot.T[0].shape[0],
            # "-",
            label="Unoptimized",
            c=CLRS["magenta"],
            linewidth=2.0,
        )
        plt.title(f"{self._opt_label} - Loss curve", fontsize=22)
        plt.xlabel("Function calls [ ]")
        plt.ylabel("Loss [ ]")
        plt.ylim([5000.0, 50e3])
        plt.legend()

        if bsave_fig:
            plt.savefig(f"{self._opt_label}_losscurve.png", dpi=300)
        plt.show()

    def plot_history_order(self, x=None, y=None, bsave_fig: bool = False):
        """Plot the optimization history a scatter plot - works only for dim=2"""

        assert (
            self._w_dim == 2
        ), f"ERROR: Plotting history order only works for dim=2.(dim={self._w_dim})"

        if x is not None and y is not None:
            data = np.asarray(x)
            colour = np.log(np.asarray(y))
        else:
            data = np.asarray(self.weight_hist)
            colour = np.log(np.asarray(self.loss_hist))

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle(f"{self._opt_label} - Optimization History", fontsize=22)

        plt.plot(data.T[0], data.T[1], zorder=2, c=CLRS["magenta"], alpha=0.3)
        plt.scatter(
            data.T[0],
            data.T[1],
            c=colour.T[1],
            cmap="plasma",
            marker="o",
            s=200.0,
            zorder=1,
        )
        for i, txt in enumerate(data):
            if i < 9:
                txt = f"0{i + 1}"
            else:
                txt = f"{i + 1}"
            axs.annotate(
                txt,
                (data.T[0, i], data.T[1, i]),
                c="white",
                bbox={"facecolor": "black", "alpha": 0.7, "pad": 1.5},
                fontsize=8,
            )
        axs.set_xlim([0.9, 1.1])
        axs.set_ylim([0.9, 1.1])
        axs.set_xlabel("w1 [ ]")
        axs.set_ylabel("w2 [ ]")
        cbar = plt.colorbar()
        cbar.set_label("Log Symmetry Loss")

        if bsave_fig:
            plt.savefig(f"{self._opt_label}_lossorder.png", dpi=300)
        plt.show()

    def save_history(self, file_name: str = None):
        """"""

        if file_name is None:
            file_name = f"{self._opt_label}_history.plk"
        with open(file_name, "wb") as outp:
            pickle.dump(
                [self.weight_hist, self.loss_hist], outp, pickle.HIGHEST_PROTOCOL
            )

    @staticmethod
    def construct_weights(weights: np.array, detector: int = 0):
        """For a given weight array, construct a passable array for panter ScanMap.

        Takes arrays with length 2, 4 and 8 due to the symmetry of the problem."""

        w_list = [1.0] * 16

        if weights.shape[0] == 2:
            for i in range(2):
                for j in range(4):
                    w_list[(8 * detector + j + i * 4)] = weights[i]
        elif weights.shape[0] == 4:
            for i in range(4):
                w_list[(8 * detector + i * 2)] = weights[i]
                w_list[(8 * detector + i * 2 + 1)] = weights[i]
        elif weights.shape[0] == 8:
            for i in range(8):
                w_list[(8 * detector + i)] = weights[i]
        else:
            assert False, "ERROR: Invalid weight array length. Needs to be 2, 4 or 8."

        return np.array(w_list)

    @staticmethod
    def _calc_dummy_losses():
        """Calculate dummy loss values to enable faster debugging."""

        symm = float(np.random.randn(1) * 8000.0 + 25e3)
        unif = float(symm - (np.random.randn(1) * 100.0 + 500.0))

        return [unif, symm]
