""""""
# TODO: Dummy loss vals
# TODO: do time.time() eval

import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from panter.map.scan2DMap import ScanMapClass

plt.rcParams.update({'font.size': 12})
warnings.filterwarnings("ignore")


class BaseScanOpt:
    """"""

    def __init__(
        self,
        scan_map_class: ScanMapClass,
        weight_dim: int = 4,
        weight_range: np.array = np.array([0.9, 1.1]),
        detector: int = 0,
        opt_label: str = "Base",
    ):
        self._smc = scan_map_class
        self._w_dim = weight_dim
        self._w_range = weight_range
        self._det = detector
        self._opt_label = opt_label

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

        new_weights = self.construct_weights(in_weights, self._det)
        self._smc.calc_peak_positions(weights=np.array(new_weights))
        losses = self._smc.calc_loss()
        print(f"Latest weights: {in_weights} \t losses: {losses}")

        if badd_to_history:
            self.weight_hist.append(in_weights)
            self.loss_hist.append(losses)

            loss_hist_array = np.array(self.loss_hist)
            argmin = np.argmin(loss_hist_array.T[1])
            self.optimum = {
                "x_opt": self.weight_hist[argmin],
                "y_opt": self.loss_hist[argmin],
            }
            print(f"Curr optimum @step{argmin + 1}: {self.optimum}")

        return losses

    def _get_random_weights(self):
        """"""

        # rng_weights = np.random.rand(4) * (w_range[1] - w_range[0]) + w_range[0]
        a = (self._w_range[1] + self._w_range[0]) * 0.5
        b = (self._w_range[1] - self._w_range[0]) * 0.25
        assert b > 0, f"Lower range bound smaller than upper bound: {self._w_range}"
        rng_weights = b * np.random.randn(self._w_dim) + a

        return rng_weights

    def plot_history(self):
        """"""

        loss_plot = np.array(self.loss_hist)

        plt.figure(figsize=(8, 8))
        plt.plot(loss_plot.T[0], "-x", label="Uniformity")
        plt.plot(loss_plot.T[1], "-o", label="Symmetry Loss")
        plt.title(f"{self._opt_label} - Loss curve")
        plt.xlabel("Optimization steps [ ]")
        plt.ylabel("Loss [ ]")
        plt.ylim([5000.0, 50e3])
        plt.legend()
        plt.show()

        plt.savefig(f"{self._opt_label}_losscurve", dpi=300)

    def plot_history_order(self):
        """WIP"""

        data = self.weight_hist
        colour = self.loss_hist

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle("Optimization History", fontsize=22)

        plt.plot(data.T[0], data.T[1], zorder=2, c="#cb4679", alpha=0.4)
        plt.scatter(data.T[0], data.T[1], c=colour, cmap="plasma", marker='o', s=200., zorder=1)
        for i, txt in enumerate(data):
            if i < 9:
                txt = f"0{i + 1}"
            else:
                txt = f"{i + 1}"
            axs.annotate(
                txt,
                (data.T[0, i], data.T[1, i]),
                c="white",
                bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 1.5},
            )
        axs.set_xlim([0.9, 1.1])
        axs.set_ylim([0.9, 1.1])
        axs.set_xlabel("w1 [ ]")
        axs.set_ylabel("w2 [ ]")
        # axs.set_facecolor('#0c0887')
        cbar = plt.colorbar()
        cbar.set_label("Symmetry Loss")
        # plt.savefig("opt_step_hist.png", dpi=300)
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
