""""""

import warnings
import numpy as np
from scipy.optimize import dual_annealing
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

from base_scanopt import BaseScanOpt

warnings.filterwarnings("ignore")


class ScanMCMCOpt(BaseScanOpt):
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
            opt_label="MCMCOpt",
        )

    def optimize(self, n_opt_steps: int = 80):
        """"""

        w_rng = [list(self._w_range)] * self._w_dim
        res = dual_annealing(
            self._objective_function,
            bounds=w_rng,
            maxiter=n_opt_steps,
            maxfun=n_opt_steps,
        )
        print(res)

        return self.optimum

    def _objective_function(self, x):
        """"""
        return self.calc_losses(in_weights=x)[1]


def main():
    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=0,
    )

    smcmc = ScanMCMCOpt(scan_map_class=smc, weight_dim=8, detector=smc.detector)
    result = smcmc.optimize(n_opt_steps=500)
    smcmc.plot_history()
    smcmc.save_history()

    print("Best optimization result: ", result)
    best_weights = smcmc.construct_weights(result["x_opt"], smc.detector)
    smc.calc_peak_positions(best_weights)
    smc.calc_loss()
    smc.plot_scanmap()


if __name__ == "__main__":
    main()
