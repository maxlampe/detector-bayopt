""""""

import warnings
import numpy as np
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

from base_scanopt import BaseScanOpt

warnings.filterwarnings("ignore")


class ScanRandomOpt(BaseScanOpt):
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
            opt_label="RandomOpt",
            bdummy_values=bdummy_values,
        )

    def optimize(self, n_opt_steps: int = 80):
        """"""

        for i in range(n_opt_steps):
            print(f"Current optimizer step:\t{i + 1}/{n_opt_steps}")
            rng_weights = self._get_random_weights()
            losses = self.calc_losses(rng_weights)

        return self.optimum


def main():
    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=0,
    )

    srndm = ScanRandomOpt(
        scan_map_class=smc, weight_dim=4, detector=smc.detector, bdummy_values=False
    )
    result = srndm.optimize(n_opt_steps=75)
    # srndm.plot_history(bsave_fig=False)
    # srndm.plot_history_order(bsave_fig=True)
    # srndm.save_history()

    print("Best optimization result: ", result)
    best_weights = srndm.construct_weights(result["x_opt"], smc.detector)
    # smc.calc_peak_positions(best_weights)
    # smc.calc_loss()
    # smc.plot_scanmap()


if __name__ == "__main__":
    main()
