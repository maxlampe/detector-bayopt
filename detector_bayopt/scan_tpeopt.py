""""""

import warnings
import numpy as np
import optuna
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

from base_scanopt import BaseScanOpt

warnings.filterwarnings("ignore")


class ScanTPEOpt(BaseScanOpt):
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
            opt_label="TPEOpt",
            bdummy_values=bdummy_values,
        )

        self._optuna_study = None

    def optimize(self, n_opt_steps: int = 80):
        """"""

        self._optuna_study = optuna.create_study(
            direction="minimize",
            study_name=f"ScanMapTPE_dim{self._w_dim}_trails{n_opt_steps}",
        )
        self._optuna_study.optimize(self._objective_func, n_trials=n_opt_steps)

        return self.optimum

    def _objective_func(self, trial):
        """"""

        f_params = []
        for i in range(self._w_dim):
            w = trial.suggest_float(f"w{i}", self._w_range[0], self._w_range[1])
            f_params.append(w)
        f_params = np.array(f_params)
        losses = self.calc_losses(f_params)

        if losses[1] is not None:
            return losses[1]
        else:
            return None


def main():
    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        label=scan_200117.label,
        detector=0,
    )

    stpe = ScanTPEOpt(scan_map_class=smc, weight_dim=4, detector=smc.detector)
    result = stpe.optimize(n_opt_steps=75)
    stpe.plot_history(bsave_fig=True)
    # stpe.plot_history_order(bsave_fig=True)
    # stpe.save_history()

    print("Best optimization result: ", result)
    # best_weights = stpe.construct_weights(result["x_opt"], smc.detector)
    # smc.calc_peak_positions(best_weights)
    # smc.calc_loss()
    # smc.plot_scanmap()


if __name__ == "__main__":
    main()
