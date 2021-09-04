""""""

import numpy as np
import optuna
import time
from panter.map.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117


def main(n_trials: int = 100):
    pos, evs = scan_200117()
    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        detector=0,
    )

    def objective(trial):
        """"""

        f1 = trial.suggest_float("f1", 0.8, 1.2)
        f2 = trial.suggest_float("f2", 0.8, 1.2)
        f3 = trial.suggest_float("f3", 0.8, 1.2)
        f4 = trial.suggest_float("f4", 0.8, 1.2)

        w_list = [1.0] * 16
        w_list[0] = f1
        w_list[1] = f1
        w_list[2] = f2
        w_list[3] = f2
        w_list[4] = f3
        w_list[5] = f3
        w_list[6] = f4
        w_list[7] = f4

        print(w_list)
        smc.calc_peak_positions(weights=np.array(w_list))
        losses = smc.calc_loss()

        if losses[1] is not None:
            return losses[1]
        else:
            return None

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(study.best_trial)
    # smc.plot_scanmap()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Time delta: ", end_time - start_time)

# out of 300 (29100 s)
# params={'f1': 0.9495966004622409, 'f2': 0.9195980955915593, 'f3': 1.0341408752398, 'f4': 1.033122930306086} 8680.060214588333
# params={'f1': 0.9910702369553084, 'f2': 0.9435642229527875, 'f3': 1.005110837338069, 'f4': 1.0140092409774735} 8024.3849319901765
