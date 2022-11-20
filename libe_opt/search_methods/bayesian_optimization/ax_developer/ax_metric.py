import numpy as np
import pandas as pd
from ax import Metric
from ax.core.data import Data


class AxMetric(Metric):
    """ Custom metric to be optimized during the experiment. """

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": trial.run_metadata[arm_name]['f'],
                "sem": np.nan,
            })
        return Data(df=pd.DataFrame.from_records(records))
