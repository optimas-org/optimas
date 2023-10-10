"""Contains the definition of the Ax metric used for multitask optimization."""

import pandas as pd
from ax import Metric
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.utils.common.result import Ok


class AxMetric(Metric):
    """Custom metric to be optimized during the experiment."""

    def fetch_trial_data(self, trial: BatchTrial):
        """Fetch data for one trial."""
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            mean, sem = trial.run_metadata[arm_name]["f"]
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": mean,
                    "sem": sem,
                }
            )
        return Ok(value=Data(df=pd.DataFrame.from_records(records)))
