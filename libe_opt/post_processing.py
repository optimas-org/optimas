"""
This file contains a class that helps post-process libE optimization
"""
import numpy as np
import pandas as pd
import os

class PostProcOptimization(object):

    def __init__(self, path):
        """
        Initialize a postprocessing object

        Parameter:
        ----------
        path: string
            Path to the folder that contains the libE optimization
        """
        # Find the `npy` file that contains the results
        output_files = [ filename for filename in os.listdir(path) \
                    if filename.startswith('libE_history_for_run_')
                   and filename.endswith('.npy')]
        assert len(output_files) == 1

        # Load the file as a pandas DataFrame
        x  = np.load( os.path.join(path, output_files[0]) )
        d = { label: x[label].flatten() for label in x.dtype.names \
                if label not in ['x', 'x_on_cube'] }
        self.df = pd.DataFrame(d)

        # Only keep the simulations that finished properly
        self.df = self.df[self.df.returned]

    def get_df(self):
        return self.df
