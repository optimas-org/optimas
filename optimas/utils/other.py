"""Definition of other utilities used internally by optimas."""

from typing import Union, List, Dict

import numpy as np
import pandas as pd


def convert_to_dataframe(
    data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
) -> pd.DataFrame:
    """Convert input data to a pandas DataFrame.

    Parameters
    ----------
    data : dict of lists, list of dicts, ndarray or DataFrame
        The input data, which can be a dictionary of lists, a list of
        dictionaries, a numpy structured array or a pandas dataframe.

    Returns
    -------
    pd.DataFrame
        The converted input data.

    Raises
    ------
    ValueError
        If the type of the input data is not supported.
    """
    # Get fields in given data.
    if isinstance(data, np.ndarray):
        # Labels with multidimensional arrays are converted to a list of lists.
        d = {
            label: data[label].tolist() if data[label].ndim > 1 else data[label]
            for label in data.dtype.names
        }
        return pd.DataFrame(d)
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        # Check whether the elements in the dictionary are arrays or not.
        # If they are not, covert to 1-element arrays for DataFrame initialization.
        element = data[list(data.keys())[0]]
        if not hasattr(element, "__len__"):
            for key, value in data.items():
                data[key] = np.ones(1, dtype=type(value)) * value
        return pd.DataFrame(data)
    elif isinstance(data, list):
        fields = list(data[0].keys())
        fields.sort()
        for row in data:
            row_fields = list(row.keys())
            row_fields.sort()
            if row_fields != fields:
                raise ValueError("Not all dictionaries contain the same keys.")
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Cannot convert {type(data)} to a pandas dataframe.")


def get_df_with_selection(df: pd.DataFrame, select: Dict) -> pd.DataFrame:
    """Return the DataFrame after applying selection criterium.

    Parameters
    ----------
    df : DataFrame
        The DataFrame object
    select: dict
        A dictionary containing the selection criteria to apply.
        e.g. {'f' : [None, -10.]} (get data with f < -10)
    """
    condition = ""
    for key in select:
        if select[key][0] is not None:
            if condition != "":
                condition += " and "
            condition += "%s > %f" % (key, select[key][0])
        if select[key][1] is not None:
            if condition != "":
                condition += " and "
            condition += "%s < %f" % (key, select[key][1])

    return df.query(condition)
