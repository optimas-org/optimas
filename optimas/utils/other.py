"""Definition of other utilities used internally by optimas."""

from typing import Any, Union, List, Dict

import numpy as np
import pandas as pd


def update_object(object_old: Any, object_new: Any) -> None:
    """Update the attributes of an object with those from a newer one.

    This method is intended to be used with objects of the same type and that
    have a `__dict__` attribute.

    Parameters
    ----------
    object_old : Any
        The object to be updated.
    object_new : Any
        The object from which to get the updated attributes.

    """
    assert isinstance(object_new, type(object_old)), "Object types don't match"
    for key, value in vars(object_new).items():
        setattr(object_old, key, value)


def convert_to_dataframe(
    data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame]
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
