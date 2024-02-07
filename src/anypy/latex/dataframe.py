from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import pandas as pd


def default_meanstd_formatter(meanstd: Tuple[float, float], precision: int) -> str:
    mean, std = meanstd
    return rf"${round(mean, precision):.{precision}f} \pm {round(std, precision):.{precision}f}$"


def df2table_meanstd(
    df: pd.DataFrame,
    rows: Sequence[str],
    columns: Sequence[str],
    metrics: Sequence[str],
    label_mapping: Optional[Dict[str, str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    meanstd_formatter: Optional[Callable[[Tuple[float, float], int], str]] = None,
    postprocess_pivot: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    columns_hardsorted: Optional[Sequence[str]] = None,
    rows_hardsorted: Optional[Sequence[str]] = None,
    str_replace: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Convert a dataframe to a latex table with mean and std for each metric.

    The expected format of the dataframe df is the following:
        - One row per experiment
        - One column per metric, any name is valid. Without the std, it will be computed internally.
        - Grouping by rows should yield an unique aggregated metric (be careful not to aggregate for missing variation factors)

    Args:
        df: the dataframe to convert.
        rows: the rows of the pivot table.
        columns: the columns of the pivot table.
        metrics: the metrics to display.
        label_mapping: a mapping from old labels to new labels.
        caption: the caption of the latex table.
        label: the label of the latex table.
        meanstd_formatter: a function to format the mean and std.
        postprocess_pivot: a function to postprocess the pivot table.
        columns_hardsorted: the columns to hard sort with a reindex.
        rows_hardsorted: the rows to hard sort with a reindex.
        str_replace: a dictionary to replace strings in the latex table.

    Returns:
        resutling dataframe and the nicely formatted latex table.
    """
    # Create the pivot table
    result = df.pivot_table(
        index=rows,
        columns=columns,
        values=metrics,
        sort=False,
        aggfunc=["mean", "std"],
    )
    if postprocess_pivot is not None:
        result = postprocess_pivot(result)

    # Retrieve the list of metrics
    metrics_name = list(set([x[1:] for x in result.columns]))

    # Create the meanstd first level column
    for metric_name in metrics_name:
        meanstd_col = result.apply(lambda row: [row[("mean", *metric_name)], row[("std", *metric_name)]], axis=1)
        result[("meanstd", *metric_name)] = meanstd_col

    # Create the list of columns to display
    display_metrics = [("meanstd", *x) for x in metrics_name]

    if label_mapping is not None:
        result = result.rename(label_mapping)

    # Consider only the meanstd column
    result = result[display_metrics]

    if columns_hardsorted is not None:
        result = result[columns_hardsorted]

    if rows_hardsorted is not None:
        result = result.reindex(index=rows_hardsorted)

    if meanstd_formatter is None:
        meanstd_formatter = partial(default_meanstd_formatter, precision=2)

    # Convert to latex table handling the mean and std formatting
    latextable = result.to_latex(
        escape=False,
        caption=caption,
        label=label,
        multirow=True,
        sparsify=True,
        multicolumn_format="c",
        float_format="%.{precision}f",
        formatters={("meanstd", *cellvalue): meanstd_formatter for cellvalue in metrics_name},
    )
    if str_replace is not None:
        for old, new in str_replace.items():
            latextable = latextable.replace(old, new)

    return result, latextable
