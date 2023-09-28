from typing import Callable, Dict, Optional, Sequence, Tuple

import pandas as pd


def df2table_meanstd(
    df: pd.DataFrame,
    rows: Sequence[str],
    metrics: Sequence[str],
    label_mapping: Optional[Dict[str, str]] = None,
    caption: Optional[str] = None,
    precision: int = 2,
    postprocess_pivot: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Convert a dataframe to a latex table with mean and std for each metric.

    The expected format of the dataframe df is the following:
        - One row per experiment
        - One column per metric, any name is valid. Without the std, it will be computed internally.
        - Grouping by rows should yield an unique aggregated metric (be careful not to aggregate for missing variation factors)

    Args:
        df: the dataframe to convert.
        rows: the rows of the pivot table.
        metrics: the metrics to display.
        label_mapping: a mapping from old labels to new labels.
        caption: the caption of the latex table.
        precision: the precision of the numbers.
        postprocess_pivot: a function to postprocess the pivot table.


    Returns:
        resutling dataframe and the nicely formatted latex table.
    """
    # Create the pivot table
    result = df.pivot_table(
        index=rows,
        values=metrics,
        sort=False,
        aggfunc=["mean", "std"],
    )
    if postprocess_pivot is not None:
        result = postprocess_pivot(result)

    # Retrieve the list of metrics
    metrics_name = sorted(set(result.columns.levels[1]))

    # Create the meanstd first level column
    for metric_name in metrics_name:
        meanstd_col = result.apply(lambda row: [row[("mean", metric_name)], row[("std", metric_name)]], axis=1)
        result[("meanstd", metric_name)] = meanstd_col

    # Create the list of columns to display
    display_metrics = [("meanstd", x) for x in metrics_name]

    def _formatter(x):
        mean, std = x
        return rf"${round(mean, precision):.{precision}f} \pm {round(std, precision):.{precision}f}$"

    if label_mapping is not None:
        result = result.rename(label_mapping)

    # Convert to latex table handling the mean and std formatting
    latextable = result[display_metrics].to_latex(
        escape=False,
        caption=caption,
        label=caption,
        multirow=True,
        sparsify=True,
        multicolumn_format="c",
        float_format="%.{precision}f",
        formatters={("meanstd", cellvalue): _formatter for cellvalue in metrics_name},
    )
    return result, latextable
