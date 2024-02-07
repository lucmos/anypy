from functools import partial

import numpy as np
import pandas as pd

from anypy.latex import default_meanstd_formatter, df2table_meanstd


def test_df2table_meanstd():
    np.random.seed(42)
    df = pd.DataFrame({"model": ["a"] * 100, "score": np.random.randn(100)})
    _, latex_table = df2table_meanstd(
        df,
        rows=["model"],
        columns=None,
        metrics=["score"],
        meanstd_formatter=partial(default_meanstd_formatter, precision=3),
        label_mapping={"a": "My Cool Model"},
        caption="Test",
        label="Test",
    )
    gt_table = r"""\begin{table}
\caption{Test}
\label{Test}
\begin{tabular}{ll}
\toprule
 & meanstd \\
 & score \\
model &  \\
\midrule
My Cool Model & $-0.104 \pm 0.908$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    assert gt_table == latex_table
