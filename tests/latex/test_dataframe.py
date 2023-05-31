import numpy as np
import pandas as pd

from anypy.latex import df2table_meanstd


def test_df2table_meanstd():
    np.random.seed(42)
    df = pd.DataFrame({"model": ["a"] * 100, "score": np.random.randn(100)})
    _, latex_table = df2table_meanstd(
        df, rows=["model"], metrics=["score"], label_mapping={"a": "My Cool Model"}, caption="Test", precision=3
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
