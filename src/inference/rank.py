from __future__ import annotations

import pandas as pd


def rank_predictions(pred_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    out = pred_df.copy()
    out = out.sort_values("confidence", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out.head(top_n)
