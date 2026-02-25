# import numpy as np
# import config

# def calculate_profit(y_true, probs):
#     results = []

#     thresholds = np.arange(
#         config.THRESHOLD_START,
#         config.THRESHOLD_END,
#         config.THRESHOLD_STEP
#     )

#     for t in thresholds:
#         preds = (probs >= t).astype(int)

#         tp = ((preds == 1) & (y_true == 1)).sum()
#         fp = ((preds == 1) & (y_true == 0)).sum()
#         fn = ((preds == 0) & (y_true == 1)).sum()
#         tn = ((preds == 0) & (y_true == 0)).sum()

#         profit = (
#             tp * config.CHURN_REVENUE
#             - (tp + fp) * config.RETENTION_COST
#         )

#         results.append((t, profit, tp, fp, fn, tn))

#     return results

import numpy as np
import pandas as pd
import config

def calculate_profit(y_true, probs):
    results = []

    thresholds = np.arange(
        config.THRESHOLD_START,
        config.THRESHOLD_END,
        config.THRESHOLD_STEP
    )

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        profit = (
            tp * config.CHURN_REVENUE
            - (tp + fp) * config.RETENTION_COST
        )

        results.append({
            "threshold": t,
            "profit": profit,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="profit",
        ascending=False
    ).reset_index(drop=True)

    return results_df