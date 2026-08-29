import pandas as pd

df = pd.read_parquet(
    "Data_processed/driver_analysis.parquet"
)

event = df[
    (df["date"] == "2026-08-31") &
    (df["region"] == "South")
].sort_values("driver_rank")

print(
    event[
        [
            "driver",
            "driver_rank",
            "driver_change",
            "correlation_same_day",
            "correlation_lag_1d",
            "correlation_lag_3d",
            "correlation_lag_7d",
            "best_lag_days",
            "best_lag_correlation",
            "segment_consistency",
            "direction_alignment",
            "evidence_score",
            "evidence_strength",
        ]
    ].to_string(index=False)
)