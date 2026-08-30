from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "Data_processed"


# ============================================================
# LOAD DATA
# ============================================================

def load_kpis():

    daily = pd.read_parquet(
        PROCESSED_DIR / "daily_kpis.parquet"
    )

    regional = pd.read_parquet(
        PROCESSED_DIR / "regional_kpis.parquet"
    )

    daily["date"] = pd.to_datetime(daily["date"])
    regional["date"] = pd.to_datetime(regional["date"])

    return daily, regional


# ============================================================
# DATASET OVERVIEW
# ============================================================

def inspect_overview(daily, regional):

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(
        f"\nDaily KPI rows     : {len(daily):,}"
    )

    print(
        f"Regional KPI rows  : {len(regional):,}"
    )

    print(
        f"Number of regions  : "
        f"{regional['region'].nunique()}"
    )

    print(
        f"Date range         : "
        f"{daily['date'].min().date()} → "
        f"{daily['date'].max().date()}"
    )

    print(
        f"Number of days     : "
        f"{daily['date'].nunique()}"
    )


# ============================================================
# KPI SUMMARY
# ============================================================

def inspect_kpi_distributions(daily):

    print("\n" + "=" * 70)
    print("KPI DISTRIBUTIONS")
    print("=" * 70)

    kpis = [
        "revenue",
        "orders",
        "aov",
        "conversion_rate",
        "marketing_spend",
        "inventory_availability",
        "delivery_delay_rate",
        "return_rate",
        "discount_rate",
    ]

    rows = []

    for kpi in kpis:

        if kpi not in daily.columns:
            continue

        series = daily[kpi].dropna()

        rows.append({
            "KPI": kpi,
            "Mean": series.mean(),
            "Std": series.std(),
            "Min": series.min(),
            "Max": series.max(),
        })

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# TOTAL BUSINESS METRICS
# ============================================================

def inspect_business_metrics(daily):

    print("\n" + "=" * 70)
    print("BUSINESS SUMMARY")
    print("=" * 70)

    total_revenue = daily["revenue"].sum()
    total_orders = daily["orders"].sum()

    overall_aov = (
        total_revenue / total_orders
        if total_orders > 0
        else 0
    )

    weighted_conversion = (
        daily["conversion_rate"]
        * daily["sessions"]
    ).sum() / daily["sessions"].sum()

    print(
        f"\nTotal revenue       : ₹{total_revenue:,.2f}"
    )

    print(
        f"Total orders        : {total_orders:,.0f}"
    )

    print(
        f"Overall AOV         : ₹{overall_aov:,.2f}"
    )

    print(
        f"Overall conversion  : "
        f"{weighted_conversion:.2%}"
    )

    print(
        f"Avg inventory avail : "
        f"{daily['inventory_availability'].mean():.2%}"
    )

    print(
        f"Avg delivery delay  : "
        f"{daily['delivery_delay_rate'].mean():.2%}"
    )

    print(
        f"Avg return rate     : "
        f"{daily['return_rate'].mean():.2%}"
    )

    print(
        f"Avg discount rate   : "
        f"{daily['discount_rate'].mean():.2%}"
    )


# ============================================================
# REGIONAL SUMMARY
# ============================================================

def inspect_regions(regional):

    print("\n" + "=" * 70)
    print("REGIONAL SUMMARY")
    print("=" * 70)

    summary = (
        regional
        .groupby("region")
        .agg(
            revenue=("revenue", "sum"),
            orders=("orders", "sum"),
            avg_aov=("aov", "mean"),
            avg_conversion=(
                "conversion_rate",
                "mean"
            ),
            avg_inventory=(
                "inventory_availability",
                "mean"
            ),
            avg_delivery_delay=(
                "delivery_delay_rate",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# TOP / BOTTOM DAYS
# ============================================================

def inspect_extreme_days(daily):

    print("\n" + "=" * 70)
    print("EXTREME REVENUE DAYS")
    print("=" * 70)

    columns = [
        "date",
        "revenue",
        "orders",
        "aov",
        "conversion_rate",
        "marketing_spend",
        "inventory_availability",
        "delivery_delay_rate",
    ]

    print("\nLowest revenue days:")

    print(
        daily
        .nsmallest(10, "revenue")[columns]
        .to_string(index=False)
    )

    print("\nHighest revenue days:")

    print(
        daily
        .nlargest(10, "revenue")[columns]
        .to_string(index=False)
    )


# ============================================================
# DAY-OVER-DAY MOVEMENTS
# ============================================================

def inspect_movements(daily):

    print("\n" + "=" * 70)
    print("LARGEST DAY-OVER-DAY KPI MOVEMENTS")
    print("=" * 70)

    df = daily.copy()

    kpis = [
        "revenue",
        "orders",
        "aov",
        "conversion_rate",
        "marketing_spend",
        "inventory_availability",
        "delivery_delay_rate",
        "return_rate",
        "discount_rate",
    ]

    for kpi in kpis:

        if kpi not in df.columns:
            continue

        pct_change = (
            df[kpi]
            .pct_change()
            .replace(
                [float("inf"), -float("inf")],
                pd.NA
            )
        )

        idx = pct_change.abs().nlargest(3).index

        print(f"\n{kpi}:")

        result = df.loc[
            idx,
            ["date", kpi]
        ].copy()

        result["change"] = (
            pct_change.loc[idx].values
        )

        print(
            result.to_string(
                index=False
            )
        )


# ============================================================
# REGIONAL REVENUE CONTRIBUTION
# ============================================================

def inspect_regional_revenue(regional):

    print("\n" + "=" * 70)
    print("REGIONAL REVENUE CONTRIBUTION")
    print("=" * 70)

    summary = (
        regional
        .groupby("region")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total = summary.sum()

    result = pd.DataFrame({
        "revenue": summary,
        "share": summary / total,
    })

    print(
        result.to_string()
    )


# ============================================================
# NORTH REGION DIAGNOSTIC
# ============================================================

def inspect_north_region(regional):

    if "North" not in regional["region"].unique():

        return

    print("\n" + "=" * 70)
    print("NORTH REGION DIAGNOSTIC")
    print("=" * 70)

    north = (
        regional[
            regional["region"] == "North"
        ]
        .sort_values("date")
    )

    columns = [
        "date",
        "revenue",
        "orders",
        "conversion_rate",
        "inventory_availability",
        "delivery_delay_rate",
    ]

    print(
        north[columns]
        .tail(20)
        .to_string(index=False)
    )


# ============================================================
# CORRELATION — DIAGNOSTIC ONLY
# ============================================================

def inspect_basic_correlations(daily):

    print("\n" + "=" * 70)
    print("BASIC CORRELATIONS — DIAGNOSTIC ONLY")
    print("=" * 70)

    columns = [
        "revenue",
        "orders",
        "aov",
        "conversion_rate",
        "marketing_spend",
        "inventory_availability",
        "delivery_delay_rate",
        "return_rate",
        "discount_rate",
    ]

    available = [
        column
        for column in columns
        if column in daily.columns
    ]

    correlation = (
        daily[available]
        .corr()["revenue"]
        .sort_values(
            ascending=False
        )
    )

    print(
        correlation.to_string()
    )

    print(
        "\nNOTE:"
    )

    print(
        "These correlations are descriptive only."
    )

    print(
        "They are NOT used as causal evidence."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("KPI DATA INSPECTION")
    print("=" * 70)

    daily, regional = load_kpis()

    inspect_overview(
        daily,
        regional
    )

    inspect_kpi_distributions(
        daily
    )

    inspect_business_metrics(
        daily
    )

    inspect_regions(
        regional
    )

    inspect_extreme_days(
        daily
    )

    inspect_movements(
        daily
    )

    inspect_regional_revenue(
        regional
    )

    inspect_north_region(
        regional
    )

    inspect_basic_correlations(
        daily
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()