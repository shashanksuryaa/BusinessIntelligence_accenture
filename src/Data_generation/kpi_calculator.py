from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "Data_raw"
PROCESSED_DIR = ROOT / "Data_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():

    print("\n" + "=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    orders = pd.read_parquet(
        RAW_DIR / "sales_orders.parquet"
    )

    sessions = pd.read_parquet(
        RAW_DIR / "web_sessions.parquet"
    )

    marketing = pd.read_parquet(
        RAW_DIR / "marketing.parquet"
    )

    inventory = pd.read_parquet(
        RAW_DIR / "inventory.parquet"
    )

    deliveries = pd.read_parquet(
        RAW_DIR / "deliveries.parquet"
    )

    return (
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    orders,
    sessions,
    marketing,
    inventory,
    deliveries,
):

    orders["date"] = pd.to_datetime(
        orders["date"]
    ).dt.normalize()

    sessions["date"] = pd.to_datetime(
        sessions["date"]
    ).dt.normalize()

    marketing["date"] = pd.to_datetime(
        marketing["date"]
    ).dt.normalize()

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    ).dt.normalize()

    deliveries["date"] = pd.to_datetime(
        deliveries["date"]
    ).dt.normalize()

    return (
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    )


# ============================================================
# SALES KPIs
# ============================================================

def calculate_sales_kpis(
    orders,
    group_columns,
):

    grouped = (
        orders
        .groupby(group_columns)
        .agg(
            revenue=(
                "net_revenue",
                "sum"
            ),

            orders=(
                "order_id",
                "nunique"
            ),

            gross_sales=(
                "gross_sales",
                "sum"
            ),

            discount_amount=(
                "discount_amount",
                "sum"
            ),

            returned_orders=(
                "return_flag",
                "sum"
            ),

            return_amount=(
                "return_amount",
                "sum"
            ),

            cogs=(
                "cost_of_goods_sold",
                "sum"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Derived KPIs
    # --------------------------------------------------------

    grouped["aov"] = (
        grouped["revenue"]
        / grouped["orders"].replace(0, pd.NA)
    )

    grouped["discount_rate"] = (
        grouped["discount_amount"]
        / grouped["gross_sales"].replace(
            0,
            pd.NA
        )
    )

    grouped["return_rate"] = (
        grouped["returned_orders"]
        / grouped["orders"].replace(
            0,
            pd.NA
        )
    )

    grouped["gross_margin"] = (
        (
            grouped["revenue"]
            - grouped["cogs"]
        )
        / grouped["revenue"].replace(
            0,
            pd.NA
        )
    )

    return grouped


# ============================================================
# CONVERSION KPI
# ============================================================

def calculate_conversion_kpis(
    sessions,
    group_columns,
):

    grouped = (
        sessions
        .groupby(group_columns)
        .agg(
            sessions=(
                "session_id",
                "nunique"
            ),

            converted_sessions=(
                "converted",
                "sum"
            ),
        )
        .reset_index()
    )

    grouped["conversion_rate"] = (
        grouped["converted_sessions"]
        / grouped["sessions"].replace(
            0,
            pd.NA
        )
    )

    return grouped


# ============================================================
# MARKETING KPI
# ============================================================

def calculate_marketing_kpis(
    marketing,
    group_columns,
):

    grouped = (
        marketing
        .groupby(group_columns)
        .agg(
            marketing_spend=(
                "spend",
                "sum"
            ),

            impressions=(
                "impressions",
                "sum"
            ),

            clicks=(
                "clicks",
                "sum"
            ),
        )
        .reset_index()
    )

    grouped["ctr"] = (
        grouped["clicks"]
        / grouped["impressions"].replace(
            0,
            pd.NA
        )
    )

    return grouped


# ============================================================
# INVENTORY KPI
# ============================================================

def calculate_inventory_kpis(
    inventory,
    group_columns,
):

    grouped = (
        inventory
        .groupby(group_columns)
        .agg(
            demanded_units=(
                "demanded_units",
                "sum"
            ),

            available_units=(
                "available_units",
                "sum"
            ),

            inventory_units=(
                "inventory_units",
                "sum"
            ),

            stockout_events=(
                "stockout_flag",
                "sum"
            ),
        )
        .reset_index()
    )

    grouped["inventory_availability"] = (
        grouped["available_units"]
        / grouped["demanded_units"].replace(
            0,
            pd.NA
        )
    )

    return grouped


# ============================================================
# DELIVERY KPI
# ============================================================

def calculate_delivery_kpis(
    deliveries,
    group_columns,
):

    grouped = (
        deliveries
        .groupby(group_columns)
        .agg(
            deliveries=(
                "delivery_id",
                "nunique"
            ),

            delayed_deliveries=(
                "delayed_flag",
                "sum"
            ),

            total_delay_hours=(
                "delivery_delay_hours",
                "sum"
            ),
        )
        .reset_index()
    )

    grouped["delivery_delay_rate"] = (
        grouped["delayed_deliveries"]
        / grouped["deliveries"].replace(
            0,
            pd.NA
        )
    )

    grouped["average_delay_hours"] = (
        grouped["total_delay_hours"]
        / grouped["deliveries"].replace(
            0,
            pd.NA
        )
    )

    return grouped


# ============================================================
# MERGE KPI TABLES
# ============================================================

def merge_kpis(
    sales_kpis,
    conversion_kpis,
    marketing_kpis,
    inventory_kpis,
    delivery_kpis,
    group_columns,
):

    result = sales_kpis.copy()

    result = result.merge(
        conversion_kpis,
        on=group_columns,
        how="outer",
    )

    result = result.merge(
        marketing_kpis,
        on=group_columns,
        how="outer",
    )

    result = result.merge(
        inventory_kpis,
        on=group_columns,
        how="outer",
    )

    result = result.merge(
        delivery_kpis,
        on=group_columns,
        how="outer",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        group_columns
    ).reset_index(drop=True)

    return result


# ============================================================
# DAILY KPIs
# ============================================================

def calculate_daily_kpis(
    orders,
    sessions,
    marketing,
    inventory,
    deliveries,
):

    print("\nCalculating daily KPIs...")

    group_columns = ["date"]

    sales = calculate_sales_kpis(
        orders,
        group_columns
    )

    conversion = calculate_conversion_kpis(
        sessions,
        group_columns
    )

    marketing_kpis = calculate_marketing_kpis(
        marketing,
        group_columns
    )

    inventory_kpis = calculate_inventory_kpis(
        inventory,
        group_columns
    )

    delivery = calculate_delivery_kpis(
        deliveries,
        group_columns
    )

    daily_kpis = merge_kpis(
        sales,
        conversion,
        marketing_kpis,
        inventory_kpis,
        delivery,
        group_columns,
    )

    daily_kpis.to_parquet(
        PROCESSED_DIR / "daily_kpis.parquet",
        index=False
    )

    print(
        f"✓ daily_kpis.parquet "
        f"({len(daily_kpis):,} rows)"
    )

    return daily_kpis


# ============================================================
# REGIONAL KPIs
# ============================================================

def calculate_regional_kpis(
    orders,
    sessions,
    marketing,
    inventory,
    deliveries,
):

    print("\nCalculating regional KPIs...")

    group_columns = [
        "date",
        "region",
    ]

    sales = calculate_sales_kpis(
        orders,
        group_columns
    )

    conversion = calculate_conversion_kpis(
        sessions,
        group_columns
    )

    marketing_kpis = calculate_marketing_kpis(
        marketing,
        group_columns
    )

    inventory_kpis = calculate_inventory_kpis(
        inventory,
        group_columns
    )

    delivery = calculate_delivery_kpis(
        deliveries,
        group_columns
    )

    regional_kpis = merge_kpis(
        sales,
        conversion,
        marketing_kpis,
        inventory_kpis,
        delivery,
        group_columns,
    )

    regional_kpis.to_parquet(
        PROCESSED_DIR / "regional_kpis.parquet",
        index=False
    )

    print(
        f"✓ regional_kpis.parquet "
        f"({len(regional_kpis):,} rows)"
    )

    return regional_kpis


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("KPI CALCULATION ENGINE")
    print("=" * 70)

    (
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    ) = load_raw_data()

    (
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    ) = prepare_data(
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    )

    daily_kpis = calculate_daily_kpis(
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    )

    regional_kpis = calculate_regional_kpis(
        orders,
        sessions,
        marketing,
        inventory,
        deliveries,
    )

    print("\n" + "=" * 70)
    print("KPI CALCULATION COMPLETE")
    print("=" * 70)

    print("\nDaily KPI sample:")
    print(
        daily_kpis[
            [
                "date",
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
        ].head(10).to_string(index=False)
    )

    print("\nRegional KPI sample:")
    print(
        regional_kpis[
            [
                "date",
                "region",
                "revenue",
                "orders",
                "aov",
                "conversion_rate",
                "inventory_availability",
                "delivery_delay_rate",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nProcessed files created:")
    print(
        f"  {PROCESSED_DIR / 'daily_kpis.parquet'}"
    )
    print(
        f"  {PROCESSED_DIR / 'regional_kpis.parquet'}"
    )


if __name__ == "__main__":
    main()