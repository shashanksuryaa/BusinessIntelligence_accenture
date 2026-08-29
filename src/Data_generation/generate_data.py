import os
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Scale down the configured business volume so the prototype
# remains manageable on a local machine.
DATA_SCALE = 0.05

ROOT = Path(__file__).resolve().parents[2]

SEMANTIC_DIR = ROOT / "Semantic"
RAW_DIR = ROOT / "Data_raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD YAML CONFIGURATION
# ============================================================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


business_config = load_yaml(
    SEMANTIC_DIR / "Business_config.yaml"
)

scenario_config = load_yaml(
    SEMANTIC_DIR / "Senario_config.yaml"
)

business = business_config["business"]
scenarios = scenario_config["scenarios"]


# ============================================================
# BASIC CONFIG VALUES
# ============================================================

START_DATE = pd.Timestamp(
    business["simulation"]["start_date"]
)

END_DATE = pd.Timestamp(
    business["simulation"]["end_date"]
)

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

REGIONS = list(business["regions"].keys())
CATEGORIES = list(business["categories"].keys())
SEGMENTS = list(business["customer_segments"].keys())

# Smart_Home exists only in the sparse-history scenario.
# Add it dynamically so the generator can create it.
for scenario in scenarios.values():

    category_scope = (
        scenario.get("scope", {}).get("category", [])
    )

    for category in category_scope:
        if category not in CATEGORIES:
            CATEGORIES.append(category)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(items, weights):
    return random.choices(
        items,
        weights=weights,
        k=1
    )[0]


def get_region():
    weights = [
        business["regions"][region]["population_weight"]
        for region in REGIONS
    ]

    return weighted_choice(REGIONS, weights)


def get_category():

    weights = []

    for category in CATEGORIES:

        # Existing baseline category
        if category in business["categories"]:
            proportion = business["categories"][category].get(
                "proportion",
                0.01
            )

        # Newly launched category
        else:
            proportion = 0.01

        weights.append(proportion)

    return weighted_choice(
        CATEGORIES,
        weights
    )


def get_segment():
    items = SEGMENTS

    weights = [
        business["customer_segments"][segment]["proportion"]
        for segment in items
    ]

    return weighted_choice(items, weights)


def get_seasonality(date):
    """
    Combines weekly and monthly seasonality.
    """

    day_name = date.day_name()
    month_name = date.month_name()

    weekly = business["seasonality"]["weekly"].get(
        day_name,
        1.0
    )

    monthly = business["seasonality"]["monthly"].get(
        month_name,
        1.0
    )

    return weekly * monthly


def active_scenario_effects(date, region=None, category=None):
    """
    Returns all scenario effects active on a given date.

    IMPORTANT:
    This function is only used by the synthetic-data generator.
    The BI system will never see scenario_config.yaml.
    """

    effects = {}

    for scenario_name, scenario in scenarios.items():

        start = pd.Timestamp(scenario["start_date"])
        end = pd.Timestamp(scenario["end_date"])

        if not (start <= date <= end):
            continue

        scope = scenario.get("scope", {})

        scoped_regions = scope.get("region", [])
        scoped_categories = scope.get("category", [])

        # Region filtering
        if scoped_regions:

            if "all" not in scoped_regions:

                if region not in scoped_regions:
                    continue

        # Category filtering
        if scoped_categories:

            if category not in scoped_categories:
                continue

        for metric, effect in scenario.get(
            "effects", {}
        ).items():

            if "change" in effect:

                effects[metric] = (
                    effects.get(metric, 0.0)
                    + effect["change"]
                )

            elif "multiplier" in effect:

                current = effects.get(metric, 0.0)

                effects[metric] = (
                    (1 + current) *
                    effect["multiplier"]
                    - 1
                )

    return effects


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


# ============================================================
# 1. CUSTOMERS
# ============================================================

def generate_customers():

    print("Generating customers...")

    initial_customers = business["customers"][
        "initial_customers"
    ]

    daily_new_customers = int(
        business["customers"]["new_customers_per_day"]
        * DATA_SCALE
    )

    customer_rows = []

    customer_counter = 1

    # Initial customer base
    for _ in range(initial_customers):

        customer_id = f"C{customer_counter:07d}"
        customer_counter += 1

        signup_date = START_DATE - pd.Timedelta(
            days=np.random.randint(1, 365)
        )

        region = get_region()
        segment = get_segment()

        customer_rows.append({
            "customer_id": customer_id,
            "signup_date": signup_date,
            "region": region,
            "customer_segment": segment,
            "acquisition_channel": random.choice([
                "Search",
                "Social",
                "Email",
                "Affiliate"
            ])
        })

    # New customers during simulation
    for date in DATES:

        for _ in range(daily_new_customers):

            customer_id = f"C{customer_counter:07d}"
            customer_counter += 1

            customer_rows.append({
                "customer_id": customer_id,
                "signup_date": date,
                "region": get_region(),
                "customer_segment": "New",
                "acquisition_channel": random.choice([
                    "Search",
                    "Social",
                    "Email",
                    "Affiliate"
                ])
            })

    customers = pd.DataFrame(customer_rows)

    customers.to_parquet(
        RAW_DIR / "customers.parquet",
        index=False
    )

    print(
        f"Customers generated: {len(customers):,}"
    )

    return customers


# ============================================================
# 2. MARKETING
# ============================================================

def generate_marketing():

    print("Generating marketing data...")

    channels = business["marketing"]["channels"]

    rows = []

    campaign_counter = 1

    for date in DATES:

        seasonality = get_seasonality(date)

        daily_budget = (
            business["marketing"]["daily_budget"]
            * DATA_SCALE
            * seasonality
        )

        for region in REGIONS:

            for channel, config in channels.items():

                budget = (
                    daily_budget
                    * config["budget_share"]
                    * business["regions"][region][
                        "demand_multiplier"
                    ]
                )

                # Apply scenario effects
                effects = active_scenario_effects(
                    date=date,
                    region=region
                )

                if "marketing_spend" in effects:

                    budget *= (
                        1 + effects["marketing_spend"]
                    )

                budget = max(0, budget)

                cpc = config["base_cpc"]

                clicks = max(
                    1,
                    int(
                        budget /
                        max(cpc, 0.1)
                    )
                )

                impressions = int(
                    clicks * np.random.uniform(
                        15,
                        30
                    )
                )

                campaign_id = (
                    f"CAM{campaign_counter:06d}"
                )

                campaign_counter += 1

                rows.append({
                    "campaign_id": campaign_id,
                    "date": date,
                    "region": region,
                    "channel": channel,
                    "campaign_type": "always_on",
                    "spend": round(budget, 2),
                    "impressions": impressions,
                    "clicks": clicks
                })

    marketing = pd.DataFrame(rows)

    marketing.to_parquet(
        RAW_DIR / "marketing.parquet",
        index=False
    )

    print(
        f"Marketing rows generated: {len(marketing):,}"
    )

    return marketing


# ============================================================
# 3. INVENTORY
# ============================================================

def generate_inventory():

    print("Generating inventory data...")

    rows = []

    category_availability = (
        business["inventory"]["by_category"]
    )

    for date in DATES:

        for region in REGIONS:

            for category in CATEGORIES:

                base = category_availability.get(
                    category,
                    business["inventory"][
                        "base_availability"
                    ]
                )

                effects = active_scenario_effects(
                    date=date,
                    region=region,
                    category=category
                )

                availability = (
                    base
                    + effects.get(
                        "inventory_availability",
                        0
                    )
                )

                availability = clamp(
                    availability
                    + np.random.normal(0, 0.01),
                    0.50,
                    1.00
                )

                # Approximate demand
                demanded_units = int(
                    np.random.poisson(300)
                )

                available_units = int(
                    demanded_units
                    * availability
                )

                inventory_units = int(
                    available_units
                    * np.random.uniform(
                        0.8,
                        1.2
                    )
                )

                rows.append({
                    "date": date,
                    "region": region,
                    "category": category,
                    "warehouse_id": (
                        f"W_{region}"
                    ),
                    "demanded_units":
                        demanded_units,
                    "available_units":
                        available_units,
                    "inventory_units":
                        inventory_units,
                    "stockout_flag":
                        int(availability < 0.80),
                    "replenishment_units":
                        int(
                            demanded_units
                            * business["inventory"][
                                "replenishment_rate"
                            ]
                        )
                })

    inventory = pd.DataFrame(rows)

    inventory.to_parquet(
        RAW_DIR / "inventory.parquet",
        index=False
    )

    print(
        f"Inventory rows generated: {len(inventory):,}"
    )

    return inventory


# ============================================================
# 4. WEB SESSIONS
# ============================================================

def generate_sessions(customers):

    print("Generating web sessions...")

    rows = []

    session_counter = 1

    # Customer lookup by region
    customer_by_region = {
        region:
        customers[
            customers["region"] == region
        ]["customer_id"].tolist()

        for region in REGIONS
    }

    base_sessions = int(
        business["traffic"]["base_daily_sessions"]
        * DATA_SCALE
    )

    region_conversion = (
        business["traffic"]["conversion_rate"][
            "by_region"
        ]
    )

    for date in DATES:

        seasonality = get_seasonality(date)

        for region in REGIONS:

            region_multiplier = business[
                "regions"
            ][region]["demand_multiplier"]

            session_count = int(
                base_sessions
                * business["regions"][
                    region
                ]["population_weight"]
                * region_multiplier
                * seasonality
            )

            effects = active_scenario_effects(
                date=date,
                region=region
            )

            conversion_rate = region_conversion[
                region
            ]

            conversion_rate += effects.get(
                "conversion_rate",
                0
            )

            conversion_rate = clamp(
                conversion_rate,
                0.005,
                0.50
            )

            region_customers = (
                customer_by_region[region]
            )

            for _ in range(session_count):

                session_id = (
                    f"S{session_counter:09d}"
                )

                session_counter += 1

                customer_id = (
                    random.choice(
                        region_customers
                    )
                )

                category = get_category()

                device = random.choice([
                    "mobile",
                    "desktop",
                    "tablet"
                ])

                traffic_source = random.choice([
                    "Organic",
                    "Search",
                    "Social",
                    "Email",
                    "Affiliate",
                    "Direct"
                ])

                converted = (
                    np.random.random()
                    < conversion_rate
                )

                order_id = None

                if converted:
                    order_id = (
                        f"O{session_counter:09d}"
                    )

                rows.append({
                    "session_id":
                        session_id,
                    "customer_id":
                        customer_id,
                    "timestamp":
                        date
                        + pd.Timedelta(
                            seconds=np.random.randint(
                                0,
                                86400
                            )
                        ),
                    "date":
                        date,
                    "region":
                        region,
                    "device":
                        device,
                    "traffic_source":
                        traffic_source,
                    "campaign_id":
                        None,
                    "product_category":
                        category,
                    "converted":
                        int(converted),
                    "order_id":
                        order_id
                })

    sessions = pd.DataFrame(rows)

    sessions.to_parquet(
        RAW_DIR / "web_sessions.parquet",
        index=False
    )

    print(
        f"Web sessions generated: {len(sessions):,}"
    )

    return sessions


# ============================================================
# 5. SALES ORDERS
# ============================================================

def generate_orders(
    sessions,
    customers
):

    print("Generating sales orders...")

    converted_sessions = sessions[
        sessions["converted"] == 1
    ].copy()

    rows = []

    customer_lookup = customers.set_index(
        "customer_id"
    )

    category_config = business["categories"]

    for idx, session in converted_sessions.iterrows():

        date = pd.Timestamp(
            session["date"]
        )

        region = session["region"]
        category = session["product_category"]

        if category not in category_config:

            # New category
            base_price = 1000
            base_cost = 600

        else:

            base_price = category_config[
                category
            ]["average_price"]

            base_cost = category_config[
                category
            ]["average_cost"]

        effects = active_scenario_effects(
            date=date,
            region=region,
            category=category
        )

        quantity = max(
            1,
            int(
                np.random.poisson(
                    business["orders"][
                        "average_items_per_order"
                    ]
                )
            )
        )

        unit_price = max(
            50,
            np.random.normal(
                base_price,
                base_price * 0.15
            )
        )

        discount_rate = (
            business["pricing"][
                "discount_rate"
            ]["by_category"].get(
                category,
                business["pricing"][
                    "discount_rate"
                ]["base"]
            )
        )

        discount_rate += effects.get(
            "discount_rate",
            0
        )

        discount_rate = clamp(
            discount_rate,
            0,
            0.50
        )

        gross_sales = (
            quantity * unit_price
        )

        discount_amount = (
            gross_sales * discount_rate
        )

        net_revenue = (
            gross_sales - discount_amount
        )

        cogs = (
            quantity
            * np.random.normal(
                base_cost,
                base_cost * 0.08
            )
        )

        customer_id = session[
            "customer_id"
        ]

        rows.append({
            "order_id":
                session["order_id"],

            "customer_id":
                customer_id,

            "timestamp":
                session["timestamp"],

            "date":
                date,

            "region":
                region,

            "category":
                category,

            "product_id":
                f"P_{category[:3].upper()}_"
                f"{np.random.randint(1, 101):03d}",

            "quantity":
                quantity,

            "unit_price":
                round(unit_price, 2),

            "discount_amount":
                round(discount_amount, 2),

            "gross_sales":
                round(gross_sales, 2),

            "net_revenue":
                round(net_revenue, 2),

            "cost_of_goods_sold":
                round(max(cogs, 0), 2),

            "order_status":
                "completed",

            "return_flag":
                0,

            "return_amount":
                0.0
        })

    orders = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    for category in orders["category"].unique():

        category_return_rate = (
            business["returns"][
                "by_category"
            ].get(
                category,
                business["returns"][
                    "base_return_rate"
                ]
            )
        )

        mask = (
            orders["category"] == category
        )

        return_flags = (
            np.random.random(mask.sum())
            < category_return_rate
        )

        orders.loc[
            mask,
            "return_flag"
        ] = return_flags.astype(int)

        orders.loc[
            mask & (
                orders["return_flag"] == 1
            ),
            "return_amount"
        ] = orders.loc[
            mask & (
                orders["return_flag"] == 1
            ),
            "net_revenue"
        ]

    orders.to_parquet(
        RAW_DIR / "sales_orders.parquet",
        index=False
    )

    print(
        f"Sales orders generated: {len(orders):,}"
    )

    return orders


# ============================================================
# 6. DELIVERIES
# ============================================================

def generate_deliveries(orders):

    print("Generating delivery data...")

    rows = []

    base_delay_rate = business[
        "delivery"
    ]["base_delay_rate"]

    regional_delay = business[
        "delivery"
    ]["by_region"]

    for _, order in orders.iterrows():

        date = pd.Timestamp(
            order["date"]
        )

        region = order["region"]

        effects = active_scenario_effects(
            date=date,
            region=region
        )

        delay_rate = regional_delay.get(
            region,
            base_delay_rate
        )

        delay_rate += effects.get(
            "delivery_delay_rate",
            0
        )

        delay_rate = clamp(
            delay_rate,
            0,
            0.95
        )

        delayed = (
            np.random.random()
            < delay_rate
        )

        promised_days = max(
            1,
            int(
                np.random.normal(
                    business["delivery"][
                        "average_delivery_days"
                    ],
                    0.5
                )
            )
        )

        if delayed:

            delay_hours = max(
                1,
                int(
                    np.random.normal(
                        24,
                        12
                    )
                )
            )

        else:

            delay_hours = 0

        promised_time = (
            pd.Timestamp(
                order["timestamp"]
            )
            + pd.Timedelta(
                days=promised_days
            )
        )

        actual_time = (
            promised_time
            + pd.Timedelta(
                hours=delay_hours
            )
        )

        rows.append({
            "delivery_id":
                f"D{len(rows) + 1:09d}",

            "order_id":
                order["order_id"],

            "date":
                date,

            "region":
                region,

            "warehouse_id":
                f"W_{region}",

            "carrier":
                random.choice([
                    "Carrier_A",
                    "Carrier_B",
                    "Carrier_C"
                ]),

            "promised_delivery_time":
                promised_time,

            "actual_delivery_time":
                actual_time,

            "delivery_delay_hours":
                delay_hours,

            "delayed_flag":
                int(delayed)
        })

    deliveries = pd.DataFrame(rows)

    deliveries.to_parquet(
        RAW_DIR / "deliveries.parquet",
        index=False
    )

    print(
        f"Deliveries generated: {len(deliveries):,}"
    )

    return deliveries


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SYNTHETIC BUSINESS DATA GENERATOR")
    print("=" * 60)

    print(
        f"Simulation period: "
        f"{START_DATE.date()} → {END_DATE.date()}"
    )

    print(
        f"Data scale: {DATA_SCALE}"
    )

    print("=" * 60 + "\n")

    # 1. Customers
    customers = generate_customers()

    # 2. Marketing
    marketing = generate_marketing()

    # 3. Inventory
    inventory = generate_inventory()

    # 4. Web traffic
    sessions = generate_sessions(
        customers
    )

    # 5. Orders
    orders = generate_orders(
        sessions,
        customers
    )

    # 6. Deliveries
    deliveries = generate_deliveries(
        orders
    )

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")

    for file in sorted(
        RAW_DIR.glob("*.parquet")
    ):

        size_mb = (
            file.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"  {file.name:<30}"
            f"{size_mb:>8.2f} MB"
        )

    print("\n")


if __name__ == "__main__":
    main()