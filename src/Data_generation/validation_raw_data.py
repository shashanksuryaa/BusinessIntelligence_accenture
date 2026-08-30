from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "Data_raw"


# ============================================================
# EXPECTED FILES
# ============================================================

EXPECTED_FILES = {
    "customers": "customers.parquet",
    "marketing": "marketing.parquet",
    "inventory": "inventory.parquet",
    "sales_orders": "sales_orders.parquet",
    "web_sessions": "web_sessions.parquet",
    "deliveries": "deliveries.parquet",
}


# ============================================================
# VALIDATION RESULT
# ============================================================

class ValidationReport:

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def success(self, message):
        self.passed.append(message)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(report):

    data = {}

    print("\n" + "=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    for name, filename in EXPECTED_FILES.items():

        path = RAW_DIR / filename

        if not path.exists():
            report.error(
                f"{filename}: FILE NOT FOUND"
            )
            continue

        try:

            df = pd.read_parquet(path)

            if df.empty:
                report.error(
                    f"{filename}: dataset is EMPTY"
                )
                continue

            data[name] = df

            print(
                f"✓ {filename:<30}"
                f"{len(df):>12,} rows"
            )

            report.success(
                f"{filename} loaded successfully"
            )

        except Exception as e:

            report.error(
                f"{filename}: failed to read - {e}"
            )

    return data


# ============================================================
# COLUMN VALIDATION
# ============================================================

EXPECTED_COLUMNS = {

    "customers": [
        "customer_id",
        "signup_date",
        "region",
        "customer_segment",
        "acquisition_channel",
    ],

    "marketing": [
        "campaign_id",
        "date",
        "region",
        "channel",
        "spend",
        "impressions",
        "clicks",
    ],

    "inventory": [
        "date",
        "region",
        "category",
        "warehouse_id",
        "demanded_units",
        "available_units",
        "inventory_units",
        "stockout_flag",
    ],

    "sales_orders": [
        "order_id",
        "customer_id",
        "timestamp",
        "date",
        "region",
        "category",
        "quantity",
        "unit_price",
        "discount_amount",
        "gross_sales",
        "net_revenue",
        "order_status",
        "return_flag",
        "return_amount",
    ],

    "web_sessions": [
        "session_id",
        "customer_id",
        "timestamp",
        "date",
        "region",
        "product_category",
        "converted",
        "order_id",
    ],

    "deliveries": [
        "delivery_id",
        "order_id",
        "date",
        "region",
        "warehouse_id",
        "promised_delivery_time",
        "actual_delivery_time",
        "delivery_delay_hours",
        "delayed_flag",
    ],
}


def validate_columns(data, report):

    print("\n" + "=" * 70)
    print("COLUMN VALIDATION")
    print("=" * 70)

    for name, expected in EXPECTED_COLUMNS.items():

        if name not in data:
            continue

        actual = set(data[name].columns)

        missing = set(expected) - actual

        if missing:

            report.error(
                f"{name}: missing columns "
                f"{sorted(missing)}"
            )

        else:

            print(
                f"✓ {name:<20} columns OK"
            )

            report.success(
                f"{name}: required columns present"
            )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validate_duplicates(data, report):

    print("\n" + "=" * 70)
    print("DUPLICATE VALIDATION")
    print("=" * 70)

    unique_keys = {
        "customers": "customer_id",
        "marketing": "campaign_id",
        "sales_orders": "order_id",
        "web_sessions": "session_id",
        "deliveries": "delivery_id",
    }

    for name, key in unique_keys.items():

        if name not in data:
            continue

        df = data[name]

        if key not in df.columns:
            continue

        duplicates = df[key].duplicated().sum()

        if duplicates > 0:

            report.error(
                f"{name}: {duplicates:,} "
                f"duplicate {key}s"
            )

            print(
                f"✗ {name:<20}"
                f"{duplicates:,} duplicates"
            )

        else:

            print(
                f"✓ {name:<20}"
                f"no duplicate {key}s"
            )

            report.success(
                f"{name}: unique {key}"
            )


# ============================================================
# NULL VALIDATION
# ============================================================

def validate_nulls(data, report):

    print("\n" + "=" * 70)
    print("NULL VALIDATION")
    print("=" * 70)

    # These fields should never be null.
    required_fields = {

        "customers": [
            "customer_id",
            "region",
            "customer_segment",
        ],

        "marketing": [
            "campaign_id",
            "date",
            "region",
            "spend",
        ],

        "inventory": [
            "date",
            "region",
            "category",
            "available_units",
        ],

        "sales_orders": [
            "order_id",
            "customer_id",
            "timestamp",
            "region",
            "category",
            "net_revenue",
        ],

        "web_sessions": [
            "session_id",
            "customer_id",
            "timestamp",
            "region",
        ],

        "deliveries": [
            "delivery_id",
            "order_id",
            "date",
            "region",
        ],
    }

    for name, columns in required_fields.items():

        if name not in data:
            continue

        df = data[name]

        for column in columns:

            if column not in df.columns:
                continue

            null_count = df[column].isna().sum()

            if null_count > 0:

                report.error(
                    f"{name}.{column}: "
                    f"{null_count:,} null values"
                )

                print(
                    f"✗ {name}.{column}: "
                    f"{null_count:,} nulls"
                )

            else:

                report.success(
                    f"{name}.{column}: no nulls"
                )


# ============================================================
# RANGE VALIDATION
# ============================================================

def validate_ranges(data, report):

    print("\n" + "=" * 70)
    print("VALUE RANGE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Sales orders
    # --------------------------------------------------------

    if "sales_orders" in data:

        df = data["sales_orders"]

        checks = {

            "quantity": (
                df["quantity"] > 0
            ),

            "unit_price": (
                df["unit_price"] >= 0
            ),

            "discount_amount": (
                df["discount_amount"] >= 0
            ),

            "gross_sales": (
                df["gross_sales"] >= 0
            ),

            "net_revenue": (
                df["net_revenue"] >= 0
            ),

            "return_amount": (
                df["return_amount"] >= 0
            ),

            "return_flag": (
                df["return_flag"].isin([0, 1])
            ),
        }

        for column, valid in checks.items():

            invalid = (~valid).sum()

            if invalid:

                report.error(
                    f"sales_orders.{column}: "
                    f"{invalid:,} invalid values"
                )

                print(
                    f"✗ sales_orders.{column}: "
                    f"{invalid:,} invalid"
                )

            else:

                print(
                    f"✓ sales_orders.{column}"
                )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    if "inventory" in data:

        df = data["inventory"]

        checks = {

            "demanded_units": (
                df["demanded_units"] >= 0
            ),

            "available_units": (
                df["available_units"] >= 0
            ),

            "inventory_units": (
                df["inventory_units"] >= 0
            ),

            "stockout_flag": (
                df["stockout_flag"].isin([0, 1])
            ),
        }

        for column, valid in checks.items():

            invalid = (~valid).sum()

            if invalid:

                report.error(
                    f"inventory.{column}: "
                    f"{invalid:,} invalid values"
                )

                print(
                    f"✗ inventory.{column}: "
                    f"{invalid:,} invalid"
                )

            else:

                print(
                    f"✓ inventory.{column}"
                )

    # --------------------------------------------------------
    # Marketing
    # --------------------------------------------------------

    if "marketing" in data:

        df = data["marketing"]

        checks = {

            "spend": (
                df["spend"] >= 0
            ),

            "impressions": (
                df["impressions"] >= 0
            ),

            "clicks": (
                df["clicks"] >= 0
            ),

        }

        for column, valid in checks.items():

            invalid = (~valid).sum()

            if invalid:

                report.error(
                    f"marketing.{column}: "
                    f"{invalid:,} invalid values"
                )

                print(
                    f"✗ marketing.{column}: "
                    f"{invalid:,} invalid"
                )

            else:

                print(
                    f"✓ marketing.{column}"
                )


# ============================================================
# REFERENTIAL INTEGRITY
# ============================================================

def validate_relationships(data, report):

    print("\n" + "=" * 70)
    print("REFERENTIAL INTEGRITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Orders → Customers
    # --------------------------------------------------------

    if (
        "sales_orders" in data
        and "customers" in data
    ):

        order_customers = set(
            data["sales_orders"]["customer_id"]
        )

        customer_ids = set(
            data["customers"]["customer_id"]
        )

        missing = (
            order_customers
            - customer_ids
        )

        if missing:

            report.error(
                f"sales_orders: "
                f"{len(missing):,} customer IDs "
                f"not found in customers"
            )

            print(
                f"✗ Orders → Customers: "
                f"{len(missing):,} missing"
            )

        else:

            print(
                "✓ Orders → Customers"
            )

            report.success(
                "Orders reference valid customers"
            )

    # --------------------------------------------------------
    # Deliveries → Orders
    # --------------------------------------------------------

    if (
        "deliveries" in data
        and "sales_orders" in data
    ):

        delivery_orders = set(
            data["deliveries"]["order_id"]
        )

        order_ids = set(
            data["sales_orders"]["order_id"]
        )

        missing = (
            delivery_orders
            - order_ids
        )

        if missing:

            report.error(
                f"deliveries: "
                f"{len(missing):,} order IDs "
                f"not found in sales_orders"
            )

            print(
                f"✗ Deliveries → Orders: "
                f"{len(missing):,} missing"
            )

        else:

            print(
                "✓ Deliveries → Orders"
            )

            report.success(
                "Deliveries reference valid orders"
            )

    # --------------------------------------------------------
    # Converted sessions → Orders
    # --------------------------------------------------------

    if (
        "web_sessions" in data
        and "sales_orders" in data
    ):

        converted = data["web_sessions"][
            data["web_sessions"]["converted"] == 1
        ]

        session_orders = set(
            converted["order_id"].dropna()
        )

        order_ids = set(
            data["sales_orders"]["order_id"]
        )

        missing = (
            session_orders
            - order_ids
        )

        if missing:

            report.warning(
                f"web_sessions: "
                f"{len(missing):,} converted "
                f"session order IDs don't exist "
                f"in sales_orders"
            )

            print(
                f"⚠ Converted Sessions → Orders: "
                f"{len(missing):,} missing"
            )

        else:

            print(
                "✓ Converted Sessions → Orders"
            )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_dates(data, report):

    print("\n" + "=" * 70)
    print("DATE VALIDATION")
    print("=" * 70)

    date_columns = {
        "customers": "signup_date",
        "marketing": "date",
        "inventory": "date",
        "sales_orders": "timestamp",
        "web_sessions": "timestamp",
        "deliveries": "date",
    }

    global_min = None
    global_max = None

    for name, column in date_columns.items():

        if name not in data:
            continue

        dates = pd.to_datetime(
            data[name][column],
            errors="coerce"
        )

        invalid = dates.isna().sum()

        if invalid:

            report.error(
                f"{name}.{column}: "
                f"{invalid:,} invalid dates"
            )

            print(
                f"✗ {name}.{column}: "
                f"{invalid:,} invalid dates"
            )

            continue

        minimum = dates.min()
        maximum = dates.max()

        print(
            f"✓ {name:<20}"
            f"{minimum.date()} → "
            f"{maximum.date()}"
        )

        report.success(
            f"{name}: valid dates"
        )

        if global_min is None:
            global_min = minimum
            global_max = maximum

        else:
            global_min = min(
                global_min,
                minimum
            )

            global_max = max(
                global_max,
                maximum
            )

    if global_min is not None:

        print(
            "\nOverall data coverage:"
        )

        print(
            f"{global_min.date()} → "
            f"{global_max.date()}"
        )


# ============================================================
# BUSINESS SANITY CHECKS
# ============================================================

def validate_business_sanity(data, report):

    print("\n" + "=" * 70)
    print("BUSINESS SANITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------------
    # Conversion rate
    # --------------------------------------------------------

    if "web_sessions" in data:

        sessions = data["web_sessions"]

        conversion_rate = (
            sessions["converted"].mean()
        )

        print(
            f"Overall conversion rate: "
            f"{conversion_rate:.2%}"
        )

        if not 0.001 <= conversion_rate <= 0.50:

            report.warning(
                "Conversion rate looks unusual: "
                f"{conversion_rate:.2%}"
            )

        else:

            report.success(
                "Conversion rate within plausible range"
            )

    # --------------------------------------------------------
    # AOV
    # --------------------------------------------------------

    if "sales_orders" in data:

        orders = data["sales_orders"]

        total_revenue = (
            orders["net_revenue"].sum()
        )

        total_orders = len(orders)

        aov = (
            total_revenue / total_orders
            if total_orders > 0
            else 0
        )

        print(
            f"AOV: ₹{aov:,.2f}"
        )

        if aov <= 0:

            report.error(
                "AOV is zero or negative"
            )

        else:

            report.success(
                "AOV is positive"
            )

    # --------------------------------------------------------
    # Return rate
    # --------------------------------------------------------

    if "sales_orders" in data:

        orders = data["sales_orders"]

        return_rate = (
            orders["return_flag"].mean()
        )

        print(
            f"Return rate: "
            f"{return_rate:.2%}"
        )

        if not 0 <= return_rate <= 1:

            report.error(
                "Return rate outside [0,1]"
            )

        else:

            report.success(
                "Return rate within valid range"
            )

    # --------------------------------------------------------
    # Delivery delay
    # --------------------------------------------------------

    if "deliveries" in data:

        deliveries = data["deliveries"]

        delay_rate = (
            deliveries["delayed_flag"].mean()
        )

        print(
            f"Delivery delay rate: "
            f"{delay_rate:.2%}"
        )

        if not 0 <= delay_rate <= 1:

            report.error(
                "Delivery delay rate outside [0,1]"
            )

        else:

            report.success(
                "Delivery delay rate within valid range"
            )


# ============================================================
# FINAL REPORT
# ============================================================

def print_final_report(report):

    print("\n" + "=" * 70)
    print("FINAL DATA QUALITY REPORT")
    print("=" * 70)

    print(
        f"\nPassed checks : "
        f"{len(report.passed)}"
    )

    print(
        f"Warnings      : "
        f"{len(report.warnings)}"
    )

    print(
        f"Errors        : "
        f"{len(report.errors)}"
    )

    if report.warnings:

        print("\nWARNINGS")

        for warning in report.warnings:

            print(
                f"⚠ {warning}"
            )

    if report.errors:

        print("\nERRORS")

        for error in report.errors:

            print(
                f"✗ {error}"
            )

    print("\n" + "-" * 70)

    if report.errors:

        print(
            "DATA QUALITY GATE: FAILED"
        )

        print(
            "Do not proceed to KPI processing."
        )

        return False

    else:

        print(
            "DATA QUALITY GATE: PASSED"
        )

        print(
            "Raw data is ready for KPI processing."
        )

        return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RAW DATA VALIDATION")
    print("=" * 70)

    report = ValidationReport()

    data = load_data(report)

    validate_columns(
        data,
        report
    )

    validate_duplicates(
        data,
        report
    )

    validate_nulls(
        data,
        report
    )

    validate_ranges(
        data,
        report
    )

    validate_relationships(
        data,
        report
    )

    validate_dates(
        data,
        report
    )

    validate_business_sanity(
        data,
        report
    )

    passed = print_final_report(
        report
    )

    return passed


if __name__ == "__main__":

    success = main()

    if not success:
        raise SystemExit(1)