"""
data_loader.py — Load and validate all CSV datasets for the Datathon pipeline.

Returns a dictionary of DataFrames keyed by table name.
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Column dtypes for validation
_PARSE_DATES = {
    "sales": ["Date"],
    "sample_submission": ["Date"],
    "orders": ["order_date"],
    "customers": ["signup_date"],
    "shipments": ["ship_date", "delivery_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "promotions": ["start_date", "end_date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
}

_FILES = {
    "sales": "sales.csv",
    "sample_submission": "sample_submission.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments.csv",
    "shipments": "shipments.csv",
    "returns": "returns.csv",
    "reviews": "reviews.csv",
    "products": "products.csv",
    "customers": "customers.csv",
    "promotions": "promotions.csv",
    "geography": "geography.csv",
    "inventory": "inventory.csv",
    "web_traffic": "web_traffic.csv",
}


def load_all(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from *data_dir* and return as ``{name: DataFrame}``.

    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    data = {}
    for name, fname in _FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning("File not found, skipping: %s", path)
            continue

        date_cols = _PARSE_DATES.get(name, [])
        df = pd.read_csv(path, parse_dates=date_cols, low_memory=False)
        data[name] = df
        logger.info("Loaded %-20s  %7d rows x %2d cols", name, len(df), df.shape[1])

    _validate(data)
    return data


def _validate(data: dict[str, pd.DataFrame]) -> None:
    """Run basic validation checks on loaded data."""
    if "sales" in data:
        s = data["sales"]
        assert s["Date"].is_monotonic_increasing, "sales.csv dates not sorted!"
        assert s["Revenue"].gt(0).all(), "Negative revenue found!"
        logger.info("Sales date range: %s -> %s (%d days)",
                     s["Date"].min().date(), s["Date"].max().date(), len(s))

    if "sample_submission" in data:
        sub = data["sample_submission"]
        logger.info("Submission date range: %s -> %s (%d days)",
                     sub["Date"].min().date(), sub["Date"].max().date(), len(sub))

    if "orders" in data and "payments" in data:
        n_orders = data["orders"]["order_id"].nunique()
        n_payments = data["payments"]["order_id"].nunique()
        logger.info("Orders: %d unique | Payments: %d unique (1:1 check: %s)",
                     n_orders, n_payments, n_orders == n_payments)

    logger.info("Data validation passed [DONE]")
