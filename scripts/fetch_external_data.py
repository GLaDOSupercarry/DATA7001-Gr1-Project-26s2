#!/usr/bin/env python3
"""Download and aggregate the two external data sources.

Sources (both Queensland Government open data, CC-BY-4.0):
  1. "Fuel price reporting" 2023-2026 -- station-level retail fuel prices,
     published as one CSV per month. Rows are *price changes only*, so a
     station's price holds until its next reported change.
     https://www.data.qld.gov.au/dataset/fuel-price-reporting

  2. "Vehicle Registration New and Transfers" -- TMR vehicle registration
     transactions, one CSV per year. We keep TRANSACTION_TYPE == 'Registration New'
     as the flow analogue of new vehicle sales.
     https://www.data.qld.gov.au/dataset/vehicle-registration-new-and-transfers-test

Outputs:
  data/raw/     raw downloads (gitignored -- large)
  data/clean/fuel_price_monthly.csv          monthly mean retail price, cents/litre
  data/clean/vehicle_registrations_monthly.csv  monthly new registrations by fuel type
  data/raw/_manifest.csv                     provenance: what was downloaded, when
"""
from __future__ import annotations

import hashlib
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
RAW.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)

CKAN = "https://www.data.qld.gov.au/api/3/action/package_show"
FUEL_PACKAGES = [
    "fuel-price-reporting",       # 2023
    "fuel-price-reporting-2024",
    "fuel-price-reporting-2025",
    "fuel-price-reporting-2026",
]
VEHICLE_PACKAGE = "vehicle-registration-new-and-transfers-test"

# Fuel types worth keeping. LPG/e85/OPAL are a few hundred rows a month statewide
# and would add columns with almost no signal.
KEEP_FUEL_TYPES = ["Unleaded", "e10", "PULP 95/96 RON", "PULP 98 RON", "Diesel"]

# Sentinel in the source for "tank empty / temporarily unavailable".
UNAVAILABLE_PRICE = 9999

# The analysis window runs Dec 2023 -> May 2026. Fuel prices are "changes only",
# so prices set before the window must be carried in; we start fetching a few
# months early to give the forward-fill a baseline.
FUEL_START = pd.Timestamp("2023-09-01")
FUEL_END = pd.Timestamp("2026-09-01")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "DATA7001-coursework/1.0"})


def get(url: str, **kwargs) -> requests.Response:
    """GET with a couple of retries -- the portal drops connections occasionally."""
    last = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=180, **kwargs)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {url}") from last


def package_resources(package_id: str) -> list[dict]:
    payload = get(CKAN, params={"id": package_id}).json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN error for {package_id}: {payload}")
    return payload["result"]["resources"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The portal switched header style partway through: 'Site_Name' became
    'Site Name' from the March 2026 file onward. Fold both to one form."""
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}

MANIFEST: list[dict] = []


def fetch_fuel() -> pd.DataFrame:
    """Download each monthly station-level CSV; return concatenated price changes."""
    frames = []
    for pkg in FUEL_PACKAGES:
        for res in package_resources(pkg):
            name, url = res.get("name") or "", res.get("url") or ""
            if res.get("format") != "CSV" or "fuel prices" not in name.lower():
                continue
            m = re.search(r"(January|February|March|April|May|June|July|August|"
                          r"September|October|November|December)\s+(\d{4})", name)
            if not m:
                print(f"  ! could not parse month from {name!r}", file=sys.stderr)
                continue
            month = pd.Timestamp(year=int(m.group(2)), month=MONTHS[m.group(1)], day=1)
            # Note: the file named for month M holds *the previous month's*
            # changes (the portal's own files are called "last month's prices").
            # Only the filename matters for caching; each row carries its real
            # timestamp, and aggregation uses that, so the labels don't leak in.
            if not (FUEL_START <= month < FUEL_END):
                continue

            local = RAW / "fuel" / f"fuel-prices-{month:%Y-%m}.csv"
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                body = get(url).content
                local.write_bytes(body)
            else:
                body = local.read_bytes()
            MANIFEST.append({"source": "fuel", "period": f"{month:%Y-%m}",
                             "url": url, "bytes": len(body)})
            print(f"  fuel {month:%Y-%m}: {len(body)/1e6:.1f} MB")

            d = normalise_columns(
                pd.read_csv(io.BytesIO(body), encoding="utf-8-sig", low_memory=False)
            )
            d = d[["SiteId", "Fuel_Type", "Price", "TransactionDateutc"]]
            d["period"] = month
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    print(f"  fuel rows: {len(out):,}")
    return out


def monthly_fuel_prices(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse price-change events into a monthly mean price per fuel type.

    Prices are reported as changes only, so each event's price applies from its
    own timestamp until the next event for that station and fuel type. We
    forward-fill onto a daily grid, then average across days and stations.
    """
    events = events[events["Fuel_Type"].isin(KEEP_FUEL_TYPES)].copy()
    # Sentinel 9999/9998 means "temporarily unavailable"; a handful of stations
    # also report nonsense near zero. Keep only prices a real bowser could show
    # (80c to $4.00 per litre, in tenths of a cent).
    events = events[(events["Price"] >= 800) & (events["Price"] <= 4000)]
    events["ts"] = pd.to_datetime(
        events["TransactionDateutc"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    events = events.dropna(subset=["ts"])
    events["date"] = events["ts"].dt.normalize()
    # One price per station per fuel per day: the last change of that day.
    events = events.sort_values("ts").drop_duplicates(
        ["SiteId", "Fuel_Type", "date"], keep="last"
    )

    def to_daily(group: pd.DataFrame) -> pd.Series:
        s = group.set_index("date")["Price"].sort_index()
        return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()

    daily = (
        events.groupby(["SiteId", "Fuel_Type"], group_keys=True)
        .apply(to_daily, include_groups=False)
        .rename("Price")
        .reset_index()
        .rename(columns={"level_2": "date"})
    )
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    # Price is in tenths of a cent.
    stats = (
        daily.groupby(["month", "Fuel_Type"])
        .agg(stations=("SiteId", "nunique"),
             days=("Price", "size"),
             mean_tenths=("Price", "mean"))
        .reset_index()
    )
    stats["price_cpl"] = stats["mean_tenths"] / 10
    return stats[["month", "Fuel_Type", "stations", "days", "price_cpl"]].sort_values(
        ["Fuel_Type", "month"]
    ).reset_index(drop=True)


def fetch_vehicles() -> pd.DataFrame:
    """Download the yearly registration transaction CSVs."""
    frames = []
    for res in package_resources(VEHICLE_PACKAGE):
        name, url = res.get("name") or "", res.get("url") or ""
        if res.get("format") != "CSV":
            continue
        m = re.search(r"-\s*(\d{4})\s*-", name)
        if not m or int(m.group(1)) < 2023:
            continue
        year = int(m.group(1))
        # Stable across runs -- builtin hash() is salted per process and would
        # make every run re-download.
        tag = hashlib.md5(url.encode()).hexdigest()[:8]
        local = RAW / "vehicles" / f"vehicle-registration-{year}-{tag}.csv"
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            body = get(url).content
            local.write_bytes(body)
        else:
            body = local.read_bytes()
        MANIFEST.append({"source": "vehicles", "period": str(year),
                         "url": url, "bytes": len(body)})
        print(f"  vehicles {year} ({name[-12:]}): {len(body)/1e6:.1f} MB")
        # BADGE occasionally contains a backslash-escaped comma (e.g. "\,"),
        # which the default parser reads as a field separator.
        frames.append(
            normalise_columns(
                pd.read_csv(io.BytesIO(body), low_memory=False, escapechar="\\")
            )
        )
    return pd.concat(frames, ignore_index=True)


def monthly_registrations(reg: pd.DataFrame) -> pd.DataFrame:
    """New registrations per month, by fuel type (and total)."""
    reg["RECORD_DATE"] = pd.to_datetime(reg["RECORD_DATE"], errors="coerce")
    reg = reg.dropna(subset=["RECORD_DATE"])
    reg["month"] = reg["RECORD_DATE"].dt.to_period("M").dt.to_timestamp()
    reg["FUEL_TYPE"] = reg["FUEL_TYPE"].fillna("Unknown").str.strip()
    new = reg[reg["TRANSACTION_TYPE"].str.strip().eq("Registration New")]

    by_fuel = (
        new.groupby(["month", "FUEL_TYPE"]).size().rename("registrations").reset_index()
        .sort_values(["month", "registrations"], ascending=[True, False])
    )
    total = (
        new.groupby("month").size().rename("registrations").reset_index()
    )
    total["FUEL_TYPE"] = "All fuel types"
    return pd.concat([total, by_fuel], ignore_index=True).sort_values(
        ["month", "FUEL_TYPE"]
    ).reset_index(drop=True)


def main() -> None:
    print("Fetching fuel prices...")
    fuel_events = fetch_fuel()
    fuel_events.to_csv(RAW / "fuel_events.csv.gz", index=False, compression="gzip")
    fuel_monthly = monthly_fuel_prices(fuel_events)
    fuel_monthly.to_csv(CLEAN / "fuel_price_monthly.csv", index=False)
    print(f"  -> {CLEAN / 'fuel_price_monthly.csv'} ({len(fuel_monthly)} rows)")

    print("Fetching vehicle registrations...")
    reg = fetch_vehicles()
    reg_monthly = monthly_registrations(reg)
    reg_monthly.to_csv(CLEAN / "vehicle_registrations_monthly.csv", index=False)
    print(f"  -> {CLEAN / 'vehicle_registrations_monthly.csv'} ({len(reg_monthly)} rows)")

    pd.DataFrame(MANIFEST).assign(
        fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")
    ).to_csv(RAW / "_manifest.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
