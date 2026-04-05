"""Generate a synthetic weekly raw sales dataset for the City-level forecasting pipeline.

The script creates a semicolon-separated CSV with columns:
- `Week`   : ISO week start date (Monday)
- `City`   : city name
- `revenue`: weekly sales

Default behavior:
- builds history from the start of the year four years before the anchor date
- ends on the last fully closed ISO week before the anchor date
- generates around 50 cities with large/medium/small revenue scales
- makes a small share of cities permanently leave the network so rows disappear
  after the exit week
- keeps the active portion of each city on a regular weekly grid suitable for Darts
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

CITY_NAMES = [
    "Moscow",
    "SaintPetersburg",
    "Novosibirsk",
    "Yekaterinburg",
    "Kazan",
    "NizhnyNovgorod",
    "Chelyabinsk",
    "Samara",
    "Omsk",
    "RostovOnDon",
    "Ufa",
    "Krasnoyarsk",
    "Voronezh",
    "Perm",
    "Volgograd",
    "Krasnodar",
    "Saratov",
    "Tyumen",
    "Tolyatti",
    "Izhevsk",
    "Barnaul",
    "Ulyanovsk",
    "Irkutsk",
    "Khabarovsk",
    "Yaroslavl",
    "Vladivostok",
    "Makhachkala",
    "Tomsk",
    "Orenburg",
    "Kemerovo",
    "Novokuznetsk",
    "Ryazan",
    "Astrakhan",
    "Penza",
    "Lipetsk",
    "Kirov",
    "Cheboksary",
    "Tula",
    "Kaliningrad",
    "Kursk",
    "UlanUde",
    "Sochi",
    "Tver",
    "Ivanovo",
    "Belgorod",
    "Smolensk",
    "Arkhangelsk",
    "Chita",
    "Yakutsk",
    "Murmansk",
]


@dataclass(frozen=True)
class CityProfile:
    city: str
    tier: str
    base_level: float
    trend_per_week: float
    seasonal_strength: float
    promo_probability: float
    noise_sigma: float
    exit_after_week_index: int | None


def _parse_date(raw: str | None) -> date:
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _last_closed_iso_week_start(anchor_date: date) -> date:
    current_week_start = anchor_date - timedelta(days=anchor_date.weekday())
    return current_week_start - timedelta(weeks=1)


def _first_iso_week_start(year: int) -> date:
    return date.fromisocalendar(year, 1, 1)


def _week_starts(start_week: date, end_week: date) -> list[date]:
    weeks: list[date] = []
    current = start_week
    while current <= end_week:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks


def _seasonal_factor(week: date, strength: float) -> float:
    iso_week = week.isocalendar().week
    annual_wave = 1.0 + strength * math.sin((2.0 * math.pi * iso_week) / 52.0)

    month = week.month
    month_factor = {
        1: 0.94,
        2: 0.96,
        3: 1.00,
        4: 1.02,
        5: 1.05,
        6: 1.03,
        7: 0.99,
        8: 0.98,
        9: 1.01,
        10: 1.06,
        11: 1.10,
        12: 1.20,
    }[month]

    quarter_boost = 1.04 if week.month in (3, 6, 9, 12) else 1.0
    return annual_wave * month_factor * quarter_boost


def _build_city_profiles(seed: int, city_count: int, exit_share: float) -> list[CityProfile]:
    if city_count > len(CITY_NAMES):
        raise ValueError(f"Requested {city_count} cities, but only {len(CITY_NAMES)} names are available.")

    rng = random.Random(seed)
    names = CITY_NAMES[:city_count]

    large_count = max(5, round(city_count * 0.16))
    medium_count = max(15, round(city_count * 0.34))
    if large_count + medium_count >= city_count:
        medium_count = max(1, city_count - large_count - 1)
    small_count = city_count - large_count - medium_count

    tiers = (
        ["large"] * large_count
        + ["medium"] * medium_count
        + ["small"] * small_count
    )
    rng.shuffle(tiers)

    exit_count = min(city_count - 1, max(1, round(city_count * exit_share)))
    exit_cities = set(rng.sample(names, exit_count))

    profiles: list[CityProfile] = []
    for index, city in enumerate(names):
        local_rng = random.Random(seed + index * 1777)
        tier = tiers[index]

        if tier == "large":
            base_level = local_rng.uniform(3_000_000, 12_000_000)
            trend_per_week = local_rng.uniform(-8_000, 24_000)
            seasonal_strength = local_rng.uniform(0.06, 0.14)
            promo_probability = local_rng.uniform(0.04, 0.08)
            noise_sigma = local_rng.uniform(0.05, 0.09)
        elif tier == "medium":
            base_level = local_rng.uniform(300_000, 1_500_000)
            trend_per_week = local_rng.uniform(-3_000, 9_000)
            seasonal_strength = local_rng.uniform(0.07, 0.16)
            promo_probability = local_rng.uniform(0.03, 0.07)
            noise_sigma = local_rng.uniform(0.06, 0.10)
        else:
            base_level = local_rng.uniform(30_000, 180_000)
            trend_per_week = local_rng.uniform(-700, 2_500)
            seasonal_strength = local_rng.uniform(0.08, 0.18)
            promo_probability = local_rng.uniform(0.02, 0.06)
            noise_sigma = local_rng.uniform(0.07, 0.12)

        exit_after_week_index: int | None = None
        if city in exit_cities:
            exit_after_week_index = local_rng.randint(90, 175)

        profiles.append(
            CityProfile(
                city=city,
                tier=tier,
                base_level=base_level,
                trend_per_week=trend_per_week,
                seasonal_strength=seasonal_strength,
                promo_probability=promo_probability,
                noise_sigma=noise_sigma,
                exit_after_week_index=exit_after_week_index,
            )
        )

    return profiles


def generate_rows(
    start_week: date,
    end_week: date,
    seed: int,
    city_count: int = 50,
    exit_share: float = 0.14,
    negative_probability: float = 0.01,
) -> list[dict[str, object]]:
    weeks = _week_starts(start_week, end_week)
    profiles = _build_city_profiles(seed=seed, city_count=city_count, exit_share=exit_share)
    rows: list[dict[str, object]] = []

    for index, profile in enumerate(profiles):
        rng = random.Random(seed + index * 4099)

        for week_index, week_start in enumerate(weeks):
            if profile.exit_after_week_index is not None and week_index > profile.exit_after_week_index:
                break

            trend_multiplier = 1.0 + (profile.trend_per_week * week_index) / max(profile.base_level, 1.0)
            seasonality = _seasonal_factor(week_start, profile.seasonal_strength)
            noise_multiplier = max(0.5, rng.gauss(1.0, profile.noise_sigma))

            revenue = profile.base_level * trend_multiplier * seasonality * noise_multiplier

            if rng.random() < profile.promo_probability:
                revenue *= rng.uniform(1.10, 1.45)

            if rng.random() < 0.02:
                revenue *= rng.uniform(0.72, 0.90)

            if rng.random() < negative_probability:
                revenue = -abs(revenue) * rng.uniform(0.06, 0.22)

            rows.append(
                {
                    "Week": week_start.isoformat(),
                    "City": profile.city,
                    "revenue": round(revenue, 2),
                }
            )

    rows.sort(key=lambda row: (str(row["City"]), str(row["Week"])))
    return rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Week", "City", "revenue"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic weekly sales data for the City forecasting pipeline.")
    parser.add_argument(
        "--output",
        default="data/raw_sales.csv",
        help="Path to the output CSV. Default: data/raw_sales.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible generation. Default: 42.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Anchor date in YYYY-MM-DD. The script uses the last fully closed ISO week before this date. Default: today().",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=4,
        help="How many calendar years back to include. Default: 4.",
    )
    parser.add_argument(
        "--city-count",
        type=int,
        default=50,
        help="How many cities to generate. Default: 50.",
    )
    parser.add_argument(
        "--exit-share",
        type=float,
        default=0.14,
        help="Share of cities that disappear permanently before the end of history. Default: 0.14.",
    )
    return parser


def run() -> None:
    args = _build_parser().parse_args()
    anchor_date = _parse_date(args.end_date)
    end_week = _last_closed_iso_week_start(anchor_date)
    start_year = end_week.year - args.years
    start_week = _first_iso_week_start(start_year)

    if start_week > end_week:
        raise ValueError("Invalid date range for weekly data generation.")

    rows = generate_rows(
        start_week=start_week,
        end_week=end_week,
        seed=args.seed,
        city_count=args.city_count,
        exit_share=args.exit_share,
    )
    write_rows(args.output, rows)

    city_names = sorted({row["City"] for row in rows})
    total_weeks = len(_week_starts(start_week, end_week))
    active_weeks_per_city: dict[str, int] = {city: 0 for city in city_names}
    for row in rows:
        active_weeks_per_city[str(row["City"])] += 1
    exited_cities = sum(1 for weeks in active_weeks_per_city.values() if weeks < total_weeks)

    print(f"Generated {len(rows)} rows")
    print(f"Week range: {start_week.isoformat()} .. {end_week.isoformat()}")
    print(f"Cities: {len(city_names)}")
    print(f"Cities with network exit: {exited_cities}")
    print(f"Output: {Path(args.output)}")


if __name__ == "__main__":
    run()
