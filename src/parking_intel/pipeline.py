from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def build_prototype(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_path = resolve_path(config["raw_csv"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_raw_csv(raw_path)
    clean, quality = clean_violations(raw, config)
    hotspots = build_hotspots(clean, config)
    station_summary = build_station_summary(clean, hotspots)
    violation_summary = build_violation_summary(clean)
    hourly_summary = build_hourly_summary(clean)
    monthly_summary = build_monthly_summary(clean)
    priority_summary = build_priority_summary(hotspots)
    cuda_status = capture_cuda_status()

    clean_path = PROCESSED_DIR / "cleaned_violations.csv"
    hotspot_path = PROCESSED_DIR / "hotspots.csv"
    priority_path = PROCESSED_DIR / "enforcement_priority.csv"
    station_path = PROCESSED_DIR / "station_summary.csv"
    violation_path = PROCESSED_DIR / "violation_summary.csv"
    hourly_path = PROCESSED_DIR / "hourly_patterns.csv"
    monthly_path = PROCESSED_DIR / "monthly_patterns.csv"
    payload_path = PROCESSED_DIR / "dashboard_payload.json"
    cuda_path = PROCESSED_DIR / "cuda_status.json"

    clean_export_columns = [
        "id",
        "latitude",
        "longitude",
        "zone_id",
        "location_clean",
        "police_station_clean",
        "junction_name_clean",
        "vehicle_type_clean",
        "primary_violation",
        "violation_labels",
        "created_at_utc",
        "created_at_local",
        "created_date",
        "month",
        "day_of_week",
        "hour",
        "is_weekend",
        "is_peak_hour",
        "validation_status_clean",
        "is_approved",
        "data_sent_flag",
        "response_minutes",
    ]
    clean[clean_export_columns].to_csv(clean_path, index=False)
    hotspots.to_csv(hotspot_path, index=False)
    hotspots.head(100).to_csv(priority_path, index=False)
    station_summary.to_csv(station_path, index=False)
    violation_summary.to_csv(violation_path, index=False)
    hourly_summary.to_csv(hourly_path, index=False)
    monthly_summary.to_csv(monthly_path, index=False)
    write_json(cuda_path, cuda_status)

    payload = build_dashboard_payload(
        config=config,
        quality=quality,
        cuda_status=cuda_status,
        hotspots=hotspots,
        station_summary=station_summary,
        violation_summary=violation_summary,
        hourly_summary=hourly_summary,
        monthly_summary=monthly_summary,
        priority_summary=priority_summary,
    )
    write_json(payload_path, payload)

    write_quality_report(quality, config, cuda_status, station_summary, violation_summary)
    write_summary_report(quality, config, cuda_status, hotspots, station_summary, violation_summary)
    write_scoring_report(config)

    return {
        "raw_rows": quality["raw_rows"],
        "clean_rows": quality["clean_rows"],
        "hotspots": int(len(hotspots)),
        "top_priority_zone": hotspots.iloc[0]["zone_id"] if len(hotspots) else None,
        "cuda_available": cuda_status["cuda_available"],
        "dashboard_payload": str(payload_path.relative_to(ROOT)),
    }


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    return pd.read_csv(
        path,
        na_values=["NULL", "null", "None", ""],
        keep_default_na=True,
        low_memory=False,
    )


def clean_violations(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw.copy()
    raw_rows = int(len(df))

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    created_at_utc = pd.to_datetime(df.get("created_datetime"), errors="coerce", utc=True)
    df["created_at_utc"] = created_at_utc

    bounds = config["coordinate_bounds"]
    valid_geo = (
        df["latitude"].between(bounds["min_latitude"], bounds["max_latitude"])
        & df["longitude"].between(bounds["min_longitude"], bounds["max_longitude"])
    )
    valid_time = df["created_at_utc"].notna()
    clean = df.loc[valid_geo & valid_time].copy()

    clean["created_at_local_dt"] = clean["created_at_utc"].dt.tz_convert(config["timezone"])
    clean["created_at_local"] = clean["created_at_local_dt"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    clean["created_date"] = clean["created_at_local_dt"].dt.strftime("%Y-%m-%d")
    clean["month"] = clean["created_at_local_dt"].dt.strftime("%Y-%m")
    clean["day_of_week"] = clean["created_at_local_dt"].dt.day_name()
    clean["hour"] = clean["created_at_local_dt"].dt.hour.astype("Int64")
    clean["is_weekend"] = clean["created_at_local_dt"].dt.dayofweek.isin([5, 6])
    clean["is_peak_hour"] = clean["hour"].isin(config["peak_hours"])

    clean["location_clean"] = text_column(clean, "location", "Unknown location")
    clean["police_station_clean"] = text_column(clean, "police_station", "Unknown station")
    clean["vehicle_type_clean"] = text_column(clean, "updated_vehicle_type", "")
    fallback_vehicle_type = text_column(clean, "vehicle_type", "Unknown vehicle")
    clean.loc[clean["vehicle_type_clean"].eq(""), "vehicle_type_clean"] = fallback_vehicle_type
    clean["junction_name_clean"] = text_column(clean, "junction_name", "No Junction")
    no_junction = clean["junction_name_clean"].str.casefold().isin({"no junction", "unknown", "nan", ""})
    clean.loc[no_junction, "junction_name_clean"] = "No Junction"
    clean["has_named_junction"] = ~clean["junction_name_clean"].eq("No Junction")

    violation_lists = clean["violation_type"].map(parse_list_cell)
    clean["primary_violation"] = violation_lists.map(lambda labels: labels[0] if labels else "UNKNOWN")
    clean["violation_labels"] = violation_lists.map(lambda labels: "; ".join(labels) if labels else "UNKNOWN")
    clean["has_road_obstruction_signal"] = clean["violation_labels"].str.contains(
        "MAIN ROAD|ROAD CROSSING|NO PARKING|WRONG PARKING|FOOTPATH",
        case=False,
        regex=True,
        na=False,
    )

    clean["validation_status_clean"] = text_column(clean, "validation_status", "unknown").str.lower()
    clean["is_approved"] = clean["validation_status_clean"].eq("approved")
    clean["data_sent_flag"] = clean.get("data_sent_to_scita", pd.Series(False, index=clean.index)).map(to_bool)

    closed_at = pd.to_datetime(clean.get("closed_datetime"), errors="coerce", utc=True)
    action_at = pd.to_datetime(clean.get("action_taken_timestamp"), errors="coerce", utc=True)
    end_at = closed_at.combine_first(action_at)
    response_minutes = (end_at - clean["created_at_utc"]).dt.total_seconds() / 60
    clean["response_minutes"] = response_minutes.where(response_minutes.ge(0))

    grid_size = float(config["grid_size_degrees"])
    clean["lat_bin"] = np.rint(clean["latitude"] / grid_size).astype("int64")
    clean["lon_bin"] = np.rint(clean["longitude"] / grid_size).astype("int64")
    clean["zone_latitude"] = clean["lat_bin"] * grid_size
    clean["zone_longitude"] = clean["lon_bin"] * grid_size
    clean["zone_id"] = clean["lat_bin"].astype(str) + "_" + clean["lon_bin"].astype(str)

    missing_by_column = {
        col: {
            "missing": int(raw[col].isna().sum()),
            "missing_pct": round(float(raw[col].isna().mean() * 100), 2),
        }
        for col in raw.columns
    }

    quality = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_rows": raw_rows,
        "clean_rows": int(len(clean)),
        "dropped_rows": int(raw_rows - len(clean)),
        "invalid_coordinate_rows": int((~valid_geo).sum()),
        "invalid_timestamp_rows": int((~valid_time).sum()),
        "date_min": iso_or_none(clean["created_at_utc"].min()),
        "date_max": iso_or_none(clean["created_at_utc"].max()),
        "unique_police_stations": int(clean["police_station_clean"].nunique()),
        "unique_vehicle_numbers": int(clean["vehicle_number"].nunique(dropna=True)) if "vehicle_number" in clean else 0,
        "unique_zones": int(clean["zone_id"].nunique()),
        "approval_rate": round(float(clean["is_approved"].mean() * 100), 2) if len(clean) else 0.0,
        "peak_hour_share": round(float(clean["is_peak_hour"].mean() * 100), 2) if len(clean) else 0.0,
        "named_junction_share": round(float(clean["has_named_junction"].mean() * 100), 2) if len(clean) else 0.0,
        "missing_by_column": missing_by_column,
    }
    return clean, quality


def build_hotspots(clean: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()

    grouped = clean.groupby("zone_id", dropna=False)
    hotspots = grouped.agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        zone_latitude=("zone_latitude", "first"),
        zone_longitude=("zone_longitude", "first"),
        violation_count=("id", "count"),
        unique_vehicle_count=("vehicle_number", "nunique"),
        active_days=("created_date", "nunique"),
        first_seen=("created_at_utc", "min"),
        last_seen=("created_at_utc", "max"),
        peak_share=("is_peak_hour", "mean"),
        weekend_share=("is_weekend", "mean"),
        approval_rate=("is_approved", "mean"),
        sent_to_scita_share=("data_sent_flag", "mean"),
        named_junction_share=("has_named_junction", "mean"),
        obstruction_signal_share=("has_road_obstruction_signal", "mean"),
        violation_diversity=("primary_violation", "nunique"),
        median_response_minutes=("response_minutes", "median"),
        police_station=("police_station_clean", mode_value),
        junction_name=("junction_name_clean", mode_value),
        location=("location_clean", mode_value),
        primary_violation=("primary_violation", mode_value),
        dominant_vehicle_type=("vehicle_type_clean", mode_value),
    ).reset_index()

    hotspots["vehicle_recurrence_share"] = (
        1 - (hotspots["unique_vehicle_count"] / hotspots["violation_count"].clip(lower=1))
    ).clip(lower=0, upper=1)
    hotspots["density_component"] = normalize_series(np.log1p(hotspots["violation_count"]))
    hotspots["persistence_component"] = normalize_series(hotspots["active_days"])
    hotspots["diversity_component"] = normalize_series(hotspots["violation_diversity"])
    hotspots["validation_component"] = hotspots["approval_rate"].fillna(0).clip(0, 1)
    hotspots["peak_component"] = hotspots["peak_share"].fillna(0).clip(0, 1)
    hotspots["junction_component"] = hotspots["named_junction_share"].fillna(0).clip(0, 1)
    hotspots["obstruction_component"] = hotspots["obstruction_signal_share"].fillna(0).clip(0, 1)
    hotspots["recurrence_component"] = hotspots["vehicle_recurrence_share"].fillna(0).clip(0, 1)

    hotspots["severity_score"] = 100 * (
        0.35 * hotspots["density_component"]
        + 0.20 * hotspots["persistence_component"]
        + 0.15 * hotspots["validation_component"]
        + 0.15 * hotspots["recurrence_component"]
        + 0.15 * hotspots["diversity_component"]
    )
    hotspots["congestion_impact_score"] = 100 * (
        0.30 * hotspots["density_component"]
        + 0.25 * hotspots["peak_component"]
        + 0.20 * hotspots["junction_component"]
        + 0.15 * hotspots["obstruction_component"]
        + 0.10 * hotspots["persistence_component"]
    )
    hotspots["priority_score"] = 0.55 * hotspots["severity_score"] + 0.45 * hotspots["congestion_impact_score"]
    hotspots["priority_class"] = hotspots["priority_score"].map(
        lambda score: priority_class(score, config["priority_thresholds"])
    )
    hotspots["recommended_action"] = hotspots["priority_class"].map(priority_action)

    score_cols = [
        "peak_share",
        "weekend_share",
        "approval_rate",
        "sent_to_scita_share",
        "named_junction_share",
        "obstruction_signal_share",
        "vehicle_recurrence_share",
        "severity_score",
        "congestion_impact_score",
        "priority_score",
    ]
    for col in score_cols:
        hotspots[col] = hotspots[col].astype(float).round(3)

    hotspots["first_seen"] = hotspots["first_seen"].map(iso_or_none)
    hotspots["last_seen"] = hotspots["last_seen"].map(iso_or_none)
    hotspots["median_response_minutes"] = hotspots["median_response_minutes"].round(1)

    order = [
        "zone_id",
        "latitude",
        "longitude",
        "zone_latitude",
        "zone_longitude",
        "priority_score",
        "priority_class",
        "recommended_action",
        "severity_score",
        "congestion_impact_score",
        "violation_count",
        "active_days",
        "unique_vehicle_count",
        "vehicle_recurrence_share",
        "peak_share",
        "weekend_share",
        "approval_rate",
        "sent_to_scita_share",
        "named_junction_share",
        "obstruction_signal_share",
        "primary_violation",
        "dominant_vehicle_type",
        "police_station",
        "junction_name",
        "location",
        "first_seen",
        "last_seen",
        "median_response_minutes",
    ]
    return hotspots[order].sort_values(
        ["priority_score", "violation_count", "congestion_impact_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_station_summary(clean: pd.DataFrame, hotspots: pd.DataFrame) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    station = clean.groupby("police_station_clean").agg(
        violation_count=("id", "count"),
        unique_zones=("zone_id", "nunique"),
        approved_count=("is_approved", "sum"),
        peak_hour_count=("is_peak_hour", "sum"),
        named_junction_count=("has_named_junction", "sum"),
    ).reset_index(names="police_station")
    station["approval_rate"] = (station["approved_count"] / station["violation_count"]).round(3)
    station["peak_share"] = (station["peak_hour_count"] / station["violation_count"]).round(3)
    station["named_junction_share"] = (station["named_junction_count"] / station["violation_count"]).round(3)

    hot_by_station = hotspots.groupby("police_station").agg(
        max_priority_score=("priority_score", "max"),
        immediate_or_high_zones=("priority_class", lambda s: int(s.isin(["Immediate patrol", "High watchlist"]).sum())),
    ).reset_index()
    station = station.merge(hot_by_station, on="police_station", how="left")
    station[["max_priority_score", "immediate_or_high_zones"]] = station[
        ["max_priority_score", "immediate_or_high_zones"]
    ].fillna(0)
    return station.sort_values(["max_priority_score", "violation_count"], ascending=False).reset_index(drop=True)


def build_violation_summary(clean: pd.DataFrame) -> pd.DataFrame:
    labels = clean[["id", "violation_labels", "is_peak_hour", "is_approved"]].copy()
    labels["violation"] = labels["violation_labels"].str.split("; ")
    labels = labels.explode("violation")
    summary = labels.groupby("violation").agg(
        violation_count=("id", "count"),
        peak_hour_count=("is_peak_hour", "sum"),
        approved_count=("is_approved", "sum"),
    ).reset_index()
    summary["peak_share"] = (summary["peak_hour_count"] / summary["violation_count"]).round(3)
    summary["approval_rate"] = (summary["approved_count"] / summary["violation_count"]).round(3)
    return summary.sort_values("violation_count", ascending=False).reset_index(drop=True)


def build_hourly_summary(clean: pd.DataFrame) -> pd.DataFrame:
    hours = clean.groupby("hour").agg(
        violation_count=("id", "count"),
        approved_count=("is_approved", "sum"),
        unique_zones=("zone_id", "nunique"),
    ).reset_index()
    hours["approval_rate"] = (hours["approved_count"] / hours["violation_count"]).round(3)
    return hours.sort_values("hour").reset_index(drop=True)


def build_monthly_summary(clean: pd.DataFrame) -> pd.DataFrame:
    monthly = clean.groupby("month").agg(
        violation_count=("id", "count"),
        unique_zones=("zone_id", "nunique"),
        peak_hour_count=("is_peak_hour", "sum"),
        approved_count=("is_approved", "sum"),
    ).reset_index()
    monthly["peak_share"] = (monthly["peak_hour_count"] / monthly["violation_count"]).round(3)
    monthly["approval_rate"] = (monthly["approved_count"] / monthly["violation_count"]).round(3)
    return monthly.sort_values("month").reset_index(drop=True)


def build_priority_summary(hotspots: pd.DataFrame) -> pd.DataFrame:
    order = ["Immediate patrol", "High watchlist", "Scheduled patrol", "Monitor"]
    summary = hotspots.groupby("priority_class").agg(
        zone_count=("zone_id", "count"),
        violation_count=("violation_count", "sum"),
        average_priority=("priority_score", "mean"),
    ).reindex(order).fillna(0).reset_index()
    summary["average_priority"] = summary["average_priority"].round(1)
    return summary


def build_dashboard_payload(
    *,
    config: dict[str, Any],
    quality: dict[str, Any],
    cuda_status: dict[str, Any],
    hotspots: pd.DataFrame,
    station_summary: pd.DataFrame,
    violation_summary: pd.DataFrame,
    hourly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    priority_summary: pd.DataFrame,
) -> dict[str, Any]:
    limit = int(config["dashboard_hotspot_limit"])
    dashboard_hotspots = hotspots.head(limit).copy()
    center = {
        "latitude": round(float(dashboard_hotspots["latitude"].mean()), 6) if len(dashboard_hotspots) else 12.9716,
        "longitude": round(float(dashboard_hotspots["longitude"].mean()), 6) if len(dashboard_hotspots) else 77.5946,
    }
    metrics = {
        "violations": quality["clean_rows"],
        "hotspots": int(len(hotspots)),
        "immediate_zones": int((hotspots["priority_class"] == "Immediate patrol").sum()) if len(hotspots) else 0,
        "high_watchlist_zones": int((hotspots["priority_class"] == "High watchlist").sum()) if len(hotspots) else 0,
        "approval_rate": quality["approval_rate"],
        "peak_hour_share": quality["peak_hour_share"],
    }
    return {
        "metadata": {
            "generated_at": quality["generated_at"],
            "city": config["city"],
            "timezone": config["timezone"],
            "date_min": quality["date_min"],
            "date_max": quality["date_max"],
            "grid_size_degrees": config["grid_size_degrees"],
            "cuda": cuda_status,
        },
        "metrics": metrics,
        "map_center": center,
        "hotspots": records(dashboard_hotspots),
        "station_summary": records(station_summary.head(20)),
        "violation_summary": records(violation_summary.head(20)),
        "hourly_summary": records(hourly_summary),
        "monthly_summary": records(monthly_summary),
        "priority_summary": records(priority_summary),
    }


def capture_cuda_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {
            "torch_imported": False,
            "cuda_available": False,
            "error": repr(error),
        }

    status: dict[str, Any] = {
        "torch_imported": True,
        "torch_version": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": None,
        "test_tensor_device": None,
    }
    if status["cuda_available"]:
        status["device_name"] = torch.cuda.get_device_name(0)
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        status["test_tensor_device"] = str(tensor.device)
    return status


def write_quality_report(
    quality: dict[str, Any],
    config: dict[str, Any],
    cuda_status: dict[str, Any],
    station_summary: pd.DataFrame,
    violation_summary: pd.DataFrame,
) -> None:
    missing_rows = [
        [col, stats["missing"], f'{stats["missing_pct"]}%']
        for col, stats in sorted(
            quality["missing_by_column"].items(),
            key=lambda item: item[1]["missing_pct"],
            reverse=True,
        )
    ]
    report = [
        "# Data Quality Report",
        "",
        f"Generated: {quality['generated_at']}",
        f"City/timezone: {config['city']} / {config['timezone']}",
        "",
        "## Dataset Health",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Raw rows", quality["raw_rows"]],
                ["Clean analysis rows", quality["clean_rows"]],
                ["Dropped rows", quality["dropped_rows"]],
                ["Invalid coordinate rows", quality["invalid_coordinate_rows"]],
                ["Invalid timestamp rows", quality["invalid_timestamp_rows"]],
                ["Date range", f"{quality['date_min']} to {quality['date_max']}"],
                ["Unique police stations", quality["unique_police_stations"]],
                ["Unique grid zones", quality["unique_zones"]],
                ["Approval rate", f"{quality['approval_rate']}%"],
                ["Peak-hour share", f"{quality['peak_hour_share']}%"],
                ["CUDA available", cuda_status["cuda_available"]],
                ["CUDA device", cuda_status.get("device_name") or "None"],
            ],
        ),
        "",
        "## Top Police Stations",
        "",
        markdown_table(
            ["Station", "Violations", "Zones", "Peak share", "Max priority"],
            [
                [
                    row["police_station"],
                    int(row["violation_count"]),
                    int(row["unique_zones"]),
                    f'{float(row["peak_share"]) * 100:.1f}%',
                    f'{float(row["max_priority_score"]):.1f}',
                ]
                for _, row in station_summary.head(10).iterrows()
            ],
        ),
        "",
        "## Top Violation Labels",
        "",
        markdown_table(
            ["Violation", "Count", "Peak share", "Approval rate"],
            [
                [
                    row["violation"],
                    int(row["violation_count"]),
                    f'{float(row["peak_share"]) * 100:.1f}%',
                    f'{float(row["approval_rate"]) * 100:.1f}%',
                ]
                for _, row in violation_summary.head(10).iterrows()
            ],
        ),
        "",
        "## Missing Values",
        "",
        markdown_table(["Column", "Missing rows", "Missing pct"], missing_rows),
        "",
    ]
    (REPORTS_DIR / "data_quality_report.md").write_text("\n".join(report), encoding="utf-8")


def write_summary_report(
    quality: dict[str, Any],
    config: dict[str, Any],
    cuda_status: dict[str, Any],
    hotspots: pd.DataFrame,
    station_summary: pd.DataFrame,
    violation_summary: pd.DataFrame,
) -> None:
    top_hotspot_rows = [
        [
            row["zone_id"],
            f'{float(row["priority_score"]):.1f}',
            row["priority_class"],
            int(row["violation_count"]),
            row["police_station"],
            row["primary_violation"],
        ]
        for _, row in hotspots.head(10).iterrows()
    ]
    report = [
        "# Parking Intelligence Prototype Summary",
        "",
        "## Prototype Status",
        "",
        markdown_table(
            ["Component", "Status"],
            [
                ["CUDA environment", "Ready" if cuda_status["cuda_available"] else "Not available"],
                ["GPU", cuda_status.get("device_name") or "None"],
                ["Cleaned dataset", "Built"],
                ["Hotspot ranking", "Built"],
                ["Congestion proxy", "Built"],
                ["Enforcement priority queue", "Built"],
                ["Map dashboard payload", "Built"],
            ],
        ),
        "",
        "## Operating Window",
        "",
        f"The cleaned dataset contains {quality['clean_rows']:,} valid records from {quality['date_min']} to {quality['date_max']}. "
        f"Hotspots are grouped into approximately {config['grid_size_degrees']} degree cells, which is close to street-block scale for a city prototype.",
        "",
        "## Top Priority Zones",
        "",
        markdown_table(
            ["Zone", "Priority", "Class", "Violations", "Station", "Dominant violation"],
            top_hotspot_rows,
        ),
        "",
        "## Strongest Station Signals",
        "",
        markdown_table(
            ["Station", "Violations", "High zones", "Max priority"],
            [
                [
                    row["police_station"],
                    int(row["violation_count"]),
                    int(row["immediate_or_high_zones"]),
                    f'{float(row["max_priority_score"]):.1f}',
                ]
                for _, row in station_summary.head(8).iterrows()
            ],
        ),
        "",
        "## Dominant Violations",
        "",
        markdown_table(
            ["Violation", "Count"],
            [[row["violation"], int(row["violation_count"])] for _, row in violation_summary.head(8).iterrows()],
        ),
        "",
        "## Next Build Steps",
        "",
        "1. Add road-network context from OpenStreetMap or city GIS layers.",
        "2. Validate top zones with local enforcement knowledge.",
        "3. Add holdout-window evaluation for priority stability.",
        "4. Train a supervised GPU ranking model once labels or validated proxy outcomes exist.",
        "",
    ]
    (REPORTS_DIR / "prototype_summary.md").write_text("\n".join(report), encoding="utf-8")


def write_scoring_report(config: dict[str, Any]) -> None:
    thresholds = config["priority_thresholds"]
    report = [
        "# Scoring Formula",
        "",
        "Hotspots are grid cells produced from latitude and longitude. Each cell is scored with normalized components so high-volume areas and smaller repeated-risk zones can both surface.",
        "",
        "## Severity Score",
        "",
        "Severity = 100 * (0.35 density + 0.20 persistence + 0.15 validation + 0.15 recurrence + 0.15 violation diversity)",
        "",
        "## Congestion Impact Proxy",
        "",
        "Impact = 100 * (0.30 density + 0.25 peak-hour pressure + 0.20 junction pressure + 0.15 obstruction signal + 0.10 persistence)",
        "",
        "## Enforcement Priority",
        "",
        "Priority = 0.55 severity + 0.45 congestion impact",
        "",
        markdown_table(
            ["Class", "Rule"],
            [
                ["Immediate patrol", f">= {thresholds['immediate_patrol']}"],
                ["High watchlist", f">= {thresholds['high_watchlist']}"],
                ["Scheduled patrol", f">= {thresholds['scheduled_patrol']}"],
                ["Monitor", f"< {thresholds['scheduled_patrol']}"],
            ],
        ),
        "",
        "The proxy does not claim measured travel delay. It estimates disruption risk from repeat illegal parking, time-of-day pressure, junction proximity, and violation type.",
        "",
    ]
    (REPORTS_DIR / "scoring_formula.md").write_text("\n".join(report), encoding="utf-8")


def parse_list_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
    if isinstance(parsed, list):
        return [normalize_label(item) for item in parsed if normalize_label(item)]
    return [normalize_label(parsed)] if normalize_label(parsed) else []


def normalize_label(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def text_column(df: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="string")
    series = df[column].fillna(default).astype(str).str.strip()
    series = series.mask(series.str.casefold().isin({"nan", "none", "null"}), default)
    return series


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def mode_value(series: pd.Series) -> str:
    values = [
        str(value).strip()
        for value in series.dropna().tolist()
        if str(value).strip() and str(value).strip().casefold() not in {"nan", "none", "null"}
    ]
    if not values:
        return "Unknown"
    return Counter(values).most_common(1)[0][0]


def normalize_series(series: pd.Series | np.ndarray) -> pd.Series:
    result = pd.Series(series, dtype="float64")
    min_value = result.min()
    max_value = result.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(result)), index=result.index)
    return ((result - min_value) / (max_value - min_value)).clip(0, 1)


def priority_class(score: float, thresholds: dict[str, int]) -> str:
    if score >= thresholds["immediate_patrol"]:
        return "Immediate patrol"
    if score >= thresholds["high_watchlist"]:
        return "High watchlist"
    if score >= thresholds["scheduled_patrol"]:
        return "Scheduled patrol"
    return "Monitor"


def priority_action(priority: str) -> str:
    return {
        "Immediate patrol": "Dispatch patrol during next peak window",
        "High watchlist": "Schedule targeted enforcement",
        "Scheduled patrol": "Add to rotating patrol plan",
        "Monitor": "Monitor trend and revisit weekly",
    }[priority]


def iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["-" for _ in headers]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def escape_markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
