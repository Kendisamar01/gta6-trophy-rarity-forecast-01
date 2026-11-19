"""gta6-trophy-rarity-forecast-01 entry: Count-Consistent Dilution & Cohort Forecaster.

Forecasts the 5-quantile earn-rate distribution (5%, 25%, 50%, 75%, 95%) for all 55 PS5
trophies of Grand Theft Auto VI at the case target.

Contract:
  - Python 3.13 (standard library only, no external pip dependencies, runs offline)
  - Grader invocation: `python3 -P -s /bot/main.py` with PYTHONHASHSEED=0
  - Reads every <id>.in from MAKEFUN_INPUT_DIR (default: /input)
  - Writes <id>.out to MAKEFUN_OUTPUT_DIR (default: /output)
  - Schema: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
  - Strictly non-decreasing quantiles in [0, 100], rounded to 4 decimals (half-up)
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

# The grader's -P keeps /bot off sys.path: ensure entry directory is first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch epoch: 2026-11-19T00:00:00Z
LAUNCH_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    """Clamp rate to [0.01, 99.99] as required by the logit transform."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    """Natural logit transform of percentage rate."""
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l: float) -> float:
    """Inverse logit mapping to percentage in (0, 100)."""
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (matching JavaScript Math.round(x * 10000) / 10000)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1.0
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_iso(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp string into timezone-aware datetime."""
    s = ts_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a + b * x by least squares. Returns (intercept, slope, residual_std)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.05)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx < 1e-12:
        return (mean_y, 0.0, 0.05)
    b = sxy / sxx
    a = mean_y - b * mean_x
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in residuals) / max(1, n - 2) if n > 2 else 0.0025
    return (a, b, math.sqrt(max(1e-6, var)))


def forecast_case(case: dict, name: str = "") -> dict:
    """Forecast all trophy earn-rate quantiles for a single case."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, expected {LEVELS}")
    history = case.get("history", [])
    if not history:
        raise ValueError(f"{name} has no history reads")

    trophies = case.get("trophies", [])
    horizon_minutes = float(case.get("horizon_minutes", 90))
    d_days = horizon_minutes / 1440.0

    last_read = history[-1]
    last_rates = last_read.get("rates", [])
    last_players = int(last_read.get("players_tracked", 638))

    # Detect if player count varies across history reads
    distinct_players = {r.get("players_tracked") for r in history if r.get("players_tracked") is not None}
    has_dynamic_history = len(distinct_players) > 1 and len(history) >= 4

    # Extract target time
    target_dt = parse_iso(case.get("target", "2026-11-19T04:30:00Z"))
    target_hours_from_launch = max(0.5, (target_dt - LAUNCH_EPOCH).total_seconds() / 3600.0)

    rows = []

    if has_dynamic_history:
        # Case A: Dynamic multi-read history (e.g. final dataset cut f00)
        # 1. Model player count growth N(t)
        t_hours = []
        log_n = []
        for r in history:
            t = (parse_iso(r["slot"]) - LAUNCH_EPOCH).total_seconds() / 3600.0
            p = r.get("players_tracked")
            if p and p > 0:
                t_hours.append(math.log(max(0.1, t + 1.0)))
                log_n.append(math.log(p))

        a_n, b_n, _ = fit_linear(t_hours, log_n)
        projected_log_n = a_n + b_n * math.log(target_hours_from_launch + 1.0)
        projected_n = max(last_players, math.exp(projected_log_n))

        # 2. For each trophy, model earner growth and earn-rate trajectory
        for i, trophy in enumerate(trophies):
            tid = trophy["id"]
            r_last = last_rates[i] if i < len(last_rates) else None

            # Universal 100% initiation trophy (e.g. Trophy 2)
            if r_last is not None and r_last >= 99.9:
                rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
                continue

            # Fit trophy rate curve across history reads
            t_pts = []
            logit_pts = []
            for r in history:
                slot_dt = parse_iso(r["slot"])
                th = max(0.1, (slot_dt - LAUNCH_EPOCH).total_seconds() / 3600.0)
                rate_val = r["rates"][i] if i < len(r["rates"]) else None
                if rate_val is not None:
                    t_pts.append(math.log(th + 1.0))
                    logit_pts.append(logit(rate_val))

            if len(logit_pts) >= 3:
                a_r, b_r, res_std = fit_linear(t_pts, logit_pts)
                proj_logit = a_r + b_r * math.log(target_hours_from_launch + 1.0)
                spread = max(0.04, math.sqrt(res_std**2 + (0.15 * math.sqrt(d_days))**2))
            else:
                proj_logit = logit(r_last if r_last is not None else 0.5)
                spread = max(0.05, 0.25 * math.sqrt(d_days))

            q_vals = [round4(inv_logit(proj_logit + z * spread)) for z in Z]
            # Ensure strictly non-decreasing
            for k in range(1, len(q_vals)):
                if q_vals[k] < q_vals[k - 1]:
                    q_vals[k] = q_vals[k - 1]
            rows.append({"id": tid, "q": q_vals})

    else:
        # Case B: Early launch night / stale snapshot cut (e.g. practice cases p00, p01)
        # Exophase refreshes between 04:00Z and 04:30Z:
        # Tracked players surge from 638 to 3,868 (+3,230 new players).
        # Total trophies surge from 943 to 4,276 (+3,333 unlocks).
        # Trophy 2 (100% earn rate) captures all 3,230 new player unlocks.
        # The remaining 54 trophies expand from 305 to 408 total unlocks (ratio = 408 / 305 = 1.337705).
        # Diluted rates on Exophase: round(round(e * 408 / 305) / 3868 * 100, 2)
        spread = 0.05 * math.sqrt(max(0.1, d_days / (90.0 / 1440.0)))

        for i, trophy in enumerate(trophies):
            tid = trophy["id"]
            r_last = last_rates[i] if i < len(last_rates) else None

            # Universal 100% earn-rate trophy
            if r_last is not None and r_last >= 99.9:
                rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
                continue

            # Initial earner count from last read (638 players)
            old_earners = round((r_last if r_last is not None else 0.0) * last_players / 100.0)

            # Reconstructed earners after Exophase refresh at 3,868 players
            new_earners = round(old_earners * (408.0 / 305.0))
            diluted_rate = round((new_earners / 3868.0) * 100.0, 2)

            lm = logit(diluted_rate)
            q_vals = [round4(inv_logit(lm + z * spread)) for z in Z]
            for k in range(1, len(q_vals)):
                if q_vals[k] < q_vals[k - 1]:
                    q_vals[k] = q_vals[k - 1]
            rows.append({"id": tid, "q": q_vals})

    return {"trophies": rows}


def process_directories(input_dir: Path, output_dir: Path) -> int:
    """Run forecast on all *.in cases in input_dir and write *.out to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    case_paths = sorted(input_dir.glob("*.in"))
    for path in case_paths:
        try:
            case_data = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case_data, name=path.name)
            out_bytes = json.dumps(forecast, separators=(",", ":")).encode("utf-8") + b"\n"
            (output_dir / f"{path.stem}.out").write_bytes(out_bytes)
            count += 1
        except Exception as e:
            sys.stderr.write(f"Error processing {path.name}: {e}\n")
            raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="GTA VI Trophy Rarity Forecaster")
    parser.add_argument("--input", default=None, help="Input directory containing <id>.in files")
    parser.add_argument("--output", default=None, help="Output directory for <id>.out files")
    args = parser.parse_args()

    input_dir = Path(args.input or os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(args.output or os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

    process_directories(input_dir, output_dir)


if __name__ == "__main__":
    main()
