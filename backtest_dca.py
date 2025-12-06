#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from typing import List, Dict

from dca_btc import StrategyConfig, Snapshot, DecisionContext, decide


def _as_float(value: str) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def load_logs(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def snapshot_from_log_row(row: Dict) -> Snapshot:
    return Snapshot(
        price=_as_float(row.get("price_spot")),
        mark_price=_as_float(row.get("price_mark", row.get("price_spot"))),
        mayer=_as_float(row.get("mayer")),
        dist200w=_as_float(row.get("dist_200w")),
        ssr=_as_float(row.get("ssr_like")),
        vol30d=_as_float(row.get("vol30d")),
        funding=_as_float(row.get("funding")),
        oi=_as_float(row.get("oi_notional")),
        trend7d=_as_float(row.get("price_trend_7d")),
        ma200d=_as_float(row.get("ma200d")),
        ma200w=_as_float(row.get("ma200w")),
    )


def ctx_from_log_row(row: Dict) -> DecisionContext:
    return DecisionContext(
        oi_trend_7d=_as_float(row.get("oi_trend_7d")),
        ma_adaptive=_as_float(row.get("ma_adaptive")),
        atr_adaptive=_as_float(row.get("atr_adaptive")),
        prev_close=_as_float(row.get("prev_close")),
        adaptive_window=int(_as_float(row.get("adaptive_window")))
        if not math.isnan(_as_float(row.get("adaptive_window")))
        else 30,
    )


def run_backtest(log_path: str, cfg: StrategyConfig) -> None:
    rows = load_logs(log_path)

    results = []
    for row in rows:
        snapshot = snapshot_from_log_row(row)
        ctx = ctx_from_log_row(row)

        orig_final_mult = _as_float(row.get("final_mult"))
        orig_invest = _as_float(row.get("invest_amount"))
        base_invest = _as_float(row.get("base_invest"))
        if math.isnan(base_invest):
            base_invest = 30.0

        decision = decide(snapshot, ctx=ctx, cfg=cfg, base_invest=base_invest)

        results.append({
            "decision_date": row.get("decision_date", ""),
            "price_spot": snapshot.price,
            "orig_final_mult": orig_final_mult,
            "new_final_mult": decision.final_mult,
            "orig_invest": orig_invest,
            "new_invest": decision.invest,
        })

    summarize_results(results)


def summarize_results(results: List[Dict]) -> None:
    if not results:
        print("No results to summarize.")
        return

    total_orig = sum(r["orig_invest"] for r in results if not math.isnan(r["orig_invest"]))
    total_new = sum(r["new_invest"] for r in results if not math.isnan(r["new_invest"]))

    avg_orig_mult = sum(r["orig_final_mult"] for r in results if not math.isnan(r["orig_final_mult"])) / max(
        1, len([r for r in results if not math.isnan(r["orig_final_mult"])])
    )
    avg_new_mult = sum(r["new_final_mult"] for r in results if not math.isnan(r["new_final_mult"])) / max(
        1, len([r for r in results if not math.isnan(r["new_final_mult"])])
    )

    diverged = [r for r in results if abs(_as_float(r["new_final_mult"]) - _as_float(r["orig_final_mult"])) > 1]

    print(f"Total original invest: {total_orig:.2f}")
    print(f"Total new      invest: {total_new:.2f}")
    print(f"Avg original final_mult: {avg_orig_mult:.3f}")
    print(f"Avg new      final_mult: {avg_new_mult:.3f}")
    print(f"Days with |Δmult| > 1: {len(diverged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/dca_runs.csv")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max_brake", type=float, default=0.6)
    parser.add_argument("--z1", type=float, default=-1.0)
    parser.add_argument("--z2", type=float, default=-1.5)
    parser.add_argument("--edge1", type=float, default=0.3)
    parser.add_argument("--edge2", type=float, default=0.6)
    parser.add_argument("--trend_brake_scale_down", type=float, default=0.5)
    args = parser.parse_args()

    cfg = StrategyConfig(
        alpha=args.alpha,
        max_brake=args.max_brake,
        z_edge_level1=args.z1,
        z_edge_level2=args.z2,
        edge_size1=args.edge1,
        edge_size2=args.edge2,
        trend_brake_scale_down=args.trend_brake_scale_down,
    )

    run_backtest(args.log, cfg)
