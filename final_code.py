import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# ---------------------------
# Config / Default Parameters
# ---------------------------

DEFAULT_LOT_SIZE = 3000
DEFAULT_INITIAL_CAPITAL = 1000000.0
DEFAULT_STOP_LOSS_PCT = 0.30
DEFAULT_MIN_HOLD_SNAPS = 5
DEFAULT_STRIKE_STEP = 50
DEFAULT_COOLDOWN = 5
TOP_N_STRATEGIES = 5   # number of top strategies to generate graphs for

RUN_OPTIMIZATION = True  # set to True to run parameter grid search


# ---------------------------
# Trade Dataclass
# ---------------------------

@dataclass
class Trade:
    direction: str
    expiry: str
    strike: float
    entry_datetime: pd.Timestamp
    exit_datetime: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    underlying_at_entry: float
    underlying_at_exit: float


# ---------------------------
# Data Preparation
# ---------------------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "DOWNLOAD_DATE", "DOWNLOAD_TIME", "SNAPSHOT_ID",
        "EXPIRY", "STRIKE", "c_OI", "c_LTP", "p_OI", "p_LTP", "UNDERLYING_VALUE"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["DOWNLOAD_DATE"] = pd.to_datetime(df["DOWNLOAD_DATE"]).dt.date
    df["DOWNLOAD_TIME"] = pd.to_datetime(df["DOWNLOAD_TIME"], format="%H:%M:%S").dt.time
    df["TIMESTAMP"] = pd.to_datetime(
        df["DOWNLOAD_DATE"].astype(str) + " " + df["DOWNLOAD_TIME"].astype(str)
    )

    df = df.sort_values(["DOWNLOAD_DATE", "SNAPSHOT_ID", "STRIKE"]).reset_index(drop=True)

    snap_keys = df[["DOWNLOAD_DATE", "SNAPSHOT_ID"]].drop_duplicates().reset_index(drop=True)
    snap_keys["SNAPSHOT_SEQ"] = range(len(snap_keys))
    df = df.merge(snap_keys, on=["DOWNLOAD_DATE", "SNAPSHOT_ID"], how="left")

    df["STRIKE"] = df["STRIKE"].astype(float)
    df["EXPIRY"] = df["EXPIRY"].astype(str)

    df = df.set_index(["SNAPSHOT_SEQ", "EXPIRY", "STRIKE"]).sort_index()
    return df


# ---------------------------
# Signal Generation
# ---------------------------

def generate_signals(df: pd.DataFrame, strike_step=DEFAULT_STRIKE_STEP, cooldown=DEFAULT_COOLDOWN):
    call_buy_signals, put_buy_signals = {}, {}

    df_r = df.reset_index()
    snap_list = sorted(df_r["SNAPSHOT_SEQ"].unique())
    under_by_snap = df_r.groupby("SNAPSHOT_SEQ")["UNDERLYING_VALUE"].first()
    exps_by_snap = df_r.groupby("SNAPSHOT_SEQ")["EXPIRY"].unique()

    last_call_entry_snap = -9999
    last_put_entry_snap = -9999

    for idx in range(len(snap_list) - 2):
        t0, t1, t2 = snap_list[idx], snap_list[idx + 1], snap_list[idx + 2]

        try:
            spot = under_by_snap.loc[t0]
            atm_strike = round(spot / strike_step) * strike_step
        except KeyError:
            continue

        valid_strikes = [atm_strike - strike_step, atm_strike, atm_strike + strike_step]

        try:
            u0, u1, u2 = under_by_snap.loc[t0], under_by_snap.loc[t1], under_by_snap.loc[t2]
            underlying_falling = (u2 < u1 < u0)
        except KeyError:
            underlying_falling = False

        for exp in exps_by_snap.loc[t0]:
            for strike in valid_strikes:
                key0, key1, key2 = (t0, exp, strike), (t1, exp, strike), (t2, exp, strike)
                if key0 not in df.index or key1 not in df.index or key2 not in df.index:
                    continue

                r0, r1, r2 = df.loc[key0], df.loc[key1], df.loc[key2]

                # CALL ENTRY
                if (
                    r2["c_LTP"] > r1["c_LTP"] > r0["c_LTP"] and
                    r2["c_LTP"] >= r0["c_LTP"] * 1.03 and
                    r2["c_OI"] >= r1["c_OI"] * 1.05 and
                    r0["c_LTP"] > 5 and
                    t2 - last_call_entry_snap > cooldown
                ):
                    call_buy_signals[t2] = (exp, strike)
                    last_call_entry_snap = t2

                # PUT ENTRY
                if (
                    underlying_falling and
                    r2["p_LTP"] > r1["p_LTP"] > r0["p_LTP"] and
                    r2["p_LTP"] >= r0["p_LTP"] * 1.03 and
                    r2["p_OI"] >= r1["p_OI"] * 1.05 and
                    r0["p_LTP"] > 5 and
                    t2 - last_put_entry_snap > cooldown
                ):
                    put_buy_signals[t2] = (exp, strike)
                    last_put_entry_snap = t2

    return call_buy_signals, put_buy_signals


# ---------------------------
# Backtest
# ---------------------------

def backtest(
    df, call_signals, put_signals,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    min_hold_snaps=DEFAULT_MIN_HOLD_SNAPS,
    lot_size=DEFAULT_LOT_SIZE
):
    df_r = df.reset_index()
    ts_map = (
        df_r.drop_duplicates(subset=["SNAPSHOT_SEQ"])
        .set_index("SNAPSHOT_SEQ")["TIMESTAMP"]
        .to_dict()
    )
    snap_list = sorted(df_r["SNAPSHOT_SEQ"].unique())

    capital = initial_capital
    call_pos, put_pos = None, None
    trades, equity_curve = [], []

    for idx, s in enumerate(snap_list):
        current_dt = ts_map[s]
        entry_date = current_dt.date()

        # EXIT LOGIC
        for pos_key, pos_var in [("CALL", call_pos), ("PUT", put_pos)]:
            if pos_var is not None:
                exp, strike = pos_var["expiry"], pos_var["strike"]
                qty, entry_price, entry_snap = pos_var["qty"], pos_var["entry_price"], pos_var["entry_snap"]
                stop_loss_price = entry_price * (1 - stop_loss_pct)

                if (s, exp, strike) in df.index:
                    curr = df.loc[(s, exp, strike)]
                    ltp_col = "c_LTP" if pos_key == "CALL" else "p_LTP"
                    oi_col = "c_OI" if pos_key == "CALL" else "p_OI"
                    curr_ltp = curr[ltp_col]

                    if idx > 0 and (snap_list[idx-1], exp, strike) in df.index:
                        prev = df.loc[(snap_list[idx-1], exp, strike)]
                        prev_ltp, prev_oi = prev[ltp_col], prev[oi_col]
                    else:
                        prev_ltp, prev_oi = curr_ltp, curr[oi_col]

                    exit_condition = (
                        curr_ltp <= stop_loss_price or
                        ((s - entry_snap) >= min_hold_snaps and curr_ltp < prev_ltp and curr[oi_col] < prev_oi)
                    )

                    if exit_condition:
                        exit_price = curr_ltp
                        capital += qty * exit_price
                        pnl = qty * (exit_price - entry_price)

                        trades.append(
                            Trade(
                                pos_key, exp, strike,
                                pos_var["entry_time"], current_dt,
                                entry_price, exit_price, qty, pnl,
                                pnl/(qty*entry_price),
                                pos_var["underlying_at_entry"],
                                float(curr["UNDERLYING_VALUE"])
                            )
                        )

                        if pos_key == "CALL":
                            call_pos = None
                        else:
                            put_pos = None

        # ENTRY CALL
        if s in call_signals and call_pos is None:
            exp, strike = call_signals[s]
            if entry_date != pd.to_datetime(exp).date() and (s, exp, strike) in df.index:
                row = df.loc[(s, exp, strike)]
                price = row["c_LTP"]
                if price > 0 and price * lot_size <= capital:
                    capital -= price * lot_size
                    call_pos = {
                        "expiry": exp,
                        "strike": strike,
                        "entry_time": current_dt,
                        "entry_price": price,
                        "qty": lot_size,
                        "entry_snap": s,
                        "underlying_at_entry": float(row["UNDERLYING_VALUE"]),
                    }

        # ENTRY PUT
        if s in put_signals and put_pos is None:
            exp, strike = put_signals[s]
            if entry_date != pd.to_datetime(exp).date() and (s, exp, strike) in df.index:
                row = df.loc[(s, exp, strike)]
                price = row["p_LTP"]
                if price > 0 and price * lot_size <= capital:
                    capital -= price * lot_size
                    put_pos = {
                        "expiry": exp,
                        "strike": strike,
                        "entry_time": current_dt,
                        "entry_price": price,
                        "qty": lot_size,
                        "entry_snap": s,
                        "underlying_at_entry": float(row["UNDERLYING_VALUE"]),
                    }

        # MARK EQUITY
        equity = capital
        if call_pos and (s, call_pos["expiry"], call_pos["strike"]) in df.index:
            equity += call_pos["qty"] * df.loc[(s, call_pos["expiry"], call_pos["strike"])]["c_LTP"]
        if put_pos and (s, put_pos["expiry"], put_pos["strike"]) in df.index:
            equity += put_pos["qty"] * df.loc[(s, put_pos["expiry"], put_pos["strike"])]["p_LTP"]

        equity_curve.append({"SNAPSHOT_SEQ": s, "TIMESTAMP": current_dt, "EQUITY": equity})

    equity_df = pd.DataFrame(equity_curve).set_index("SNAPSHOT_SEQ")
    trades_df = pd.DataFrame([asdict(t) for t in trades])

    # PERFORMANCE METRICS
    equity_series = equity_df["EQUITY"]
    equity_pct_change = equity_series.pct_change().dropna()

    drawdown_series = equity_series / equity_series.cummax() - 1
    max_drawdown_pct = float(drawdown_series.min() * 100)

    # CAGR
    first_ts = equity_df["TIMESTAMP"].iloc[0]
    last_ts = equity_df["TIMESTAMP"].iloc[-1]
    num_days = (last_ts - first_ts).days or 1
    years = num_days / 365
    cagr = (equity_series.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan

    # SHARPE/SORTINO
    if not equity_pct_change.empty:
        sharpe = equity_pct_change.mean() / equity_pct_change.std() * np.sqrt(252)
        downside = equity_pct_change[equity_pct_change < 0]
        sortino = equity_pct_change.mean() / downside.std() * np.sqrt(252) if downside.std() != 0 else 0
    else:
        sharpe = sortino = 0

    # PROFIT FACTOR
    total_profit = trades_df.loc[trades_df.pnl > 0, "pnl"].sum()
    total_loss = abs(trades_df.loc[trades_df.pnl < 0, "pnl"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    total_return_pct = float((equity_series.iloc[-1] / initial_capital - 1) * 100)
    calmar = total_return_pct / abs(max_drawdown_pct) if max_drawdown_pct != 0 else np.nan

    stats = {
        "initial_capital": initial_capital,
        "final_equity": float(equity_series.iloc[-1]),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": len(trades_df),
        "win_rate_pct": trades_df[trades_df.pnl > 0].shape[0] / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "profit_factor": profit_factor,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "cagr_pct": float(cagr * 100),
        "calmar_ratio": float(calmar),
        "start_date": first_ts,
        "end_date": last_ts,
        "num_days": num_days,
    }

    return trades_df, equity_df, stats



# ---------------------------
# Helper: Run Backtest
# ---------------------------

def run_single_backtest(df, params):
    call_sigs, put_sigs = generate_signals(
        df,
        strike_step=params.get("STRIKE_STEP", DEFAULT_STRIKE_STEP),
        cooldown=params.get("COOLDOWN", DEFAULT_COOLDOWN),
    )
    return backtest(
        df,
        call_sigs, put_sigs,
        initial_capital=params.get("INITIAL_CAPITAL", DEFAULT_INITIAL_CAPITAL),
        stop_loss_pct=params.get("STOP_LOSS_PCT", DEFAULT_STOP_LOSS_PCT),
        min_hold_snaps=params.get("MIN_HOLD_SNAPS", DEFAULT_MIN_HOLD_SNAPS),
        lot_size=params.get("LOT_SIZE", DEFAULT_LOT_SIZE),
    )


# ---------------------------
# Grid Search
# ---------------------------

def grid_search(df, param_grid):
    results = []
    for params in param_grid:
        _, _, stats = run_single_backtest(df, params)
        results.append({**params, **stats})
    return pd.DataFrame(results)


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    df_raw = pd.read_csv("NIFTY_OI.csv")
    df = prepare_data(df_raw)

    # ====== BASE BACKTEST ======
    base_params = {
        "STOP_LOSS_PCT": DEFAULT_STOP_LOSS_PCT,
        "COOLDOWN": DEFAULT_COOLDOWN,
        "MIN_HOLD_SNAPS": DEFAULT_MIN_HOLD_SNAPS,
        "STRIKE_STEP": DEFAULT_STRIKE_STEP,
        "INITIAL_CAPITAL": DEFAULT_INITIAL_CAPITAL,
        "LOT_SIZE": DEFAULT_LOT_SIZE,
    }

    trades_df, equity_df, stats = run_single_backtest(df, base_params)

    trades_df.to_csv("trades_log_final.csv", index=False)
    equity_df.to_csv("equity_curve_final.csv")

    # === Plot Base Curve ===
    plt.figure(figsize=(12,6))
    plt.plot(equity_df["TIMESTAMP"], equity_df["EQUITY"])
    plt.title("Equity Curve - Base Strategy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve_base.png")
    plt.close()

    print("\n=== BASE STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


import os
import json

if RUN_OPTIMIZATION:

    # --- Create master output folder ---
    ROOT_FOLDER = "backtest_results"
    os.makedirs(ROOT_FOLDER, exist_ok=True)

    # --- Build parameter grid ---
    param_grid = []
    for sl in [0.1,0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]: 
        for cd in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            for mh in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
                param_grid.append(
                    {
                        "STOP_LOSS_PCT": sl,
                        "COOLDOWN": cd,
                        "MIN_HOLD_SNAPS": mh,
                        "STRIKE_STEP": DEFAULT_STRIKE_STEP,
                        "INITIAL_CAPITAL": DEFAULT_INITIAL_CAPITAL,
                        "LOT_SIZE": DEFAULT_LOT_SIZE,
                    }
                )

    print(f"\nRunning {len(param_grid)} strategies...\n")

    # --- For collecting stats into optimization_results.csv ---
    all_results = []

    for params in param_grid:

        # Folder name = SL_0.2__CD_3__MH_3
        folder_name = f"SL_{params['STOP_LOSS_PCT']}_CD_{params['COOLDOWN']}_MH_{params['MIN_HOLD_SNAPS']}"
        folder_path = os.path.join(ROOT_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        print(f"Running: {folder_name}")

        # Run the backtest
        trades_x, equity_x, stats_x = run_single_backtest(df, params)

        # --- Save CSVs ---
        equity_csv_path = os.path.join(folder_path, "equity.csv")
        trades_csv_path = os.path.join(folder_path, "trades.csv")
        stats_json_path = os.path.join(folder_path, "stats.json")
        graph_png_path = os.path.join(folder_path, "equity.png")

        equity_x.to_csv(equity_csv_path)
        trades_x.to_csv(trades_csv_path, index=False)

        # --- Save stats JSON ---
        with open(stats_json_path, "w") as f:
            json.dump(stats_x, f, indent=4, default=str)

        # --- Plot and save equity curve ---
        plt.figure(figsize=(12, 6))
        plt.plot(equity_x["TIMESTAMP"], equity_x["EQUITY"])
        plt.title(f"Equity Curve - {folder_name}")
        plt.xlabel("Date/Time")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_png_path)
        plt.close()

        # Add to global optimization summary
        all_results.append({**params, **stats_x})

    # --- Save optimization summary ---
    opt_df = pd.DataFrame(all_results)
    opt_df.to_csv("optimization_results.csv", index=False)

    print("\nAll backtests complete!")
    print(f"Folders created in: {ROOT_FOLDER}")

    # Show top 10
    print("\n=== TOP STRATEGIES BY TOTAL RETURN ===")
    print(opt_df.sort_values("total_return_pct", ascending=False).head(10))

