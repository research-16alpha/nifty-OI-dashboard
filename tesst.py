import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import math

LOT_SIZE = 3000
INITIAL_CAPITAL = 1000000.0
STOP_LOSS_PCT = 0.30
MIN_HOLD_SNAPS = 5
STRIKE_STEP = 50
COOLDOWN = 5  #snapshot cooldown

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

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "DOWNLOAD_DATE", "DOWNLOAD_TIME", "SNAPSHOT_ID",
        "EXPIRY", "STRIKE", "c_OI", "c_LTP", "p_OI", "p_LTP", "UNDERLYING_VALUE"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["DOWNLOAD_DATE"] = pd.to_datetime(df["DOWNLOAD_DATE"]).dt.date
    df["DOWNLOAD_TIME"] = pd.to_datetime(df["DOWNLOAD_TIME"]).dt.time
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

def generate_signals(df: pd.DataFrame):
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
            atm_strike = round(spot / STRIKE_STEP) * STRIKE_STEP
        except KeyError:
            continue

        valid_strikes = [atm_strike - STRIKE_STEP, atm_strike, atm_strike + STRIKE_STEP]

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

                if (
                    r2["c_LTP"] > r1["c_LTP"] > r0["c_LTP"] and
                    r2["c_LTP"] >= r0["c_LTP"] * 1.03 and
                    r2["c_OI"] >= r1["c_OI"] * 1.05 and
                    r0["c_LTP"] > 5 and
                    t2 - last_call_entry_snap > COOLDOWN
                ):
                    call_buy_signals[t2] = (exp, strike)
                    last_call_entry_snap = t2

                if (
                    underlying_falling and
                    r2["p_LTP"] > r1["p_LTP"] > r0["p_LTP"] and
                    r2["p_LTP"] >= r0["p_LTP"] * 1.03 and
                    r2["p_OI"] >= r1["p_OI"] * 1.05 and
                    r0["p_LTP"] > 5 and
                    t2 - last_put_entry_snap > COOLDOWN
                ):
                    put_buy_signals[t2] = (exp, strike)
                    last_put_entry_snap = t2

    return call_buy_signals, put_buy_signals

def backtest(df, call_signals, put_signals):
    df_r = df.reset_index()
    ts_map = df_r.drop_duplicates(subset=["SNAPSHOT_SEQ"]).set_index("SNAPSHOT_SEQ")["TIMESTAMP"].to_dict()
    snap_list = sorted(df_r["SNAPSHOT_SEQ"].unique())

    capital = INITIAL_CAPITAL
    call_pos, put_pos = None, None
    trades, equity_curve = [], []

    for idx, s in enumerate(snap_list):
        current_dt = ts_map[s]
        entry_date = current_dt.date()

        for pos_key, pos_var in [("CALL", call_pos), ("PUT", put_pos)]:
            if pos_var is not None:
                exp, strike = pos_var["expiry"], pos_var["strike"]
                qty, entry_price, entry_snap = pos_var["qty"], pos_var["entry_price"], pos_var["entry_snap"]
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)

                if (s, exp, strike) in df.index:
                    curr = df.loc[(s, exp, strike)]
                    curr_ltp = curr["c_LTP"] if pos_key == "CALL" else curr["p_LTP"]

                    if curr_ltp <= stop_loss_price or (
                        (s - entry_snap) >= MIN_HOLD_SNAPS and idx > 0 and
                        curr_ltp < df.loc[(snap_list[idx-1], exp, strike)][f"{pos_key.lower()[0]}_LTP"] and
                        curr[f"{pos_key.lower()[0]}_OI"] < df.loc[(snap_list[idx-1], exp, strike)][f"{pos_key.lower()[0]}_OI"]
                    ):
                        exit_price = curr_ltp
                        capital += qty * exit_price
                        pnl = qty * (exit_price - entry_price)

                        trades.append(Trade(
                            pos_key, exp, strike,
                            pos_var["entry_time"], current_dt,
                            entry_price, exit_price,
                            qty, pnl, pnl/(qty*entry_price),
                            pos_var["underlying_at_entry"],
                            float(curr["UNDERLYING_VALUE"])
                        ))
                        if pos_key == "CALL":
                            call_pos = None
                        else:
                            put_pos = None

        if s in call_signals and call_pos is None:
            exp, strike = call_signals[s]
            if entry_date != pd.to_datetime(exp).date() and (s, exp, strike) in df.index:
                row = df.loc[(s, exp, strike)]
                price = row["c_LTP"]
                if price > 0 and price * LOT_SIZE <= capital:
                    capital -= price * LOT_SIZE
                    call_pos = {
                        "expiry": exp,
                        "strike": strike,
                        "entry_time": current_dt,
                        "entry_price": price,
                        "qty": LOT_SIZE,
                        "entry_snap": s,
                        "underlying_at_entry": float(row["UNDERLYING_VALUE"])
                    }

        if s in put_signals and put_pos is None:
            exp, strike = put_signals[s]
            if entry_date != pd.to_datetime(exp).date() and (s, exp, strike) in df.index:
                row = df.loc[(s, exp, strike)]
                price = row["p_LTP"]
                if price > 0 and price * LOT_SIZE <= capital:
                    capital -= price * LOT_SIZE
                    put_pos = {
                        "expiry": exp,
                        "strike": strike,
                        "entry_time": current_dt,
                        "entry_price": price,
                        "qty": LOT_SIZE,
                        "entry_snap": s,
                        "underlying_at_entry": float(row["UNDERLYING_VALUE"])
                    }

        equity = capital
        if call_pos is not None and (s, call_pos["expiry"], call_pos["strike"]) in df.index:
            equity += call_pos["qty"] * df.loc[(s, call_pos["expiry"], call_pos["strike"])]["c_LTP"]
        if put_pos is not None and (s, put_pos["expiry"], put_pos["strike"]) in df.index:
            equity += put_pos["qty"] * df.loc[(s, put_pos["expiry"], put_pos["strike"])]["p_LTP"]
        equity_curve.append({"SNAPSHOT_SEQ": s, "TIMESTAMP": current_dt, "EQUITY": equity})

    equity_df = pd.DataFrame(equity_curve).set_index("SNAPSHOT_SEQ")
    trades_df = pd.DataFrame([asdict(t) for t in trades])

    # ---------------- PERFORMANCE METRICS -----------------

    total_profit = trades_df[trades_df.pnl > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df.pnl < 0]['pnl'].sum())
    win_rate = len(trades_df[trades_df.pnl > 0]) / len(trades_df) * 100 if len(trades_df) else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    avg_win = trades_df[trades_df.pnl > 0]['pnl'].mean() if total_profit else 0
    avg_loss = trades_df[trades_df.pnl < 0]['pnl'].mean() if total_loss else 0

    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    equity_pct_change = equity_df['EQUITY'].pct_change().dropna()
    sharpe_ratio = (equity_pct_change.mean() / equity_pct_change.std()) * np.sqrt(252) if equity_pct_change.std() else 0

    negative_returns = equity_pct_change[equity_pct_change < 0]
    sortino_ratio = (equity_pct_change.mean() / negative_returns.std()) * np.sqrt(252) if negative_returns.std() else 0

    max_drawdown_pct = ((equity_df['EQUITY'] / equity_df['EQUITY'].cummax()) - 1).min() * 100

    trade_returns = trades_df['return_pct']
    best_trade = trade_returns.max() if not trades_df.empty else None
    worst_trade = trade_returns.min() if not trades_df.empty else None

    recovery_factor = (equity_df['EQUITY'].iloc[-1] - INITIAL_CAPITAL) / abs(max_drawdown_pct) if max_drawdown_pct else 0

    stats = {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(equity_df["EQUITY"].iloc[-1]),
        "total_return_pct": float((equity_df["EQUITY"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": len(trades_df),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade_pct": best_trade,
        "worst_trade_pct": worst_trade,
        "recovery_factor": recovery_factor,
    }

    return trades_df, equity_df, stats


if __name__ == "__main__":
    df_raw = pd.read_csv("NIFTY_OI.csv")
    df = prepare_data(df_raw)

    call_signals, put_signals = generate_signals(df)

    trades_df, equity_df, stats = backtest(df, call_signals, put_signals)

    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    trades_df.to_csv("trades_log_final.csv", index=False)
    equity_df.to_csv("equity_curve_final.csv")
