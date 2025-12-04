import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# ----------------------------
# Config / parameters
# ----------------------------

LOT_SIZE = 300              # per your spec
INITIAL_CAPITAL = 3000.0
OI_RISE_THRESH = 0.03       # 3% step-up per snapshot (for building OI)
LTP_RISE_THRESH = 0.05      # 5% step-up per snapshot
OI_FLAT_THRESH = 0.01       # <=1% = flat
OI_FALL_MIN = 0.01          # >1% fall
LTP_FALL_MIN = 0.05         # >=5% fall
CALL_OI_FALL_FOR_PUT = 0.02  # >=2% fall on call OI for put buy
CALL_LTP_FALL_FOR_PUT = 0.05 # >=5% fall on call LTP for put buy

# ----------------------------
# Helper dataclasses
# ----------------------------

@dataclass
class Trade:
    direction: str          # "CALL" or "PUT"
    expiry: str
    strike: float
    entry_datetime: pd.Timestamp
    exit_datetime: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float


# ----------------------------
# Data preparation
# ----------------------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Parse dates
    - Create a global snapshot sequence (0..N-1) ordered by date & snapshot id
    - Create a timestamp column
    - Set MultiIndex (SNAPSHOT_SEQ, EXPIRY, STRIKE) for fast lookup
    """
    # Ensure required columns exist
    required_cols = [
        "DOWNLOAD_DATE", "DOWNLOAD_TIME", "SNAPSHOT_ID",
        "EXPIRY", "STRIKE",
        "c_OI", "c_LTP", "p_OI", "p_LTP"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    # Parse date/time
    df["DOWNLOAD_DATE"] = pd.to_datetime(df["DOWNLOAD_DATE"]).dt.date
    df["DOWNLOAD_TIME"] = pd.to_datetime(df["DOWNLOAD_TIME"], format="%H:%M:%S").dt.time

    # Timestamp = date + time
    df["TIMESTAMP"] = pd.to_datetime(
        df["DOWNLOAD_DATE"].astype(str) + " " + df["DOWNLOAD_TIME"].astype(str)
    )

    # Sort by date, snapshot, strike (for deterministic idxmax)
    df = df.sort_values(["DOWNLOAD_DATE", "SNAPSHOT_ID", "STRIKE"]).reset_index(drop=True)

    # Build global snapshot sequence across all days
    snap_keys = df[["DOWNLOAD_DATE", "SNAPSHOT_ID"]].drop_duplicates().reset_index(drop=True)
    snap_keys["SNAPSHOT_SEQ"] = range(len(snap_keys))

    df = df.merge(snap_keys, on=["DOWNLOAD_DATE", "SNAPSHOT_ID"], how="left")

    # Ensure types
    df["STRIKE"] = df["STRIKE"].astype(float)
    df["EXPIRY"] = df["EXPIRY"].astype(str)

    # MultiIndex for fast slicing: (SNAPSHOT_SEQ, EXPIRY, STRIKE)
    df = df.set_index(["SNAPSHOT_SEQ", "EXPIRY", "STRIKE"]).sort_index()

    return df


# ----------------------------
# Pre-compute highest OI per snapshot
# ----------------------------

def get_max_oi_by_snapshot(df: pd.DataFrame):
    """
    Returns:
        call_max: DataFrame indexed by SNAPSHOT_SEQ with columns [EXPIRY, STRIKE]
                  representing highest c_OI in that snapshot.
        put_max:  DataFrame indexed by SNAPSHOT_SEQ with columns [EXPIRY, STRIKE]
                  representing highest p_OI in that snapshot.
    """
    # Reset index to group by snapshot
    tmp = df.reset_index()

    # Highest call OI per snapshot
    call_idx = tmp.groupby("SNAPSHOT_SEQ")["c_OI"].idxmax()
    call_max = tmp.loc[call_idx, ["SNAPSHOT_SEQ", "EXPIRY", "STRIKE"]].set_index("SNAPSHOT_SEQ")

    # Highest put OI per snapshot
    put_idx = tmp.groupby("SNAPSHOT_SEQ")["p_OI"].idxmax()
    put_max = tmp.loc[put_idx, ["SNAPSHOT_SEQ", "EXPIRY", "STRIKE"]].set_index("SNAPSHOT_SEQ")

    return call_max, put_max


# ----------------------------
# Signal generation
# ----------------------------

def safe_positive(values):
    """Return False if any value <= 0 (used to drop trades when 0 shows up)."""
    values = np.array(values, dtype=float)
    return np.all(values > 0)


def generate_signals(df: pd.DataFrame, call_max: pd.DataFrame, put_max: pd.DataFrame):
    """
    For each window of 3 consecutive snapshots [t, t+1, t+2]:

    - For CALL BUY:
        * Candidate is strike with highest Call OI at snapshot t
        * Use that same contract at t, t+1, t+2
        * CE_OI rising > OI_RISE_THRESH each step
        * CE_LTP rising > LTP_RISE_THRESH each step
        * PE (highest OI at t) is either:
            - OI flat (<=1% overall) across t -> t+2
            - OR OI falling >1% AND LTP falling >=5% overall t -> t+2
        * All OI/LTP values in window must be > 0 (otherwise drop trade).

        Signal is placed at snapshot t+2.

    - For PUT BUY:
        * Candidate is strike with highest Put OI at snapshot t
        * Use that same contract at t, t+1, t+2
        * PE_OI rising > OI_RISE_THRESH each step
        * PE_LTP rising > LTP_RISE_THRESH each step
        * CE at that time (highest Call OI at t) is weakening:
            - Call OI falling >= CALL_OI_FALL_FOR_PUT overall t -> t+2
            - AND Call LTP falling >= CALL_LTP_FALL_FOR_PUT overall t -> t+2
        * All relevant OI/LTP values > 0.

        Signal is placed at snapshot t+2.

    Returns:
        call_buy_signals, put_buy_signals
        Each is a dict: { snapshot_seq: (expiry, strike) } if buy, else no key.
    """
    snap_list = sorted(df.reset_index()["SNAPSHOT_SEQ"].unique())
    n_snaps = len(snap_list)

    call_buy_signals = {}
    put_buy_signals = {}

    # For fast lookup
    # df is MultiIndex: (SNAPSHOT_SEQ, EXPIRY, STRIKE)
    for idx in range(n_snaps - 2):
        t0 = snap_list[idx]
        t1 = snap_list[idx + 1]
        t2 = snap_list[idx + 2]

        # ----- CALL BUY window -----
        # Candidate: highest Call OI at t0
        cm0 = call_max.loc[t0]
        c_exp, c_strike = cm0["EXPIRY"], cm0["STRIKE"]

        # Candidate: highest Put OI at t0 (for PE-side filter)
        pm0 = put_max.loc[t0]
        p_exp, p_strike = pm0["EXPIRY"], pm0["STRIKE"]

        try:
            c0 = df.loc[(t0, c_exp, c_strike)]
            c1 = df.loc[(t1, c_exp, c_strike)]
            c2 = df.loc[(t2, c_exp, c_strike)]

            p0 = df.loc[(t0, p_exp, p_strike)]
            p1 = df.loc[(t1, p_exp, p_strike)]
            p2 = df.loc[(t2, p_exp, p_strike)]
        except KeyError:
            # Require continuous presence, otherwise skip
            c0 = c1 = c2 = p0 = p1 = p2 = None

        if c0 is not None:
            ce_oi = [c0["c_OI"], c1["c_OI"], c2["c_OI"]]
            ce_ltp = [c0["c_LTP"], c1["c_LTP"], c2["c_LTP"]]
            pe_oi = [p0["p_OI"], p1["p_OI"], p2["p_OI"]]
            pe_ltp = [p0["p_LTP"], p1["p_LTP"], p2["p_LTP"]]

            # Drop trade if any zero/negative
            if safe_positive(ce_oi + ce_ltp + pe_oi + pe_ltp):
                # Call trend up (step-wise)
                ce_oi_up = ce_oi[2] >= ce_oi[0] * (1 + OI_RISE_THRESH)
                           
                ce_ltp_up = ce_ltp[2] >= ce_ltp[0] * (1 + LTP_RISE_THRESH)
                            

                # Put flat / falling across the window (overall t0 -> t2)
                pe_oi0, pe_oi2 = pe_oi[0], pe_oi[2]
                pe_ltp0, pe_ltp2 = pe_ltp[0], pe_ltp[2]

                pe_oi_flat = abs(pe_oi2 - pe_oi0) / pe_oi0 <= OI_FLAT_THRESH
                pe_oi_fall = pe_oi2 <= pe_oi0 * (1 - OI_FALL_MIN)
                pe_ltp_fall = pe_ltp2 <= pe_ltp0 * (1 - LTP_FALL_MIN)

                pe_ok = pe_oi_flat or (pe_oi_fall and pe_ltp_fall)

                if ce_oi_up and ce_ltp_up and pe_ok:
                    # Place CALL BUY signal at t2
                    call_buy_signals[t2] = (c_exp, float(c_strike))

        # ----- PUT BUY window -----
        # Candidate: highest Put OI at t0
        pm0 = put_max.loc[t0]
        p_exp2, p_strike2 = pm0["EXPIRY"], pm0["STRIKE"]

        # For CE-side weakness, use highest Call OI at t0
        cm0b = call_max.loc[t0]
        c_exp2, c_strike2 = cm0b["EXPIRY"], cm0b["STRIKE"]

        try:
            p0b = df.loc[(t0, p_exp2, p_strike2)]
            p1b = df.loc[(t1, p_exp2, p_strike2)]
            p2b = df.loc[(t2, p_exp2, p_strike2)]

            c0b = df.loc[(t0, c_exp2, c_strike2)]
            c1b = df.loc[(t1, c_exp2, c_strike2)]
            c2b = df.loc[(t2, c_exp2, c_strike2)]
        except KeyError:
            p0b = p1b = p2b = c0b = c1b = c2b = None

        if p0b is not None:
            pe_oi2_arr = [p0b["p_OI"], p1b["p_OI"], p2b["p_OI"]]
            pe_ltp2_arr = [p0b["p_LTP"], p1b["p_LTP"], p2b["p_LTP"]]
            ce_oi2_arr = [c0b["c_OI"], c1b["c_OI"], c2b["c_OI"]]
            ce_ltp2_arr = [c0b["c_LTP"], c1b["c_LTP"], c2b["c_LTP"]]

            if safe_positive(pe_oi2_arr + pe_ltp2_arr + ce_oi2_arr + ce_ltp2_arr):
                # Put trend up (step-wise)
                pe_oi_up = pe_oi2_arr[2] >= pe_oi2_arr[0] * (1 + OI_RISE_THRESH)
                           
                pe_ltp_up = pe_ltp2_arr[2] >= pe_ltp2_arr[0] * (1 + LTP_RISE_THRESH)
                            

                # Call weakening overall
                ce_oi0b, ce_oi2b = ce_oi2_arr[0], ce_oi2_arr[2]
                ce_ltp0b, ce_ltp2b = ce_ltp2_arr[0], ce_ltp2_arr[2]

                ce_oi_fall_enough = ce_oi2b <= ce_oi0b * (1 - CALL_OI_FALL_FOR_PUT)
                ce_ltp_fall_enough = ce_ltp2b <= ce_ltp0b * (1 - CALL_LTP_FALL_FOR_PUT)

                ce_ok = ce_oi_fall_enough and ce_ltp_fall_enough

                if pe_oi_up and pe_ltp_up and ce_ok:
                    put_buy_signals[t2] = (p_exp2, float(p_strike2))

    return call_buy_signals, put_buy_signals


# ----------------------------
# Backtest engine
# ----------------------------

def backtest(df: pd.DataFrame,
             call_max: pd.DataFrame,
             put_max: pd.DataFrame,
             call_buy_signals: dict,
             put_buy_signals: dict):
    """
    Runs the trading simulation:

    - Only longs (no short options).
    - Max 1 CALL position and 1 PUT position at a time.
    - Position size: LOT_SIZE contracts (300) if capital can afford premium (cost <= capital).
    - Positions can carry across days.
    - Exit rules:
        * CALL:
           - CE_OI falling AND CE_LTP falling vs previous snapshot
           OR
           - PE_OI rising OR PE_LTP rising (highest OI put).
        * PUT:
           - PE_OI falling AND PE_LTP falling vs previous snapshot
           OR
           - CE_OI rising OR CE_LTP rising (highest OI call).

    Capital accounting:
    - At entry: capital -= qty * entry_price
    - At exit: capital += qty * exit_price
    - Equity = capital + MTM value of open positions each snapshot.

    Returns:
        trades_df, equity_df, stats_dict
    """
    # Reset_index for timestamps
    base = df.reset_index()[["SNAPSHOT_SEQ", "EXPIRY", "STRIKE", "TIMESTAMP"]]
    ts_map = base.drop_duplicates(subset=["SNAPSHOT_SEQ"])[["SNAPSHOT_SEQ", "TIMESTAMP"]].set_index("SNAPSHOT_SEQ")["TIMESTAMP"].to_dict()

    snap_list = sorted(df.reset_index()["SNAPSHOT_SEQ"].unique())
    n_snaps = len(snap_list)

    capital = INITIAL_CAPITAL
    call_pos = None  # dict or None
    put_pos = None

    equity_curve = []
    trades = []

    for idx, s in enumerate(snap_list):
        # ---- EXIT logic first ----
        # Need previous snapshot for comparisons
        if idx > 0:
            s_prev = snap_list[idx - 1]

            # CALL EXIT
            if call_pos is not None:
                c_exp, c_strike = call_pos["expiry"], call_pos["strike"]

                try:
                    curr_row = df.loc[(s, c_exp, c_strike)]
                    prev_row = df.loc[(s_prev, c_exp, c_strike)]
                except KeyError:
                    curr_row = prev_row = None

                exit_call = False

                if curr_row is not None:
                    ce_oi_curr = curr_row["c_OI"]
                    ce_oi_prev = prev_row["c_OI"]
                    ce_ltp_curr = curr_row["c_LTP"]
                    ce_ltp_prev = prev_row["c_LTP"]

                    ce_falling = (ce_oi_curr < ce_oi_prev) and (ce_ltp_curr < ce_ltp_prev)

                    # Highest put that snapshot for PE-side condition
                    pm_curr = put_max.loc[s]
                    pm_prev = put_max.loc[s_prev]

                    try:
                        pe_curr = df.loc[(s, pm_curr["EXPIRY"], pm_curr["STRIKE"])]
                        pe_prev = df.loc[(s_prev, pm_prev["EXPIRY"], pm_prev["STRIKE"])]
                    except KeyError:
                        pe_curr = pe_prev = None

                    pe_rising = False
                    if pe_curr is not None:
                        pe_rising = (pe_curr["p_OI"] > pe_prev["p_OI"]) or (pe_curr["p_LTP"] > pe_prev["p_LTP"])

                    exit_call = ce_falling or pe_rising

                if exit_call and curr_row is not None:
                    exit_price = curr_row["c_LTP"]
                    qty = call_pos["qty"]

                    capital += qty * exit_price
                    pnl = qty * (exit_price - call_pos["entry_price"])
                    ret_pct = pnl / (call_pos["entry_price"] * qty)

                    trades.append(Trade(
                        direction="CALL",
                        expiry=call_pos["expiry"],
                        strike=call_pos["strike"],
                        entry_datetime=call_pos["entry_time"],
                        exit_datetime=ts_map[s],
                        entry_price=call_pos["entry_price"],
                        exit_price=exit_price,
                        quantity=qty,
                        pnl=pnl,
                        return_pct=ret_pct
                    ))

                    call_pos = None

            # PUT EXIT
            if put_pos is not None:
                p_exp, p_strike = put_pos["expiry"], put_pos["strike"]

                try:
                    curr_row = df.loc[(s, p_exp, p_strike)]
                    prev_row = df.loc[(s_prev, p_exp, p_strike)]
                except KeyError:
                    curr_row = prev_row = None

                exit_put = False

                if curr_row is not None:
                    pe_oi_curr = curr_row["p_OI"]
                    pe_oi_prev = prev_row["p_OI"]
                    pe_ltp_curr = curr_row["p_LTP"]
                    pe_ltp_prev = prev_row["p_LTP"]

                    pe_falling = (pe_oi_curr < pe_oi_prev) and (pe_ltp_curr < pe_ltp_prev)

                    # Highest call that snapshot for CE-side condition
                    cm_curr = call_max.loc[s]
                    cm_prev = call_max.loc[s_prev]

                    try:
                        ce_curr = df.loc[(s, cm_curr["EXPIRY"], cm_curr["STRIKE"])]
                        ce_prev = df.loc[(s_prev, cm_prev["EXPIRY"], cm_prev["STRIKE"])]
                    except KeyError:
                        ce_curr = ce_prev = None

                    ce_rising = False
                    if ce_curr is not None:
                        ce_rising = (ce_curr["c_OI"] > ce_prev["c_OI"]) or (ce_curr["c_LTP"] > ce_prev["c_LTP"])

                    exit_put = pe_falling or ce_rising

                if exit_put and curr_row is not None:
                    exit_price = curr_row["p_LTP"]
                    qty = put_pos["qty"]

                    capital += qty * exit_price
                    pnl = qty * (exit_price - put_pos["entry_price"])
                    ret_pct = pnl / (put_pos["entry_price"] * qty)

                    trades.append(Trade(
                        direction="PUT",
                        expiry=put_pos["expiry"],
                        strike=put_pos["strike"],
                        entry_datetime=put_pos["entry_time"],
                        exit_datetime=ts_map[s],
                        entry_price=put_pos["entry_price"],
                        exit_price=exit_price,
                        quantity=qty,
                        pnl=pnl,
                        return_pct=ret_pct
                    ))

                    put_pos = None

        # ---- BUY logic ----
        # CALL BUY
        if (s in call_buy_signals) and (call_pos is None):
            c_exp, c_strike = call_buy_signals[s]
            try:
                row = df.loc[(s, c_exp, c_strike)]
                entry_price = row["c_LTP"]
            except KeyError:
                row = None

            if row is not None and entry_price > 0:
                cost = LOT_SIZE * entry_price
                if cost <= capital:  # Only buy if capital is enough
                    capital -= cost
                    call_pos = {
                        "expiry": c_exp,
                        "strike": c_strike,
                        "entry_time": ts_map[s],
                        "entry_price": entry_price,
                        "qty": LOT_SIZE
                    }

        # PUT BUY
        if (s in put_buy_signals) and (put_pos is None):
            p_exp, p_strike = put_buy_signals[s]
            try:
                row = df.loc[(s, p_exp, p_strike)]
                entry_price = row["p_LTP"]
            except KeyError:
                row = None

            if row is not None and entry_price > 0:
                cost = LOT_SIZE * entry_price
                if cost <= capital:  # Only buy if capital is enough
                    capital -= cost
                    put_pos = {
                        "expiry": p_exp,
                        "strike": p_strike,
                        "entry_time": ts_map[s],
                        "entry_price": entry_price,
                        "qty": LOT_SIZE
                    }

        # ---- Equity mark-to-market ----
        equity = capital

        # Mark CALL position
        if call_pos is not None:
            c_exp, c_strike = call_pos["expiry"], call_pos["strike"]
            try:
                row = df.loc[(s, c_exp, c_strike)]
                mtm_price = row["c_LTP"]
            except KeyError:
                mtm_price = call_pos["entry_price"]  # fallback
            equity += call_pos["qty"] * mtm_price

        # Mark PUT position
        if put_pos is not None:
            p_exp, p_strike = put_pos["expiry"], put_pos["strike"]
            try:
                row = df.loc[(s, p_exp, p_strike)]
                mtm_price = row["p_LTP"]
            except KeyError:
                mtm_price = put_pos["entry_price"]
            equity += put_pos["qty"] * mtm_price

        equity_curve.append({
            "SNAPSHOT_SEQ": s,
            "TIMESTAMP": ts_map[s],
            "EQUITY": equity
        })

    # ----------------------
    # Build outputs
    # ----------------------
    equity_df = pd.DataFrame(equity_curve).set_index("SNAPSHOT_SEQ")
    trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame(columns=[
        "direction", "expiry", "strike",
        "entry_datetime", "exit_datetime",
        "entry_price", "exit_price", "quantity",
        "pnl", "return_pct"
    ])

    # Compute performance stats
    stats = {}

    if not equity_df.empty:
        final_equity = equity_df["EQUITY"].iloc[-1]
        total_return = final_equity / INITIAL_CAPITAL - 1.0

        t0 = equity_df["TIMESTAMP"].iloc[0]
        tn = equity_df["TIMESTAMP"].iloc[-1]
        days = max((tn - t0).total_seconds() / (24 * 3600), 1e-6)
        years = days / 365.25
        cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1 if final_equity > 0 else np.nan

        # Max drawdown
        eq = equity_df["EQUITY"].values
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0

        stats = {
            "initial_capital": INITIAL_CAPITAL,
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_dd * 100
        }

    return trades_df, equity_df, stats


# ----------------------------
# Main usage
# ----------------------------

if __name__ == "__main__":
    # CHANGE THIS: path to your CSV
    csv_path = "NIFTY_OI.csv"

    raw_df = pd.read_csv(csv_path)
    df = prepare_data(raw_df)

    # Precompute max OI strikes
    call_max, put_max = get_max_oi_by_snapshot(df)

    # Generate entry signals
    call_buy_signals, put_buy_signals = generate_signals(df, call_max, put_max)

    # Run backtest
    trades_df, equity_df, stats = backtest(df, call_max, put_max, call_buy_signals, put_buy_signals)

    # Print / save outputs
    print("=== STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== SAMPLE TRADES ===")
    print(trades_df.head())

    # Optionally save
    trades_df.to_csv("trades_log.csv", index=False)
    equity_df.to_csv("equity_curve.csv")
