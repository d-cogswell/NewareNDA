import logging
from typing import Literal

import numpy as np
import polars as pl

from .dicts import state_dict

logger = logging.getLogger('newarenda')

charge_keys = [k for k,v in state_dict.items() if v.endswith("_Chg")]
discharge_keys = [k for k,v in state_dict.items() if v.endswith("_DChg")]

def _generate_cycle_number(
        df: pl.DataFrame,
        cycle_mode: Literal["chg", "dchg", "auto"] = "chg",
    ) -> pl.DataFrame:
    """Generate a cycle number to match Neware.

    cycle_mode = chg: (Default) Sets new cycles with a Charge step following a Discharge.
        dchg: Sets new cycles with a Discharge step following a Charge.
        auto: Identifies the first non-rest state as the incremental state.
    """
    # Auto: find the first non rest cycle
    if cycle_mode.lower() == "auto":
        cycle_mode = _id_first_state(df)

    # Set increment key and non-increment/off key
    if cycle_mode == "chg":
        inkeys = charge_keys
        offkeys = [*discharge_keys, 17]
    elif cycle_mode == "dchg":
        inkeys = discharge_keys
        offkeys = [*charge_keys, 17]
    else:
        msg = "Cycle_Mode %s not recognized. Supported options are 'chg', 'dchg', and 'auto'."
        raise KeyError(msg, cycle_mode)

    incs = df["Status"].is_in(inkeys).to_numpy()
    flags = df["Status"].is_in(offkeys).to_numpy()
    cycles = np.zeros(len(df), dtype=np.uint32)
    cycles[0] = np.uint32(1)
    flag = False
    for i in range(len(df)):
        if not flag and flags[i]:
            flag=True
        elif flag and incs[i]:
            flag = False
            cycles[i] = np.uint32(1)
    return df.with_columns(
        pl.Series(name="Cycle", values = cycles, dtype=pl.UInt32).cum_sum()
    )


def _count_changes(series: pl.Series) -> pl.Series:
    """Enumerate the number of value changes in a series."""
    return series.diff().fill_null(1).abs().gt(0).cum_sum()


def _id_first_state(df: pl.DataFrame) -> str:
    """Identify the first non-rest state in the DataFrame."""
    # Filter on non-rest keys, check first row
    filtered = df.filter(pl.col("Status").is_in(charge_keys + discharge_keys)).head(1)
    if not filtered.is_empty() and filtered[0, "Status"] in charge_keys:
        return "chg"
    return "dchg"
