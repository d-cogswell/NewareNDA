import logging
import numpy as np
from .dicts import state_dict

logger = logging.getLogger('newarenda')

charge_keys = [k for k,v in state_dict.items() if v.endswith("_Chg")]
discharge_keys = [k for k,v in state_dict.items() if v.endswith("_DChg")]

def _generate_cycle_number(df, cycle_mode='chg'):
    """
    Generate a cycle number to match Neware.

    cycle_mode = chg: (Default) Sets new cycles with a Charge step following a Discharge.
        dchg: Sets new cycles with a Discharge step following a Charge.
        auto: Identifies the first non-rest state as the incremental state.
    """

    # Auto: find the first non rest cycle
    if cycle_mode.lower() == 'auto':
        cycle_mode = _id_first_state(df)

    # Set increment key and non-increment/off key
    if cycle_mode.lower() == "chg":
        inkeys = charge_keys
        offkeys = discharge_keys + [17]
    elif cycle_mode.lower() == "dchg":
        inkeys = discharge_keys
        offkeys = charge_keys + [17]
    else:
        logger.error(f"Cycle_Mode '{cycle_mode}' not recognized. Supported options are 'chg', 'dchg', and 'auto'.")
        raise KeyError(f"Cycle_Mode '{cycle_mode}' not recognized. Supported options are 'chg', 'dchg', and 'auto'.")

    incs = df["Status"].is_in(inkeys).to_numpy()
    flags = df["Status"].is_in(offkeys).to_numpy()
    cycles = np.zeros(len(df), dtype=int)
    cyc = 1
    flag = False
    for i in range(len(df)):
        if not flag and flags[i]:
            flag=True
        elif flag and incs[i]:
            cyc += 1
            flag = False
        cycles[i] = cyc
    return cycles


def _count_changes(series):
    """Enumerate the number of value changes in a series"""
    return series.diff().fill_null(1).abs().gt(0).cum_sum()


def _id_first_state(df):
    """Helper function to identify the first non-rest state in a cycling profile"""
    mask = df["Status"].is_in(charge_keys + discharge_keys)
    # If there is chg/dchg and first state is chg, return "chg"
    if mask.any() and df["Status"][mask.arg_min()] in charge_keys:
        return "chg"
    # If first state is dchg or if no chg/dchg, return "dchg"
    return "dchg"

