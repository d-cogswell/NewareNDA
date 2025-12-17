import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

import NewareNDA
from NewareNDA.dicts import state_dict


def test_NewareNDA(nda_file, ref_file, software_cycle_number, cycle_mode, btsda_ref_file):
    df = NewareNDA.read(nda_file, software_cycle_number, cycle_mode)
    ref_df = pd.read_feather(ref_file)

    # Convert dates to timestamps for comparison
    df['Timestamp'] = df['Timestamp'].apply(datetime.timestamp)
    ref_df['Timestamp'] = ref_df['Timestamp'].apply(datetime.timestamp)

    pd.testing.assert_frame_equal(df, ref_df, check_like=True)


def test_NewareNDAcli(nda_file, ref_file, software_cycle_number, cycle_mode, btsda_ref_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, os.path.basename(nda_file))
        os.system(
            f"python -m NewareNDA --format=feather "
            f"{'' if software_cycle_number else '--no_software_cycle_number'} "
            f"--cycle_mode={cycle_mode} "
            f"\"{nda_file}\" \"{filename}.ftr\"")
        df = pd.read_feather(f"{filename}.ftr")
    ref_df = pd.read_feather(ref_file)

    # Convert dates to timestamps for comparison
    df['Timestamp'] = df['Timestamp'].apply(datetime.timestamp)
    ref_df['Timestamp'] = ref_df['Timestamp'].apply(datetime.timestamp)

    pd.testing.assert_frame_equal(df, ref_df, check_like=True)

def test_vs_btsda(
    nda_file: str,
    btsda_ref_file: str,
    ref_file: str,
    software_cycle_number: bool,
    cycle_mode: str,
) -> None:
    """Check if NewareNDA output matches BTSDA reference."""
    if not Path(btsda_ref_file).exists():
        warnings.warn(f"No BTSDA reference found for {nda_file}", stacklevel=2)
        return

    df = NewareNDA.read(nda_file, software_cycle_number=False)
    ref_df = pd.read_parquet(btsda_ref_file)

    # Voltage
    pd.testing.assert_series_equal(
        df["Voltage"],
        ref_df["Voltage(mV)"]/1000,
        atol=5e-4, # should be accurate to 1 mV
        check_names=False,
        check_dtype=False,
    )

    # Current
    pd.testing.assert_series_equal(
        df["Current(mA)"],
        ref_df["Current(uA)"]/1000,
        atol=5e-4, # should be accurate to 1 uA
        check_names=False,
        check_dtype=False,
    )

    # Time (within a step)
    pd.testing.assert_series_equal(
        df["Time"],
        ref_df["Time"],
        atol=5e-4, # should be accurate to 1 ms
        check_names=False,
        check_dtype=False,
    )

    # Step type
    step_mapping = {v:k for k,v in state_dict.items()}
    df["Status"] = df["Status"].map(step_mapping).astype("int")
    pd.testing.assert_series_equal(
        df["Status"],
        ref_df["Step Type"],
        check_names=False,
        check_dtype=False,
    )

    # Cycle count
    if not all(df["Cycle"] == ref_df["Cycle Index"]):
        df_software_cycle = NewareNDA.read(nda_file, software_cycle_number=True)
        pd.testing.assert_series_equal(
            df_software_cycle["Cycle"],
            ref_df["Cycle Index"],
            check_names=False,
            check_dtype=False,
        )
        warnings.warn(
            f"{nda_file}: Cycle matches BTSDA reference only with software cycle number",
            stacklevel=2,
        )

    # Step count
    pd.testing.assert_series_equal(
        df["Step"],
        ref_df["Step Count"],
        check_names=False,
        check_dtype=False,
    )

    # Date - Neware usually timezone unaware, check offset is at least sensible
    start_test = df["Timestamp"].iloc[0]
    start_ref = ref_df["Date"].iloc[0]
    start_test = start_test.replace(tzinfo=None)
    start_ref = start_ref.replace(tzinfo=None)
    assert abs((start_test - start_ref).total_seconds()) < 24*60*60, "Date does not match"

    # Check that timestamps are equal ignoring the offset
    pd.testing.assert_series_equal(
        df["Timestamp"] - df["Timestamp"].iloc[0],
        ref_df["Date"] - ref_df["Date"].iloc[0],
        check_names=False,
        check_dtype=False,
    )

    # Capacity and energy
    # These are more complicated to check, Neware stores them in a few different ways
    # In NewareNDA there are two columns, charge and discharge capacity/energy, which are always positive
    # In BTSDA outputs there is one column, which can be negative, and can go up and down within one step
    # In some BTSDA outputs the capacity/energy does not count 'reverse' currents during charge or discharge

    abs_tol = 3e-4  # in mAh and mWh
    rel_tol = 1e-6

    # Capacity
    df["Capacity(mAh)"] = (df["Charge_Capacity(mAh)"] - df["Discharge_Capacity(mAh)"]).abs()
    abs_diff = df["Capacity(mAh)"] - ref_df["Capacity(mAs)"].abs()/3600
    rel_diff = abs_diff / ref_df["Capacity(mAs)"].abs()/3600
    if ((abs_diff > abs_tol) & (rel_diff > rel_tol)).any():
        # If this fails, sometimes Neware does not count negative current during charge towards the capacity
        cap_ignore_negs = (
            df.groupby("Step")["Capacity(mAh)"]
            .transform(lambda s: s.abs().cummax())
        )
        abs_diff = cap_ignore_negs - ref_df["Capacity(mAs)"].abs()/3600
        rel_diff = abs_diff / ref_df["Capacity(mAs)"].abs()/3600
        if ((abs_diff > abs_tol) & (rel_diff > rel_tol)).any():
            msg = "Capacity columns are different."
            raise ValueError(msg)

    # Energy
    df["Energy(mWh)"] = (df["Charge_Energy(mWh)"] - df["Discharge_Energy(mWh)"]).abs()
    abs_diff = df["Energy(mWh)"] - ref_df["Energy(mWs)"].abs()/3600
    rel_diff = abs_diff / ref_df["Energy(mWs)"].abs()/3600
    if ((abs_diff > abs_tol) & (rel_diff > rel_tol)).any():
        # If this fails, sometimes Neware does not count negative current during charge towards the energy
        cap_ignore_negs = (
            df.groupby("Step")["Energy(mWh)"]
            .transform(lambda s: s.abs().cummax())
        )
        abs_diff = cap_ignore_negs - ref_df["Energy(mWs)"].abs()/3600
        rel_diff = abs_diff / ref_df["Energy(mWs)"].abs()/3600
        if ((abs_diff > abs_tol) & (rel_diff > rel_tol)).any():
            msg = "Energy columns are different."
            raise ValueError(msg)

