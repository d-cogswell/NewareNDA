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
