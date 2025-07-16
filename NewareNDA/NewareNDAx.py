# © 2022-2024 Copyright SES AI
# Author: Daniel Cogswell
# Email: danielcogswell@ses.ai

import logging
import mmap
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from .dicts import (
    aux_chl_type_columns,
    dtype_dict,
    multiplier_dict,
    rec_columns,
    state_dict,
)
from .utils import _count_changes, _generate_cycle_number

logger = logging.getLogger('newarenda')


def read_ndax(file, software_cycle_number=False, cycle_mode='chg'):
    """
    Function to read electrochemical data from a Neware ndax binary file.

    Args:
        file (str): Name of an .ndax file to read
        software_cycle_number (bool): Regenerate the cycle number field
        cycle_mode (str): Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.PyZipFile(file)

        # Read version information
        try:
            version_info = zf.extract('VersionInfo.xml', path=tmpdir)
            with open(version_info, 'r', encoding='gb2312') as f:
                config = ET.fromstring(f.read()).find('config/ZwjVersion')
            logger.info(f"Server version: {config.attrib['SvrVer']}")
            logger.info(f"Client version: {config.attrib['CurrClientVer']}")
            logger.info(f"Control unit version: {config.attrib['ZwjVersion']}")
            logger.info(f"Tester version: {config.attrib['MainXwjVer']}")
        except Exception:
            pass

        # Read active mass
        try:
            step = zf.extract('Step.xml', path=tmpdir)
            with open(step, 'r', encoding='gb2312') as f:
                config = ET.fromstring(f.read()).find('config')
            active_mass = float(config.find('Head_Info/SCQ').attrib['Value'])
            logger.info(f"Active mass: {active_mass/1000} mg")
        except Exception:
            pass

        # Find all auxiliary channel files
        # Auxiliary files files need to be matched to entries in TestInfo.xml
        # Sort by the numbers in the filename, assume same order in TestInfo.xml
        aux_data = []
        for f in zf.namelist():
            m = re.search(r"data_AUX_(\d+)_(\d+)_(\d+)\.ndc", f)
            if m:
                aux_data.append((f, list(map(int, m.groups()))))
            else:
                m = re.search(r".*_(\d+)\.ndc", f)
                if m:
                    aux_data.append((f, [int(m.group(1)), 0, 0]))

        # Sort by the three integers
        aux_data.sort(key=lambda x: x[1])
        aux_filenames = [f for f, _ in aux_data]

        # Find all auxiliary channel dicts in TestInfo.xml
        aux_dicts = []
        if aux_filenames:
            step = zf.extract('TestInfo.xml', path=tmpdir)
            with open(step, 'r', encoding='gb2312') as f:
                config = ET.fromstring(f.read()).find('config')
            for child in config.find("TestInfo"):
                if "aux" in child.tag.lower():
                    aux_dicts.append({k: int(v) if v.isdigit() else v for k, v in child.attrib.items()})

        # ASSUME channel files are in the same order as TestInfo.xml, map filenames to dicts
        if len(aux_dicts) == len(aux_filenames):
            aux_ch_dict = {f: d for f, d in zip(aux_filenames, aux_dicts)}
        else:
            aux_ch_dict = {}
            logger.critical("Found a different number of aux channels in files and TestInfo.xml!")

        # Extract and parse all of the .ndc files into dataframes in parallel
        files_to_read = ["data.ndc", "data_runInfo.ndc", "data_step.ndc", *aux_filenames]
        dfs = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(extract_and_read_ndc, zf, fname, tmpdir): fname
                for fname in files_to_read
            }
            for future in as_completed(futures):
                fname, df = future.result()
                if df is not None:
                    dfs[fname] = df

        if "data.ndc" not in dfs:
            msg = "File type not yet supported!"
            raise NotImplementedError(msg)

        df = dfs["data.ndc"]

        if "data_runInfo.ndc" in dfs:
            df = df.join(dfs["data_runInfo.ndc"], how="left", on="Index")
        if "data_step.ndc" in dfs:
            df = df.with_columns([pl.col("Step").forward_fill()])
            df = df.join(dfs["data_step.ndc"], how="left", on="Step")

        # Interpolate missing data if necessary
        if df["Time"].is_null().any():
            df = _data_interpolation(df)

        # Column calculations in parallel:
        # round time to ms, Status -> categories, uts -> Timestamp, software cycle number
        cols = [
            pl.col("Time").round(3),
            pl.col("Status").replace_strict(state_dict, default=None).alias("Status"),
        ]
        if "uts" in df.columns:
            cols += [pl.from_epoch(pl.col("uts"), time_unit="s").alias("Timestamp")]
        if software_cycle_number:
            cols += [pl.Series(name="Cycle", values=_generate_cycle_number(df, cycle_mode))]

        df = df.with_columns(cols)

        # Keep only record columns
        df = df.select(rec_columns)
        df = df.cast(dtype_dict)

        # Merge the aux data if it exists
        for i, (f, aux_dict) in enumerate(aux_ch_dict.items()):
            if f not in dfs:
                continue
            else:
                aux_df = dfs[f]

            # Get aux ID, use -i if not present to avoid conflicts
            aux_id = aux_dict.get("AuxID", -i)

            # If ? column exists, rename name by ChlType (T, t, H)
            if "?" in aux_df.columns and aux_dict.get("ChlType") in aux_chl_type_columns:
                col = aux_chl_type_columns[aux_dict["ChlType"]]
                aux_df = aux_df.rename({"?": f"{col}{aux_id}"})
            else:  # Otherwise just append aux ID to column names
                aux_df = aux_df.rename({col: f"{col}{aux_id}" for col in aux_df.columns if col not in ["Index"]})
            df = df.join(aux_df, how="left", on="Index")

    # Convert to pandas, change timestamp to local timezone
    df = df.to_pandas()
    tz = datetime.now().astimezone().tzinfo
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    return df

def extract_and_read_ndc(zf: zipfile.ZipFile, filename: str, tmpdir: str) -> tuple[str, pd.DataFrame | None]:
    """Extract .ndc from a zipfile and reads it into a DataFrame."""
    if filename in zf.namelist():
        file_path = zf.extract(filename, path=tmpdir)
        return filename, read_ndc(file_path)
    return filename, None

def _data_interpolation(df):
    """
    Some ndax from from BTS Server 8 do not seem to contain a complete dataset.
    This helper function fills in missing times, capacities, and energies.
    """
    # Get time by forward filling differences
    df = df.with_columns([
        pl.col("Time").is_not_null().alias("nan_mask"),
        pl.col("Time").is_not_null().cum_sum().shift(1).fill_null(0).alias("group_idx"),
        pl.col("dt", "Time", "uts", "Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
           "Charge_Energy(mWh)", "Discharge_Energy(mWh)").fill_null(strategy="forward"),
    ])

    df = df.with_columns([
        (pl.col("dt").cum_sum().over("group_idx") * (~pl.col("nan_mask"))).alias("cdt"),
        (
            (pl.col("dt") * pl.col("Current(mA)") / 3600)
            .cum_sum().over("group_idx") * ~pl.col("nan_mask")
        ).alias("inc_capacity"),
        (
            (pl.col("dt") * pl.col("Voltage") * pl.col("Current(mA)") / 3600)
            .cum_sum().over("group_idx") * ~pl.col("nan_mask")
        ).alias("inc_energy"),
    ])

    df = df.with_columns([
        (pl.col("Time") + pl.col("cdt")).alias("Time"),
        (pl.col("uts") + pl.col("cdt")).alias("uts"),
        (
            pl.col("Charge_Capacity(mAh)").abs() + pl.col("inc_capacity").clip(lower_bound=0).abs()
        ).alias("Charge_Capacity(mAh)"),
        (
            pl.col("Discharge_Capacity(mAh)").abs() + pl.col("inc_capacity").clip(upper_bound=0).abs()
        ).alias("Discharge_Capacity(mAh)"),
        (
            pl.col("Charge_Energy(mWh)").abs() + pl.col("inc_energy").clip(lower_bound=0).abs()
        ).alias("Charge_Energy(mWh)"),
        (
            pl.col("Discharge_Energy(mWh)").abs() + pl.col("inc_energy").clip(upper_bound=0).abs()
        ).alias("Discharge_Energy(mWh)"),
    ])

    # Sanity checks
    if (df["uts"].diff() < 0).any():
        logger.warning(
            "IMPORTANT: This ndax has negative jumps in the 'Timestamp' column! "
            "This can sometimes happen in the ndax file itself. "
            "Use the 'Time' column for analysis.",
        )

    return df


def read_ndc(file):
    """
    Function to read electrochemical data from a Neware ndc binary file.

    Args:
        file (str): Name of an .ndc file to read
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
        aux_df (pd.DataFrame): DataFrame containing any temperature data
    """
    with open(file, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Get ndc file version and filetype
        [ndc_filetype] = struct.unpack("<B", mm[0:1])
        [ndc_version] = struct.unpack("<B", mm[2:3])
        logger.debug("NDC version: %d filetype: %d", ndc_version, ndc_filetype)
        try:
            func = getattr(sys.modules[__name__], f"_read_ndc_{ndc_version}_filetype_{ndc_filetype}")
            return func(mm)
        except AttributeError:
            raise NotImplementedError(f"ndc version {ndc_version} filetype {ndc_filetype} is not yet supported!")


def _read_ndc_2_filetype_1(mm):
    dtype = np.dtype([  # 0
        ("_pad1",  "V8"),  # 0-7
        ("Index",  np.uint32),  # 8-11
        ("Cycle",  np.uint32),  # 12-15
        ("Step", np.uint8),  # 16
        ("Status", np.uint8),  # 17
        ("_pad2",  "V5"),  # 18-22
        ("Time",   np.uint64), # 23-30
        ("Voltage", np.int32), # 31-34
        ("Current(mA)", np.int32), # 35-38
        ("_pad3", "V4"), # 39-42
        ("Charge_Capacity(mAh)", np.int64),  # 43-50
        ("Discharge_Capacity(mAh)", np.int64),  # 51-58
        ("Charge_Energy(mWh)", np.int64),  # 59-66
        ("Discharge_Energy(mWh)", np.int64),  # 67-74
        ("Y", np.uint16), # 75-76
        ("M", np.uint8), # 77
        ("D", np.uint8), # 78
        ("h", np.uint8), # 79
        ("m", np.uint8), # 80
        ("s", np.uint8), # 81
        ("Range", np.int32), # 82-85
        ("_pad4", "V8"), # 86-93
    ])
    df = _read_ndc(mm, dtype, 5, 37, record_size = 512, file_header_size = 512).with_columns([
        pl.col("Cycle") + 1,
        pl.col("Time").cast(pl.Float32) * 1e-3,
        pl.col("Voltage").cast(pl.Float32) * 1e-4,
        pl.col("Range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("Multiplier"),
        pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias("Timestamp"),
    ])
    df = df.with_columns([
        pl.col("Current(mA)") * pl.col("Multiplier"),
        (pl.col(
            ["Charge_Capacity(mAh)", "Discharge_Capacity(mAh)", "Charge_Energy(mWh)", "Discharge_Energy(mWh)"],
        ) * pl.col("Multiplier") / 3600).cast(pl.Float32),
    ])
    return df.drop(["Y", "M", "D", "h", "m", "s"])


def _read_ndc_2_filetype_5(mm):
    # This dtype is missing humudity % column - does not exist in current test data
    dtype = np.dtype([
        ("_pad2",  "V8"),  # 4-7
        ("Index",  np.uint32), # 8-11
        ("_pad3", "V19"),  # 12 - 30
        ("V", np.int32),  # 31-34
        ("_pad4", "V6"), # 35-40
        ("T", np.int16),  # 41-42
        ("t", np.int16),  # 43-44
        ("_pad5", "V49"),  # 45-93
    ])
    return _read_ndc(mm, dtype, 5, 37, record_size = 512, file_header_size = 512).with_columns(
        pl.col("V").cast(pl.Float32) / 10000,
        pl.col("T").cast(pl.Float32) * 0.1,
        pl.col("t").cast(pl.Float32) * 0.1,
    )


def _read_ndc_5_filetype_1(mm):
    dtype = np.dtype([  # 0
        ("_pad1",  "V8"),  # 0-7
        ("Index",  np.uint32),  # 8-11
        ("Cycle",  np.uint32),  # 12-15
        ("Step", np.uint8),  # 16
        ("Status", np.uint8),  # 17
        ("_pad2",  "V5"),  # 18-22
        ("Time",   np.uint64), # 23-30
        ("Voltage", np.int32), # 31-34
        ("Current(mA)", np.int32), # 35-38
        ("_pad3", "V4"), # 39-42
        ("Charge_Capacity(mAh)", np.int64),  # 43-50
        ("Discharge_Capacity(mAh)", np.int64),  # 51-58
        ("Charge_Energy(mWh)", np.int64),  # 59-66
        ("Discharge_Energy(mWh)", np.int64),  # 67-74
        ("Y", np.uint16), # 75-76
        ("M", np.uint8), # 77
        ("D", np.uint8), # 78
        ("h", np.uint8), # 79
        ("m", np.uint8), # 80
        ("s", np.uint8), # 81
        ("Range", np.int32), # 82-85
        ("_pad4", "V1"), # 86
    ])
    df = _read_ndc(mm, dtype, 125, 56).with_columns([
        pl.col("Cycle") + 1,
        pl.col("Time").cast(pl.Float32) * 1e-3,
        pl.col("Voltage").cast(pl.Float32) * 1e-4,
        pl.col("Range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("Multiplier"),
        pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias("Timestamp"),
    ])
    df = df.with_columns([
        pl.col("Current(mA)") * pl.col("Multiplier"),
        (pl.col(
            ["Charge_Capacity(mAh)", "Discharge_Capacity(mAh)", "Charge_Energy(mWh)", "Discharge_Energy(mWh)"],
        ) * pl.col("Multiplier") / 3600).cast(pl.Float32),
    ])
    return df.drop(["Y", "M", "D", "h", "m", "s"])


def _read_ndc_5_filetype_5(mm):
    dtype = np.dtype([
        ("_pad2",  "V8"),  # 4-7
        ("Index",  np.uint32), # 8-11
        ("_pad3", "V19"),  # 12 - 30
        ("V", np.int32),  # 31-34
        ("_pad4", "V6"), # 35-40
        ("T", np.int16),  # 41-42
        ("t", np.int16),  # 43-44
        ("_pad5", "V42"),  # 45-86
    ])
    return _read_ndc(mm, dtype, 125, 56).with_columns(
        pl.col("V").cast(pl.Float32) * 1e-4,
        pl.col("T").cast(pl.Float32) * 0.1,
        pl.col("t").cast(pl.Float32) * 0.1,
    )


def _read_ndc_11_filetype_1(mm):
    dtype = np.dtype([
        ("Voltage", "<f4"),
        ("Current(mA)", "<f4"),
    ])
    return _read_ndc(mm, dtype, 132, 4).with_columns([
        pl.col("Voltage") * 1e-4,  # 0.1mV -> V
    ])


def _read_ndc_11_filetype_5(mm):
    header = 4096

    if mm[header+132:header+133] == b"\x65":
        dtype = np.dtype([
            ("mask", "<i1"),
            ("V", "<f4"),
            ("T", "<i2"),
        ])
        return _read_ndc(mm, dtype, 132, 2, mask=101).with_columns([
            pl.col("V") * 1e-4,  # 0.1 mV -> V
            pl.col("T").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Index"),
        ])

    if mm[header+132:header+133] == b"\x74":
        dtype = np.dtype([
            ("_pad1", "V1"),
            ("Index", "<i4"),
            ("Aux", "<i1"),
            ("_pad2", "V29"),
            ("T", "<i2"),
            ("_pad3", "V51"),
        ])
        return _read_ndc(mm, dtype, 132, 4).with_columns([
            pl.col("T").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
        ]).drop("Aux")  # Aux channel inferred from TestInfo.xml

    msg = "Unknown file structure for ndc version 11 filetype 5."
    raise NotImplementedError(msg)


def _read_ndc_11_filetype_7(mm):
    dtype = np.dtype([
        ("Cycle", "<i4"),
        ("Step_Index",  "<i4"),
        ("_pad1", "V16"),
        ("Status", "<i1"),
        ("_pad2", "V12"),
    ])
    return _read_ndc(mm, dtype, 132, 5).with_columns([
        pl.col("Cycle") + 1,
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Step"),
    ])


def _read_ndc_11_filetype_18(mm):
    dtype = np.dtype([
        ("Time", "<i4"),
        ("_pad1",  "V1"),
        ("Charge_Capacity(mAh)", "<f4"),
        ("Discharge_Capacity(mAh)", "<f4"),
        ("Charge_Energy(mWh)", "<f4"),
        ("Discharge_Energy(mWh)", "<f4"),
        ("_pad2",  "V8"),
        ("dt", "<i4"),
        ("uts_s", "<i4"),
        ("Step", "<i4"),
        ("Index", "<i4"),
        ("uts_ms", "<i2"),
    ])
    return _read_ndc(mm, dtype, 132, 16).with_columns([
        pl.col("Time", "dt").cast(pl.Float32) / 1000,  # Division in 32-bit
        pl.col("Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
            "Charge_Energy(mWh)", "Discharge_Energy(mWh)") / 3600, # mAs|mWs -> mAh|mWh
        (pl.col("uts_s") + pl.col("uts_ms") / 1000).alias("uts"),
        _count_changes(pl.col("Step")).alias("Step"),
    ])


def _read_ndc_14_filetype_1(mm):
    dtype = np.dtype([
        ("Voltage", "<f4"),
        ("Current(mA)", "<f4"),
    ])
    return _read_ndc(mm, dtype, 132, 4).with_columns([
        pl.col("Current(mA)") * 1000,
    ])


def _read_ndc_14_filetype_5(mm):
    dtype = np.dtype([
        ("?", "<f4"),  # Column name is assigned later from TestInfo.xml
    ])
    return _read_ndc(mm, dtype, 132, 4).with_columns([
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Index"),
    ])


def _read_ndc_14_filetype_7(mm):
    dtype = np.dtype([
        ("Cycle", "<i4"),
        ("Step_Index", "<i4"),
        ("_pad1", "V16"),
        ("Status", "<i1"),
        ("_pad2", "V12"),
    ])
    return _read_ndc(mm, dtype, 132, 5).with_columns([
        pl.col("Cycle") + 1,
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Step"),
    ])


def _read_ndc_14_filetype_18(mm):
    dtype = np.dtype([
        ("Time", "<i4"),
        ("_pad1",  "V1"),
        ("Charge_Capacity(mAh)", "<f4"),
        ("Discharge_Capacity(mAh)", "<f4"),
        ("Charge_Energy(mWh)", "<f4"),
        ("Discharge_Energy(mWh)", "<f4"),
        ("_pad2",  "V8"),
        ("dt", "<i4"),
        ("uts_s", "<i4"),
        ("Step", "<i4"),
        ("Index", "<i4"),
        ("uts_ms", "<i2"),
        ("_pad3",  "V8"),
    ])
    return _read_ndc(mm, dtype, 132, 4).with_columns([
        pl.col("Time", "dt").cast(pl.Float32) / 1000,  # ms -> s
        pl.col("Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
            "Charge_Energy(mWh)", "Discharge_Energy(mWh)") * 1000,  # Ah|Wh -> mAh|mWh
        (pl.col("uts_s") + pl.col("uts_ms") / 1000).alias("uts"),
        pl.col("Step").diff().fill_null(1).abs().gt(0).cum_sum().alias("Step"),
    ])


def _read_ndc_17_filetype_1(mm):
    return _read_ndc_14_filetype_1(mm)


def _read_ndc_17_filetype_7(mm):
    dtype = np.dtype([
        ("Cycle", "<i4"),
        ("Step", "<i4"),
        ("_pad1", "V16"),
        ("Status", "<i1"),
        ("_pad2", "V8"),
        ("Step_Index", "<i4"),
        ("_pad3", "V63"),
    ])
    return _read_ndc(mm, dtype, 132, 64).with_columns([
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Cycle"),
        _count_changes(pl.col("Step")).alias("Step"),
    ])


def _read_ndc_17_filetype_18(mm):
    dtype = np.dtype([
        ("Time", "<i4"),
        ("_pad1",  "V1"),
        ("Charge_Capacity(mAh)", "<f4"),
        ("Discharge_Capacity(mAh)", "<f4"),
        ("Charge_Energy(mWh)", "<f4"),
        ("Discharge_Energy(mWh)", "<f4"),
        ("_pad2",  "V8"),
        ("dt", "<i4"),
        ("uts_s", "<i4"),
        ("Step", "<i4"),
        ("Index", "<i4"),
        ("uts_ms", "<i2"),
        ("_pad3",  "V53"),
    ])
    return _read_ndc(mm,dtype, 132, 64).with_columns([
        pl.col("Time", "dt").cast(pl.Float32) / 1000,
        (pl.col("Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
            "Charge_Energy(mWh)", "Discharge_Energy(mWh)") * 1000).cast(pl.Float32),  # Ah|Wh -> mAh|mWh
        (pl.col("uts_s") + pl.col("uts_ms") / 1000).alias("uts"),
    ])

def _read_ndc(
    mm: mmap.mmap,
    dtype: np.dtype,
    record_header_size: int,
    record_footer_size: int,
    record_size: int = 4096,
    file_header_size: int = 4096,
    mask: int | None = None,
):
    """Read ndc file into a polars DataFrame.

    Args:
        mm (mmap.mmap): Memory-mapped file object.
        dtype (np.dtype): Numpy dtype describing the record structure.
        record_header_size (int): Size of the record header in bytes.
        record_footer_size (int): Size of the record footer in bytes.
        record_size (int): Total size of a single record in bytes.
        file_header_size (int): Size of the file header in bytes.
        mask (int | None): Optional mask to filter, assumes a column named
            "mask" and keeps rows where "mask" equals this value.

    Returns:
        pl.DataFrame: Polars DataFrame containing the records.

    """
    # Read entire file into 1 byte array nrecords x record_size
    num_records = (len(mm)-file_header_size) // record_size
    arr = np.frombuffer(mm[file_header_size:], dtype=np.int8).reshape((num_records, record_size))
    # Slice the header and footer
    arr = arr[:, record_header_size:-record_footer_size]
    # Remove padding columns
    useful_cols = [name for name in dtype.names if not name.startswith("_")]
    dtype_no_pad = dtype[useful_cols]
    arr = arr.view(dtype=dtype_no_pad)
    # Flatten
    arr = arr.reshape(-1)

    # If a mask is provided, filter the array
    if mask is not None and "mask" in arr.dtype.names:
        arr = arr[arr["mask"] == mask]
        return pl.DataFrame(arr).drop("mask")

    # If runInfo file, remove 0 index rows
    if "Index" in arr.dtype.names:
        arr = arr[arr["Index"] != 0]
        return pl.DataFrame(arr)

    # If step file, remove 0 step index rows
    if "Step_Index" in arr.dtype.names:
        arr = arr[arr["Step_Index"] != 0]
        return pl.DataFrame(arr)

    # If data file, remove 0.0 voltage rows and add Index column
    if "Voltage" in arr.dtype.names:
        arr = arr[arr["Voltage"] != 0]
        return pl.DataFrame(arr).with_columns([
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("Index"),
        ])

    return pl.DataFrame(arr)
