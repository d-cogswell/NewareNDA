# © 2022-2024 Copyright SES AI
# Author: Daniel Cogswell
# Email: danielcogswell@ses.ai

import logging
import mmap
import os
import struct
from datetime import datetime, timezone

import numpy as np
import polars as pl
import tzlocal

from .dicts import multiplier_dict, pl_dtype_dict, rec_columns, state_dict
from .NewareNDAx import read_ndax
from .utils import _count_changes, _generate_cycle_number

logger = logging.getLogger('newarenda')


def read(file, software_cycle_number=True, cycle_mode='chg', log_level='INFO'):
    """
    Read electrochemical data from an Neware nda or ndax binary file.

    Args:
        file (str): Name of an .nda or .ndax file to read
        software_cycle_number (bool): Regenerate the cycle number to match
            Neware's "Charge First" circular statistic setting
        cycle_mode (str): Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.
        log_level (str): Sets the modules logging level. Default: 'INFO'
            Options: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
    """

    # Set up logging
    log_level = log_level.upper()
    if log_level in logging._nameToLevel.keys():
        logger.setLevel(log_level)
    else:
        logger.warning(f"Logging level '{log_level}' not supported; Defaulting to 'INFO'. "
                       f"Supported options are: {', '.join(logging._nameToLevel.keys())}")

    # Identify file type and process accordingly
    _, ext = os.path.splitext(file)
    if ext == '.nda':
        return read_nda(file, software_cycle_number, cycle_mode)
    elif ext == '.ndax':
        return read_ndax(file, software_cycle_number, cycle_mode)
    else:
        logger.error("File type not supported!")
        raise TypeError("File type not supported!")


def read_nda(file, software_cycle_number, cycle_mode='chg'):
    """
    Function read electrochemical data from a Neware nda binary file.

    Args:
        file (str): Name of a .nda file to read
        software_cycle_number (bool): Generate the cycle number field
            to match old versions of BTSDA
        cycle_mode (str): Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
    """
    with open(file, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm.read(6) != b'NEWARE':
            logger.error(f"{file} does not appear to be a Neware file.")
            raise ValueError(f"{file} does not appear to be a Neware file.")

        # Get the file version
        [nda_version] = struct.unpack('<B', mm[14:15])
        logger.warning(f"NDA version: {nda_version}")

        # Try to find server and client version info
        version_loc = mm.find(b'BTSServer')
        if version_loc != -1:
            mm.seek(version_loc)
            server = mm.read(50).strip(b'\x00').decode()
            logger.info(f"Server: {server}")
            mm.seek(50, 1)
            client = mm.read(50).strip(b'\x00').decode()
            logger.info(f"Client: {client}")
        else:
            logger.info("BTS version not found!")

        # version specific settings
        if nda_version == 29:
            data_df, aux_df = _read_nda_29(mm)
        elif nda_version == 130:
            if mm[1024:1025] == b'\x55':  # It is BTS 9.1
                data_df, aux_df = _read_nda_130_91(mm)
            else:
                data_df, aux_df = _read_nda_130_90(mm)
        else:
            logger.error(f"nda version {nda_version} is not yet supported!")
            raise NotImplementedError(f"nda version {nda_version} is not yet supported!")

    # Convert uts_s to Timestamp and replace Status ints with strings
    # Leave timezone localization to the end! Doing in polars then casting to pandas can cause kernel crashes
    try:
        tz = tzlocal.get_localzone_name()
    except Exception:
        logger.info("Could not get local timezone, using UTC.")
        tz = "UTC"
    data_df = data_df.with_columns([
        _count_changes(pl.col("Step")).alias("Step"),
        pl.col("Time").round(3),  # Round to nearest ms
        pl.from_epoch(pl.col("uts"), time_unit="s").alias("Timestamp") if "uts" in data_df.columns else pl.lit(None),
        pl.col("Status").replace_strict(state_dict, default=None).alias("Status"),
        pl.Series(name="Cycle", values=_generate_cycle_number(data_df, cycle_mode)) if software_cycle_number else pl.lit(None),
    ])

    data_df = data_df.select(rec_columns)

    # Drop duplicate indexes
    data_df = data_df.unique(subset="Index")

    # Join temperature data
    if not aux_df.is_empty():
        if "Aux" in aux_df.columns:
            aux_df = aux_df.unique(subset=["Index", "Aux"])
            aux_df = aux_df.pivot(index="Index", on="Aux", separator="")
        else:
            aux_df = aux_df.unique(subset=["Index"])
        data_df = data_df.join(aux_df, on="Index")

    data_df = data_df.cast(pl_dtype_dict)
    data_df = data_df.sort("Index")
    data_df = data_df.to_pandas()
    data_df["Timestamp"] = data_df["Timestamp"].dt.tz_localize(tz, ambiguous="infer")
    return data_df


def _read_nda_29(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 29, return data and aux DataFrames."""
    mm_size = mm.size()

    # Get the active mass
    [active_mass] = struct.unpack('<I', mm[152:156])
    logger.info(f"Active mass: {active_mass/1000} mg")

    try:
        remarks = mm[2317:2417].decode('ASCII')
        # Clean null characters
        remarks = remarks.replace(chr(0), '').strip()
        logger.info(f"Remarks: {remarks}")
    except UnicodeDecodeError:
        logger.warning("Converting remark bytes into ASCII failed")
        remarks = ""

    # Identify the beginning of the data section
    record_len = 86
    identifier = b'\x00\x00\x00\x00\x55\x00'
    header = mm.find(identifier)
    if header == -1:
        logger.error("File does not contain any valid records.")
        raise EOFError("File does not contain any valid records.")
    while (((mm[header + 4 + record_len] != 85)
            | (not _valid_record(mm[header+4:header+4+record_len])))
            if header + 4 + record_len < mm_size
            else False):
        header = mm.find(identifier, header + 4)
    mm.seek(header + 4)

    # Read data records
    num_records = (len(mm)-header-4) // record_len
    arr = np.frombuffer(mm[header + 4:], dtype=np.int8).reshape((num_records, record_len))
    # Remove rows where last 4 bytes are zero
    mask = (arr[:, 82:].view(np.int32) == 0).flatten()
    arr = arr[mask]

    # Split into two arrays, one for data and one for aux

    # Data array - first byte is \x55
    data_mask = arr[:, 0] == 85
    data_dtype = np.dtype([
        ("_pad1",  "V2"), # 0-1
        ("Index",  np.uint32), # 2-5
        ("Cycle",  np.uint32), # 6-9
        ("_pad2",   "V2"), # 10-11
        ("Status", np.uint8), # 12
        ("Step", np.uint8), # 13 (records jumps)
        ("Time",   np.uint64), # 14-21
        ("Voltage", np.int32), # 22-25
        ("Current(mA)", np.int32), # 26-29
        ("_pad3", "V8"), # 30-37
        ("Charge_Capacity(mAh)", np.int64),  # 38-45
        ("Discharge_Capacity(mAh)", np.int64),  # 46-53
        ("Charge_Energy(mWh)", np.int64),  # 54-61
        ("Discharge_Energy(mWh)", np.int64),  # 62-69
        ("Y", np.uint16), # 70-71
        ("M", np.uint8), # 72
        ("D", np.uint8), # 73
        ("h", np.uint8), # 74
        ("m", np.uint8), # 75
        ("s", np.uint8), # 76
        ("_pad4", "V1"), # 77
        ("Range", np.int32), # 78-81
        ("_pad5", "V4"), # 82-85
    ])
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]
    data_arr = arr[data_mask].view(data_dtype_no_pad).flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = data_df.with_columns([
        pl.col("Cycle") + 1,
        pl.col("Time").cast(pl.Float32) / 1000,
        pl.col("Voltage").cast(pl.Float32) / 10000,
        pl.col("Range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("Multiplier"),
        pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias("Timestamp"),
    ])
    data_df = data_df.with_columns([
        pl.col("Current(mA)") * pl.col("Multiplier"),
        (pl.col(
            ["Charge_Capacity(mAh)", "Discharge_Capacity(mAh)", "Charge_Energy(mWh)", "Discharge_Energy(mWh)"],
        ) * pl.col("Multiplier") / 3600).cast(pl.Float32),
    ])
    data_df = data_df.drop(["Y", "M", "D", "h", "m", "s"])

    # Aux array - first byte is \x65
    aux_mask = arr[:, 0] == 101
    aux_dtype = np.dtype([
        ("_pad1", "V1"), # 0
        ("Aux", np.uint8), # 1
        ("Index", np.uint32), # 2-5
        ("_pad2", "V16"), # 6-21
        ("V", np.int32), # 22-25
        ("_pad3", "V8"), # 26-33
        ("T", np.int16), # 34-35
        ("_pad4", "V50"), # 36-81
    ])
    aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
    aux_arr = arr[aux_mask].view(aux_dtype_no_pad).flatten()
    aux_df = pl.DataFrame(aux_arr)
    aux_df = aux_df.with_columns([
        pl.col("T").cast(pl.Float32) / 10,  # 0.1'C -> 'C
        pl.col("V").cast(pl.Float32) / 10000,  # 0.1 mV -> V
    ])

    return data_df, aux_df

def _read_nda_130_91(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 130 BTS9.1, return data and aux DataFrames."""
    record_len = mm.find(mm[1024:1026], 1026) - 1024  # Get record length
    _read_footer(mm)  # Log metadata
    num_records = (len(mm)-2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024:1024+num_records*record_len], dtype=np.int8).reshape((num_records, record_len))

    # In BTS9.1, data and aux are in the same rows
    mask = (arr[:, 0] == 85) & (arr[:, 8:12].view(np.uint32) != 0).flatten()
    dtype_list = [
        ("_pad1", "V2"),  # 0-1
        ("Step", np.uint8),  # 2
        ("Status", np.uint8),  # 3
        ("Cycle", np.uint32),  # 4-7
        ("Index", np.uint32),  # 8-11
        ("Time", np.uint32),  # 12-15
        ("Time_ns", np.uint32),  # 16-19
        ("Current(mA)", np.float32),  # 20-23
        ("Voltage", np.float32),  # 24-27
        ("Capacity", np.float32),  # 28-31
        ("Energy", np.float32),  # 32-35
        ("_pad3", "V8"),  # 36-43
        ("uts_s", np.uint32),  # 44-47
        ("uts_ns", np.uint32),  # 48-51
    ]
    if record_len > 52:
        dtype_list.append(("_pad4", f"V{record_len-52}"))
    data_dtype = np.dtype(dtype_list)
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]

    # Mask, view, flatten, recalculate some columns
    data_arr = arr[mask].view(data_dtype_no_pad)
    data_arr = data_arr.flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = data_df.with_columns([
        pl.col("Capacity").clip(lower_bound=0).alias("Charge_Capacity(mAh)") / 3600,
        pl.col("Capacity").clip(upper_bound=0).abs().alias("Discharge_Capacity(mAh)") / 3600,
        pl.col("Energy").clip(lower_bound=0).alias("Charge_Energy(mWh)") / 3600,
        pl.col("Energy").clip(upper_bound=0).abs().alias("Discharge_Energy(mWh)") / 3600,
        (pl.col("Time") + pl.col("Time_ns") / 1e9).cast(pl.Float32).alias("Time"),
        (pl.col("uts_s") + pl.col("uts_ns") / 1e9).alias("uts"),
    ])
    data_df = data_df.drop(["uts_s", "uts_ns", "Energy", "Capacity", "Time_ns"])

    # If the record length is 56, then there is an additional temperature column
    # Read into separate DataFrame and merge later for compatibility with other versions
    if record_len == 56:
        aux_dtype = np.dtype([
            ("_pad1", "V8"),  # 0-7
            ("Index", np.uint32),  # 8-11
            ("_pad2", "V40"),  # 12-51
            ("T1", np.float32),  # 52-55
        ])
        aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
        aux_arr = arr[mask].view(aux_dtype_no_pad)
        aux_arr = aux_arr.flatten()
        aux_df = pl.DataFrame(aux_arr).with_columns(
            pl.lit(None).cast(pl.Float32).alias("V1"),  # Empty column for regression compatibility
        )
    else:
        aux_df = pl.DataFrame()

    return data_df, aux_df

def _read_nda_130_90(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 130 BTS9.0, return data and aux DataFrames."""
    record_len = 88
    _read_footer(mm)  # Log metadata
    num_records = (len(mm)-2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024:1024+num_records*record_len], dtype=np.int8).reshape((num_records, record_len))

    # Data and aux stored in different rows
    data_mask = np.all(arr[:, :6] == arr[0, :6], axis=1).flatten()
    aux_mask = (arr[:, 1:5].view(np.int32) == 101).flatten()

    data_dtype = np.dtype([
        ("_pad1", "V9"),  # 0-8
        ("Step", np.uint8),  # 9
        ("Status", np.uint8),  # 10
        ("_pad2", "V5"),  # 11-15
        ("Index", np.uint32),  # 16-19
        ("_pad3", "V8"),  # 20-27
        ("Time", np.uint64),  # 28-35
        ("Voltage", np.float32),  # 36-39
        ("Current(mA)", np.float32),  # 40-43
        ("_pad4", "V8"),  # 44-51
        ("Charge_Capacity(mAh)", np.float32),  # 52-55
        ("Charge_Energy(mWh)", np.float32),  # 56-59
        ("Discharge_Capacity(mAh)", np.float32),  # 60-63
        ("Discharge_Energy(mWh)", np.float32),  # 64-67
        ("uts", np.uint64),  # 68-75
        ("_pad5", "V12"),  # 76-87
    ])
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]
    data_arr = arr[data_mask].view(data_dtype_no_pad)
    data_arr = data_arr.flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = data_df.with_columns([
        pl.col("uts").cast(pl.Float64) / 1e6,  # us -> s
        (pl.col("Time") / 1e6).cast(pl.Float32),  # us -> s
        pl.col(["Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
                "Charge_Energy(mWh)", "Discharge_Energy(mWh)"]) / 3600,
    ])

    aux_dtype = np.dtype([
        ("_pad1", "V5"),  # 0-4
        ("Aux", np.uint8),  # 5
        ("Index", np.uint32),  # 6-9
        ("_pad2", "V16"),  # 10-25
        ("V", np.int32),  # 26-29
        ("_pad3", "V8"),  # 30-37
        ("T", np.int16),  # 38-41
        ("_pad4", "V48"),  # 42-87
    ])
    aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
    aux_arr = arr[aux_mask].view(aux_dtype_no_pad)
    aux_arr = aux_arr.flatten()
    aux_df = pl.DataFrame(aux_arr)
    aux_df = aux_df.with_columns([
        pl.col("T").cast(pl.Float32) / 10,  # 0.1'C -> 'C
        pl.col("V").cast(pl.Float32) / 10000,  # 0.1 mV -> V
    ])

    return data_df, aux_df

def _read_footer(mm: mmap.mmap) -> None:
    # Identify footer
    footer = mm.rfind(b'\x06\x00\xf0\x1d\x81\x00\x03\x00\x61\x90\x71\x90\x02\x7f\xff\x00', 1024)
    if footer:
        mm.seek(footer+16)
        bytes = mm.read(499)

        # Get the active mass
        [active_mass] = struct.unpack('<d', bytes[-8:])
        logger.info(f"Active mass: {active_mass} mg")

        # Get the remarks
        remarks = bytes[363:491].decode('ASCII')

        # Clean null characters
        remarks = remarks.replace(chr(0), '').strip()
        logger.info(f"Remarks: {remarks}")

def _valid_record(bytes):
    """Helper function to identify a valid record"""
    # Check for a non-zero Status
    [Status] = struct.unpack('<B', bytes[12:13])
    return (Status != 0)
