# © 2022-2024 Copyright SES AI
# Author: Daniel Cogswell
# Email: danielcogswell@ses.ai

import sys
import mmap
import struct
import logging
import tempfile
import zipfile
import re
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd

from .utils import _generate_cycle_number, _count_changes
from .dicts import rec_columns, dtype_dict, aux_dtype_dict, state_dict, \
    multiplier_dict

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

        # Read aux channel mapping from TestInfo.xml
        aux_ch_dict = {}
        try:
            step = zf.extract('TestInfo.xml', path=tmpdir)
            with open(step, 'r', encoding='gb2312') as f:
                config = ET.fromstring(f.read()).find('config')

            for child in config.find("TestInfo"):
                aux_ch_dict.update({int(child.attrib['RealChlID']): int(child.attrib['AuxID'])})

        except Exception:
            pass

        # Try to read data.ndc
        if 'data.ndc' in zf.namelist():
            data_file = zf.extract('data.ndc', path=tmpdir)
            data_df = read_ndc(data_file)
        else:
            raise NotImplementedError("File type not yet supported!")

        # Some ndax have data spread across 3 different ndc files. Others have
        # all data in data.ndc.
        # Check if data_runInfo.ndc and data_step.ndc exist
        if all(i in zf.namelist() for i in ['data_runInfo.ndc', 'data_step.ndc']):

            # Read data from separate files
            runInfo_file = zf.extract('data_runInfo.ndc', path=tmpdir)
            step_file = zf.extract('data_step.ndc', path=tmpdir)
            runInfo_df = read_ndc(runInfo_file)
            step_df = read_ndc(step_file)

            # Merge dataframes
            data_df = data_df.merge(runInfo_df, how='left', on='Index')
            data_df['Step'] = data_df['Step'].ffill()
            data_df = data_df.merge(step_df, how='left', on='Step')

            # Fill in missing data - Neware appears to fabricate data
            if data_df.isnull().any(axis=None):
                _data_interpolation(data_df)

            # Convert uts_s to timezone aware Timestamp
            data_df["Timestamp"] = pd.to_datetime(data_df["uts"], unit="s", utc=True)
            tz = datetime.now().astimezone().tzinfo
            data_df["Timestamp"] = data_df["Timestamp"].dt.tz_convert(tz)

            # Keep only record columns
            data_df = data_df.reindex(columns=rec_columns)


        # Read and merge Aux data from ndc files
        aux_df = pd.DataFrame([])
        for f in zf.namelist():

            # If the filename contains a channel number, convert to aux_id
            m = re.search("data_AUX_([0-9]+)_[0-9]+_[0-9]+[.]ndc", f)
            if m:
                ch = int(m[1])
                aux_id = aux_ch_dict[ch]
            else:
                m = re.search(".*_([0-9]+)[.]ndc", f)
                if m:
                    aux_id = int(m[1])

            if m:
                aux_file = zf.extract(f, path=tmpdir)
                aux = read_ndc(aux_file)
                aux['Aux'] = aux_id
                aux_df = pd.concat([aux_df, aux], ignore_index=True)
        if not aux_df.empty:
            aux_df = aux_df.astype(
                {k: aux_dtype_dict[k] for k in aux_dtype_dict.keys() & aux_df.columns})
            pvt_df = aux_df.pivot(index='Index', columns='Aux')
            pvt_df.columns = pvt_df.columns.map(lambda x: ''.join(map(str, x)))
            data_df = data_df.join(pvt_df, on='Index')

    if software_cycle_number:
        data_df['Cycle'] = _generate_cycle_number(data_df, cycle_mode)

    # Convert statuses to strings
    if "Status" in data_df.columns:
        data_df["Status"] = data_df["Status"].map(state_dict)

    return data_df.astype(dtype=dtype_dict)


def _data_interpolation(df):
    """
    Some ndax from from BTS Server 8 do not seem to contain a complete dataset.
    This helper function fills in missing times, capacities, and energies.
    """

    # Rows missing from runinfo
    nan_mask = df["Time"].notna()

    # Get time by forward filling differences
    df["dt"] = df["dt"].ffill()
    cdt = df["dt"].ffill().groupby(nan_mask.cumsum().shift()).cumsum() * ~nan_mask
    df["Time"] = df["Time"].ffill() + cdt
    df["uts"] = df["uts"].ffill() + cdt

    # Integrate to get capacity and fill missing values
    capacity = df['dt']*abs(df['Current(mA)'])/3600
    inc = capacity.groupby(nan_mask.cumsum()).cumsum()
    chg = df['Charge_Capacity(mAh)'].ffill() + \
        inc.where(df['Current(mA)'] > 0, 0).shift()
    dch = df['Discharge_Capacity(mAh)'].ffill() + \
        inc.where(df['Current(mA)'] < 0, 0).shift()
    df['Charge_Capacity(mAh)'] = df['Charge_Capacity(mAh)'].where(nan_mask, chg)
    df['Discharge_Capacity(mAh)'] = df['Discharge_Capacity(mAh)'].where(nan_mask, dch)

    # Integrate to get energy and fill missing values
    energy = capacity*df['Voltage']
    inc = energy.groupby(nan_mask.cumsum()).cumsum()
    chg = df['Charge_Energy(mWh)'].ffill() + \
        inc.where(df['Current(mA)'] > 0, 0).shift()
    dch = df['Discharge_Energy(mWh)'].ffill() + \
        inc.where(df['Current(mA)'] < 0, 0).shift()
    df['Charge_Energy(mWh)'] = df['Charge_Energy(mWh)'].where(nan_mask, chg)
    df['Discharge_Energy(mWh)'] = df['Discharge_Energy(mWh)'].where(nan_mask, dch)

    # Sanity checks
    if any(df["uts"].diff() < 0):
        logger.warning(
            "IMPORTANT: This ndax has negative jumps in the 'Timestamp' column! "
            "This can sometimes happen in the ndax file itself. "
            "Use the 'Time' column for analysis.",
        )
    if any(df.groupby("Step")["Time"].diff() < 0):
        logger.warning(
            "IMPORTANT: This ndax has negative jumps in the Time column! "
            "There may be issues with read_ndc for this file type.",
        )


def read_ndc(file):
    """
    Function to read electrochemical data from a Neware ndc binary file.

    Args:
        file (str): Name of an .ndc file to read
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
        aux_df (pd.DataFrame): DataFrame containing any temperature data
    """
    with open(file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

            # Get ndc file version and filetype
            [ndc_filetype] = struct.unpack('<B', mm[0:1])
            [ndc_version] = struct.unpack('<B', mm[2:3])

            try:
                f = getattr(sys.modules[__name__], f"_read_ndc_{ndc_version}_filetype_{ndc_filetype}")
                return f(mm)
            except AttributeError:
                raise NotImplementedError(f"ndc version {ndc_version} filetype {ndc_filetype} is not yet supported!")


def _read_ndc_2_filetype_1(mm):
    record_len = 94
    identifier = mm[517:525]

    # Read data records
    output = []
    header = mm.find(identifier)
    while header != -1:
        mm.seek(header)
        bytes = mm.read(record_len)
        if bytes[0:1] == b'\x55':
            output.append(_bytes_to_list_ndc(bytes))
        else:
            logger.warning("Unknown record type: "+bytes[0:1].hex())

        header = mm.find(identifier, header + record_len)

    # Postprocessing
    df = pd.DataFrame(output, columns=rec_columns)

    return df


def _read_ndc_2_filetype_5(mm):
    record_len = 94
    identifier = mm[517:525]

    # Read aux records
    aux = []
    header = mm.find(identifier)
    while header != -1:
        mm.seek(header)
        bytes = mm.read(record_len)
        if bytes[0:1] == b'\x65':
            aux.append(_aux_bytes_65_to_list_ndc(bytes))
        elif bytes[0:1] == b'\x74':
            aux.append(_aux_bytes_74_to_list_ndc(bytes))
        else:
            logger.warning("Unknown record type: "+bytes[0:1].hex())

        header = mm.find(identifier, header + record_len)

    # Postprocessing
    aux_df = pd.DataFrame([])
    if identifier[0:1] == b'\x65':
        aux_df = pd.DataFrame(aux, columns=['Index', 'Aux', 'V', 'T'])
    elif identifier[0:1] == b'\x74':
        aux_df = pd.DataFrame(aux, columns=['Index', 'Aux', 'V', 'T', 't'])

    return aux_df


def _read_ndc_5_filetype_1(mm):
    mm_size = mm.size()
    record_len = 4096
    header = 4096

    # Read data records
    output = []
    mm.seek(header)
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        for i in struct.iter_unpack('<87s', bytes[125:-56]):
            if i[0][7:8] == b'\x55':
                output.append(_bytes_to_list_ndc(i[0]))

    # Postprocessing
    df = pd.DataFrame(output, columns=rec_columns)

    return df


def _read_ndc_5_filetype_5(mm):
    mm_size = mm.size()
    record_len = 4096
    header = 4096

    # Read aux records
    aux65 = []
    aux74 = []
    mm.seek(header)
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        for i in struct.iter_unpack('<87s', bytes[125:-56]):
            if i[0][7:8] == b'\x65':
                aux65.append(_aux_bytes_65_to_list_ndc(i[0]))
            elif i[0][7:8] == b'\x74':
                aux74.append(_aux_bytes_74_to_list_ndc(i[0]))

    # Concat aux65 and aux74 if they both contain data
    aux_df = pd.DataFrame(aux65, columns=['Index', 'Aux', 'V', 'T'])
    aux74_df = pd.DataFrame(aux74, columns=['Index', 'Aux', 'V', 'T', 't'])
    if (not aux_df.empty) & (not aux74_df.empty):
        aux_df = pd.concat([aux_df, aux74_df.drop(columns=['t'])])
    elif (not aux74_df.empty):
        aux_df = aux74_df

    return aux_df


def _read_ndc_11_filetype_1(mm):
    dtype = np.dtype([
        ("Voltage", "<f4"),
        ("Current(mA)", "<f4"),
    ])
    df = _read_ndc(mm, dtype, 132, 4)
    df["Voltage"] *= 1e-4  # 0.1mV -> V
    return df


def _read_ndc_11_filetype_5(mm):
    header = 4096
    if mm[header+132:header+133] == b'\x65':
        dtype = np.dtype([
            ("_pad1", "V1"),
            ("V", "<f4"),
            ("T", "<i2"),
        ])
        df = _read_ndc(mm, dtype, 132, 2)
        df["V"] *= 1e-4  # 0.1mV -> V
        df["T"] *= 0.1 # 0.1'C -> 'C

    elif mm[header+132:header+133] == b'\x74':
        dtype = np.dtype([
            ("_pad1", "V1"),
            ("Index", "<i4"),
            ("Aux", "<i1"),
            ("_pad2", "V29"),
            ("T", "<i2"),
            ("_pad3", "V51"),
        ])
        df = _read_ndc(mm, dtype, 132, 4)
        df["T"] *= 0.1  # 0.1'C -> 'C

    else:
        msg = "Unknown file structure for ndc version 11 filetype 5."
        raise NotImplementedError(msg)

    return df


def _read_ndc_11_filetype_7(mm):
    dtype = np.dtype([
        ("Cycle", "<i4"),
        ("Step_Index",  "<i4"),
        ("_pad1", "V16"),
        ("Status", "<i1"),
        ("_pad2", "V12"),
    ])
    df = _read_ndc(mm, dtype, 132, 5)
    df["Cycle"] = df["Cycle"] + 1
    df["Step"] = (df.index + 1).astype(np.int32)
    return df


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
    df = _read_ndc(mm, dtype, 132, 63)
    df["Time"] /= 1000  # ms -> s
    df["dt"] /= 1000  # ms -> s
    df["Charge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Discharge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Charge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["Discharge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["uts"] = df["uts_s"] + df["uts_ms"] / 1000 # s
    df['Step'] = _count_changes(df['Step'])
    return df


def _read_ndc_14_filetype_1(mm):
    dtype = np.dtype([
        ("Voltage", "<f4"),
        ("Current(mA)", "<f4"),
    ])
    df = _read_ndc(mm, dtype, 132, 4)
    df["Current(mA)"] *= 1000  # A -> mA
    return df

def _read_ndc_14_filetype_5(mm):
    dtype = np.dtype([
        ("T", "<f4"),
    ])
    return _read_ndc(mm, dtype, 132, 4)


def _read_ndc_14_filetype_7(mm):
    dtype = np.dtype([
        ("Cycle", "<i4"),
        ("Step_Index", "<i4"),
        ("_pad1", "V16"),
        ("Status", "<i1"),
        ("_pad2", "V12"),
    ])
    df = _read_ndc(mm, dtype, 132, 5)
    df["Step"] = (df.index + 1).astype(np.int32)
    return df


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
    df = _read_ndc(mm,dtype, 132, 4)
    df["Time"] /= 1000  # ms -> s
    df["dt"] /= 1000  # ms -> s
    df["Charge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Discharge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Charge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["Discharge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["uts"] = df["uts_s"] + df["uts_ms"] / 1000 # s
    df['Step'] = _count_changes(df['Step'])
    return df


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
    df = _read_ndc(mm, dtype, 132, 64)
    df["Cycle"] = df["Cycle"] + 1
    df["Step"] = _count_changes(df["Step"])
    return df

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
    df = _read_ndc(mm,dtype, 132, 64)
    df["Time"] /= 1000  # ms -> s
    df["dt"] /= 1000  # ms -> s
    df["Charge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Discharge_Capacity(mAh)"] *= 1000  # Ah -> mAh
    df["Charge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["Discharge_Energy(mWh)"] *= 1000  # Wh -> mWh
    df["uts"] = df["uts_s"] + df["uts_ms"] / 1000 # s
    return df

def _read_ndc(
    mm: mmap.mmap,
    dtype: np.dtype,
    record_header_size: int,
    record_footer_size: int,
    record_size: int = 4096,
    file_header_size: int = 4096,
):
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
    if "Index" in arr.dtype.names:  # Remove 0 index rows for runInfo/step files
        arr = arr[arr["Index"] != 0]
        df = pd.DataFrame(arr)
    elif "Step_Index" in arr.dtype.names:  # Remove 0 step rows for step files
        arr = arr[arr["Step_Index"] != 0]
        df = pd.DataFrame(arr)
    else:  # Add index column for data files
        df = pd.DataFrame(arr)
        df["Index"] = (np.arange(len(df)) + 1).astype('int32')
    return df


def _bytes_to_list_ndc(bytes):
    """Helper function for interpreting an ndc byte string"""

    # Extract fields from byte string
    [Index, Cycle, Step, Status] = struct.unpack('<IIBB', bytes[8:18])
    [Time, Voltage, Current] = struct.unpack('<Qii', bytes[23:39])
    [Charge_capacity, Discharge_capacity,
     Charge_energy, Discharge_energy] = struct.unpack('<qqqq', bytes[43:75])
    [Y, M, D, h, m, s] = struct.unpack('<HBBBBB', bytes[75:82])
    [Range] = struct.unpack('<i', bytes[82:86])

    multiplier = multiplier_dict[Range]

    # Create a record
    list = [
        Index,
        Cycle + 1,
        Step,
        state_dict[Status],
        Time/1000,
        Voltage/10000,
        Current*multiplier,
        Charge_capacity*multiplier/3600,
        Discharge_capacity*multiplier/3600,
        Charge_energy*multiplier/3600,
        Discharge_energy*multiplier/3600,
        datetime(Y, M, D, h, m, s)
    ]
    return list


def _aux_bytes_65_to_list_ndc(bytes):
    """Helper function for intepreting auxiliary records"""
    [Aux] = struct.unpack('<B', bytes[3:4])
    [Index] = struct.unpack('<I', bytes[8:12])
    [T] = struct.unpack('<h', bytes[41:43])
    [V] = struct.unpack('<i', bytes[31:35])

    return [Index, Aux, V/10000, T/10]


def _aux_bytes_74_to_list_ndc(bytes):
    """Helper function for intepreting auxiliary records"""
    [Aux] = struct.unpack('<B', bytes[3:4])
    [Index] = struct.unpack('<I', bytes[8:12])
    [V] = struct.unpack('<i', bytes[31:35])
    [T, t] = struct.unpack('<hh', bytes[41:45])

    return [Index, Aux, V/10000, T/10, t/10]
