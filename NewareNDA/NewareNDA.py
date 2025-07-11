# © 2022-2024 Copyright SES AI
# Author: Daniel Cogswell
# Email: danielcogswell@ses.ai

import os
import mmap
import struct
import logging
from datetime import datetime, timezone
import pandas as pd
import re

from .utils import _generate_cycle_number, _count_changes
from .dicts import rec_columns, dtype_dict, aux_dtype_dict, state_dict, \
    multiplier_dict
from .NewareNDAx import read_ndax

logger = logging.getLogger('newarenda')


def _decode_remarks(byte_data):
    """
    Intelligently decode remark byte data, supporting multiple encoding formats
    
    Parameters:
        byte_data: Byte data to be decoded
    
    Returns:
        str: Decoded string, returns empty string if all encodings fail
    """
    # List of common encoding formats, sorted by priority
    encodings = ['utf-8', 'gb2312', 'gbk', 'ascii']
    
    for encoding in encodings:
        try:
            remarks = byte_data.decode(encoding)
            # Remove null characters and strip whitespace
            remarks = remarks.replace(chr(0), '').strip()
            if remarks:  # If decoding successful and not empty
                logger.info(f"Remarks (using {encoding} encoding): {remarks}")
                return remarks
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Finally try latin1, but check if it contains printable characters
    try:
        remarks = byte_data.decode('latin1')
        remarks = remarks.replace(chr(0), '').strip()
        # Check if mainly composed of printable ASCII characters
        if remarks and any(32 <= ord(c) <= 126 for c in remarks):
            # Check ratio of printable characters
            printable_ratio = sum(1 for c in remarks if 32 <= ord(c) <= 126) / len(remarks)
            if printable_ratio >= 0.8:  # If 80% or more are printable characters
                logger.info(f"Remarks (using latin1 encoding): {remarks}")
                return remarks
    except (UnicodeDecodeError, UnicodeError):
        pass
    
    # If all encodings fail or decoding result is unreasonable
    logger.warning("Unable to parse remark byte data using any encoding format")
    return ""


def read(file, software_cycle_number=True, cycle_mode='chg', log_level='INFO'):
    """
    Read electrochemical data from Neware nda or ndax binary files.

    Parameters:
        file (str): Name of the .nda or .ndax file to read
        software_cycle_number (bool): Regenerate cycle numbers to match
            Neware's "charge priority" cycle counting setting
        cycle_mode (str): Select cycle increment method.
            'chg': (default) Set new cycle with charge step after discharge.
            'dchg': Set new cycle with discharge step after charge.
            'auto': Identify the first non-rest state as increment state.
        log_level (str): Set logging level for the module. Default: "INFO"
            Options: "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
    """

    # Set logging
    log_level = log_level.upper()
    if log_level in logging._nameToLevel.keys():
        logger.setLevel(log_level)
    else:
        logger.warning(f"Logging level '{log_level}' not supported; Defaulting to 'INFO'. "
                       f"Supported options are: {', '.join(logging._nameToLevel.keys())}")

    # Identify file type and handle accordingly
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
    Function to read electrochemical data from Neware nda binary files.

    Parameters:
        file (str): Name of the .nda file to read
        software_cycle_number (bool): Generate cycle number field to match
            older versions of BTSDA
        cycle_mode (str): Select cycle increment method.
            'chg': (default) Set new cycle with charge step after discharge.
            'dchg': Set new cycle with discharge step after charge.
            'auto': Identify the first non-rest state as increment state.
    Returns:
        df (pd.DataFrame): DataFrame containing all records in the file
    """
    with open(file, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm.read(6) != b'NEWARE':
            logger.error(f"{file} does not appear to be a Neware file.")
            raise ValueError(f"{file} does not appear to be a Neware file.")

        # Get file version
        [nda_version] = struct.unpack('<B', mm[14:15])
        logger.info(f"NDA version: {nda_version}")

        # Try to find server and client version information
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

        # Version-specific settings
        # print(f"nda_version == {nda_version}")
        if nda_version == 8: # 8 based on 29 modification
            output, aux = _read_nda_8(mm)
        elif nda_version == 22: # 22 based on 29 modification
            output, aux = _read_nda_22(mm)
        elif nda_version == 23: # 23 based on 22 modification
            output, aux = _read_nda_23(mm)
        elif nda_version == 26: # 26 same as 29
            output, aux = _read_nda_26(mm)
        elif nda_version == 28: # 28 same as 29
            output, aux = _read_nda_29(mm)
        elif nda_version == 29: # original version
            output, aux = _read_nda_29(mm)
        elif nda_version == 130: # original version
            output, aux = _read_nda_130(mm)
        else:
            logger.error(f"nda version {nda_version} not supported!")
            raise NotImplementedError(f"nda version {nda_version} not supported!")

    # Create DataFrame and sort by index
    df = pd.DataFrame(output, columns=rec_columns)
    df.drop_duplicates(subset='Index', inplace=True)

    if not df['Index'].is_monotonic_increasing:
        df.sort_values('Index', inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Join temperature data
    aux_df = pd.DataFrame(aux, columns=['Index', 'Aux', 'T', 'V'])
    aux_df.drop_duplicates(inplace=True)
    if not aux_df.empty:
        aux_df = aux_df.astype(
            {k: aux_dtype_dict[k] for k in aux_dtype_dict.keys() & aux_df.columns})
        pvt_df = aux_df.pivot(index='Index', columns='Aux')
        pvt_df.columns = pvt_df.columns.map(lambda x: ''.join(map(str, x)))
        df = df.join(pvt_df, on='Index')

    # Post-processing
    df['Step'] = _count_changes(df['Step'])

    # v8 file supplement capacity and energy (obtained through numerical integration)
    if nda_version == 8:
        # Δt processing: Time resets when encountering new step, need to replace negative differences with current Time
        dt = df['Time'].diff()
        dt.iloc[0] = 0
        dt = dt.where(dt >= 0, df['Time'])

        # Absolute timestamp
        if df['Timestamp'].notnull().any():
            start_ts = df['Timestamp'].iloc[0]
            df['Timestamp'] = start_ts + pd.to_timedelta(dt.cumsum(), unit='s')

        # mAh increment
        cap_inc = (df['Current(mA)'].abs()) * dt / 3600

        # Charge/discharge capacity accumulation
        df['Charge_Capacity(mAh)'] = (cap_inc.where(df['Current(mA)'] > 0, 0)).cumsum()
        df['Discharge_Capacity(mAh)'] = (cap_inc.where(df['Current(mA)'] < 0, 0)).cumsum()

        # Energy increment (mWh)
        en_inc = df['Voltage'] * cap_inc
        df['Charge_Energy(mWh)'] = (en_inc.where(df['Current(mA)'] > 0, 0)).cumsum()
        df['Discharge_Energy(mWh)'] = (en_inc.where(df['Current(mA)'] < 0, 0)).cumsum()

    if software_cycle_number:
        df['Cycle'] = _generate_cycle_number(df, cycle_mode)
    df = df.astype(dtype=dtype_dict)

    # Output specified columns uniformly as required
    desired_columns = [
        'Index', 'Cycle', 'Step', 'Status', 'Time', 'Voltage',
        'Current(mA)', 'Timestamp'
    ]
    # Filter out columns that don't exist in df
    existing_columns = [col for col in desired_columns if col in df.columns]
    df = df[existing_columns]

    return df, nda_version


def _read_nda_8(mm):
    """Helper function for nda version 8.

    Structure differs significantly from later versions, with record length of only 59 B,
    but still starts with 0x55 0x00, no explicit header identifier. Using following empirical parsing:

    Byte offset (based on empirical file):
    0–1   : 0x55 0x00 fixed frame header
    2–5   : Index (uint32)
    10    : Step (uint8)
    18    : Status (uint8)
    20–23 : Current (int32, μA)
    24–27 : Voltage (int32, 1e-4 V)
    Other fields left empty or not yet parsed.
    """

    mm_size = mm.size()

    # Active material mass and remarks section don't exist or cannot be reliably read in old files, skip here.

    record_len = 59

    # Search for first 0x55 0x00 anywhere in file as starting record.
    header = -1
    search_pos = 0
    while True:
        cand = mm.find(b'\x55\x00', search_pos)
        if cand == -1 or cand + record_len >= mm_size:
            break
        # Check if next 0x55 0x00 is exactly 59 bytes apart
        if mm[cand + record_len: cand + record_len + 2] == b'\x55\x00':
            header = cand
            break
        search_pos = cand + 1

    if header == -1:
        logger.error("Could not find record header satisfying 59 B interval pattern, unable to parse nda v8 file!")
        raise EOFError("File does not contain any valid records.")

    mm.seek(header)

    output = []
    idx = 1  # Auto-increment index as backup for Index

    # --------------------
    # Extract start timestamp (ASCII string)
    # Typical format: "YYYY.MM.DD HH:MM:SS"
    # --------------------
    start_ts = None
    m = re.search(rb'20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2}', mm[:header])
    if m:
        try:
            ts_str = m.group().decode()
            start_ts = datetime.strptime(ts_str, "%Y.%m.%d %H:%M:%S")
            # Convert to local timezone
            start_ts = start_ts.replace(tzinfo=timezone.utc).astimezone()
        except Exception:
            start_ts = None

    while mm.tell() + record_len <= mm_size:
        chunk = mm.read(record_len)

        # Record must start with 0x55 0x00
        if chunk[0:2] != b'\x55\x00':
            break

        rec = _bytes_to_list_8(chunk, fallback_index=idx)
        if rec:
            # Calculate absolute timestamp
            if start_ts is not None:
                try:
                    rec[11] = start_ts + pd.to_timedelta(rec[4], unit='s')
                except Exception:
                    pass
            output.append(rec)
        idx += 1

    # v8 has no auxiliary temperature records found yet
    aux = []
    return output, aux


def _bytes_to_list_8(bytes, fallback_index=0):
    """Parse nda v8 version 59 B data record, return list conforming to rec_columns."""

    # Index field in v8 files is usually 0, so use fallback index directly
    Index = fallback_index

    # Status code at byte 18; use it directly as "step number", renumbered later by _count_changes
    Status_code = bytes[18]
    Step = Status_code  # Step number based on status changes, not record count

    # v8 version record format (reverse engineered)
    # 20–23 : Time (int32, s)
    # 24–27 : Voltage (int32, 1e-4 V)
    # 28–31 : Current (int32, µA)

    Time = struct.unpack('<i', bytes[20:24])[0]
    Voltage_raw = struct.unpack('<i', bytes[24:28])[0]
    Current_raw = struct.unpack('<i', bytes[28:32])[0]

    Voltage = Voltage_raw / 10000            # Convert to V
    Current = Current_raw / 1000             # Convert to mA

    # Placeholders: capacity and energy calculated later in read_nda() based on current integration
    Charge_capacity = Discharge_capacity = 0.0
    Charge_energy = Discharge_energy = 0.0

    # v8 files usually contain only single cycle
    Cycle = 1

    # Custom v8 status mapping. Fall back to general state_dict if not included
    v8_state_dict = {
        1: 'Rest',
        2: 'CC_DChg',
        3: 'CV_DChg',
        4: 'CC_DChg',
    }

    # First try to get status from v8_state_dict, if not found then from state_dict
    # If neither exists, raise KeyError
    try:
        Status = v8_state_dict[Status_code]
    except KeyError:
        try:
            Status = state_dict[Status_code]
        except KeyError:
            logger.error(f"Unknown status code {Status_code} in nda v8 file")
            raise KeyError(f"Unknown status code {Status_code} in nda v8 file")

    # Timestamp placeholder; calculated later in _read_nda_8 using start time
    timestamp = None

    return [
        Index,
        Cycle,
        Step,
        Status,
        Time,
        Voltage,
        Current,
        Charge_capacity,
        Discharge_capacity,
        Charge_energy,
        Discharge_energy,
        timestamp
    ]


def _read_nda_22(mm):
    """Helper function for nda version 22 (temporarily same as version 29)"""
    mm_size = mm.size()

    # Get active material mass
    [active_mass] = struct.unpack('<I', mm[152:156])
    logger.info(f"Active mass: {active_mass/1000} mg")

    # Use intelligent decoding function to handle remarks
    remarks = _decode_remarks(mm[2317:2417])

    # Identify beginning of data section
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
    output = []
    aux = []
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        if len(bytes) == record_len:

            # Check data record
            if (bytes[0:2] == b'\x55\x00'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                output.append(_bytes_to_list_22(bytes))

            # Check auxiliary record
            elif (bytes[0:1] == b'\x65'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                aux.append(_aux_bytes_to_list(bytes))

    return output, aux

def _read_nda_23(mm):
    """Helper function for nda version 23"""
    mm_size = mm.size()

    # Get active material mass
    [active_mass] = struct.unpack('<I', mm[152:156])
    logger.info(f"Active mass: {active_mass/1000} mg")

    # Use intelligent decoding function to handle remarks
    remarks = _decode_remarks(mm[2317:2417])

    # Identify beginning of data section
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
    output = []
    aux = []
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        if len(bytes) == record_len:

            # Check data record
            if (bytes[0:2] == b'\x55\x00'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                output.append(_bytes_to_list_23(bytes))

            # Check auxiliary record
            elif (bytes[0:1] == b'\x65'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                aux.append(_aux_bytes_to_list(bytes))

    return output, aux


def _read_nda_26(mm):
    """Helper function for nda version 26"""
    mm_size = mm.size()

    # Get active material mass
    [active_mass] = struct.unpack('<I', mm[152:156])
    logger.info(f"Active mass: {active_mass/1000} mg")

    # Use intelligent decoding function to handle remarks
    remarks = _decode_remarks(mm[2317:2417])

    # Identify beginning of data section
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
    output = []
    aux = []
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        if len(bytes) == record_len:

            # Check data record
            if (bytes[0:2] == b'\x55\x00'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                output.append(_bytes_to_list(bytes))

            # Check auxiliary record
            elif (bytes[0:1] == b'\x65'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                aux.append(_aux_bytes_to_list(bytes))

    return output, aux


def _read_nda_29(mm):
    """Helper function for nda version 29"""
    mm_size = mm.size()

    # Get active material mass
    [active_mass] = struct.unpack('<I', mm[152:156])
    logger.info(f"Active mass: {active_mass/1000} mg")

    # Use intelligent decoding function to handle remarks
    remarks = _decode_remarks(mm[2317:2417])

    # Identify beginning of data section
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
    output = []
    aux = []
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        if len(bytes) == record_len:

            # Check data record
            if (bytes[0:2] == b'\x55\x00'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                output.append(_bytes_to_list(bytes))

            # Check auxiliary record
            elif (bytes[0:1] == b'\x65'
                    and bytes[82:87] == b'\x00\x00\x00\x00'):
                aux.append(_aux_bytes_to_list(bytes))

    return output, aux


def _read_nda_130(mm):
    """Helper function for nda version 130"""
    mm_size = mm.size()

    # Identify beginning of data section
    record_len = 88
    identifier = mm[1024:1030]
    if mm[1024:1025] == b'\x55':  # BTS 9.1
        # Find next record and get length
        record_len = mm.find(mm[1024:1026], 1026) - 1024
    mm.seek(1024)

    # Read data records
    output = []
    aux = []
    while mm.tell() < mm_size:
        bytes = mm.read(record_len)
        if len(bytes) == record_len:

            # Check data record
            if bytes[0:1] == b'\x55':
                output.append(_bytes_to_list_BTS91(bytes))
                if record_len == 56:
                    aux.append(_aux_bytes_to_list_BTS91(bytes))
            elif bytes[0:6] == identifier:
                output.append(_bytes_to_list_BTS9(bytes[4:]))

            # Check auxiliary record
            elif bytes[0:5] == b'\x00\x00\x00\x00\x65':
                aux.append(_aux_bytes_to_list(bytes[4:]))

            elif bytes[0:1] == b'\x81':
                break

    # Find footer data block
    footer = mm.rfind(b'\x06\x00\xf0\x1d\x81\x00\x03\x00\x61\x90\x71\x90\x02\x7f\xff\x00', 1024)
    if footer != -1:
        mm.seek(footer+16)
        bytes = mm.read(499)

        # Get active material mass
        [active_mass] = struct.unpack('<d', bytes[-8:])
        logger.info(f"Active mass: {active_mass} mg")

        # Use intelligent decoding function to handle remarks
        remarks = _decode_remarks(bytes[363:491])

    return output, aux


def _valid_record(bytes):
    """Helper function to identify valid records"""
    # Check non-zero status
    [Status] = struct.unpack('<B', bytes[12:13])
    return (Status != 0)


def _bytes_to_list(bytes):
    """Helper function to interpret byte strings"""

    # Extract fields from byte string
    [Index, Cycle, Step] = struct.unpack('<III', bytes[2:14])
    [Status, Jump, Time, Voltage, Current] = struct.unpack('<BBQii', bytes[12:30])
    [Charge_capacity, Discharge_capacity,
     Charge_energy, Discharge_energy] = struct.unpack('<qqqq', bytes[38:70])
    [Y, M, D, h, m, s] = struct.unpack('<HBBBBB', bytes[70:77])
    [Range] = struct.unpack('<i', bytes[78:82])

    # Index should not be zero
    if Index == 0 or Status == 0:
        return []

    multiplier = multiplier_dict[Range]

    # Create dictionary for record
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


def _bytes_to_list_BTS9(bytes):
    """Helper function to interpret BTS9 byte strings"""
    [Step, Status] = struct.unpack('<BB', bytes[5:7])
    [Index] = struct.unpack('<I', bytes[12:16])
    [Time, Voltage, Current] = struct.unpack('<Qff', bytes[24:40])
    [Charge_Capacity, Charge_Energy,
     Discharge_Capacity, Discharge_Energy,
     Date] = struct.unpack('<ffffQ', bytes[48:72])

    # Create dictionary for record
    list = [
        Index,
        0,
        Step,
        state_dict[Status],
        Time/1e6,
        Voltage,
        Current,
        Charge_Capacity/3600,
        Discharge_Capacity/3600,
        Charge_Energy/3600,
        Discharge_Energy/3600,
        datetime.fromtimestamp(Date/1e6, timezone.utc).astimezone()
    ]
    return list


def _bytes_to_list_BTS91(bytes):
    """Helper function to interpret BTS9.1 byte strings"""
    [Step, Status] = struct.unpack('<BB', bytes[2:4])
    [Index, Time, Time_ns] = struct.unpack('<III', bytes[8:20])
    [Current, Voltage, Capacity, Energy] = struct.unpack('<ffff', bytes[20:36])
    [Date, Date_ns] = struct.unpack('<II', bytes[44:52])

    # Convert capacity and energy to charge and discharge fields
    Charge_Capacity = 0 if Capacity < 0 else Capacity
    Discharge_Capacity = 0 if Capacity > 0 else abs(Capacity)
    Charge_Energy = 0 if Energy < 0 else Energy
    Discharge_Energy = 0 if Energy > 0 else abs(Energy)

    # Create dictionary for record
    list = [
        Index,
        0,
        Step,
        state_dict[Status],
        Time + 1e-9*Time_ns,
        Voltage,
        Current,
        Charge_Capacity/3600,
        Discharge_Capacity/3600,
        Charge_Energy/3600,
        Discharge_Energy/3600,
        datetime.fromtimestamp(Date + 1e-9*Date_ns, timezone.utc).astimezone()
    ]
    return list


def _aux_bytes_to_list(bytes):
    """Helper function to interpret auxiliary records"""
    [Aux, Index] = struct.unpack('<BI', bytes[1:6])
    [V] = struct.unpack('<i', bytes[22:26])
    [T] = struct.unpack('<h', bytes[34:36])

    return [Index, Aux, T/10, V/10000]


def _aux_bytes_to_list_BTS91(bytes):
    [Index] = struct.unpack('<I', bytes[8:12])
    [T] = struct.unpack('<f', bytes[52:56])
    return [Index, 1, T, None]


def _bytes_to_list_22(bytes):
    """Parse nda version 22 data record

    This version has mostly the same data structure as the newer version 29, but with following differences:
    1. Step field only occupies 2 bytes (uint16), immediately following Cycle.
    2. Date time stored as Unix epoch (seconds) + milliseconds, 4 bytes and 2 bytes respectively,
       at the position where original year and other fields were located.
    3. Range field shifted back 2 bytes to offset 80–83.
    """
    
    # v22 version specific multiplier dictionary, uses 1e-2 scaling when Range is 0
    multiplier_dict_v22 = multiplier_dict.copy()
    multiplier_dict_v22[0] = 1e-2

    # Basic field parsing
    Index, Cycle = struct.unpack('<II', bytes[2:10])
    Step,         = struct.unpack('<H',  bytes[10:12])
    Status, Jump  = struct.unpack('<BB', bytes[12:14])

    # Time, voltage, current
    Time, Voltage, Current = struct.unpack('<Qii', bytes[14:30])

    # Capacity / energy (int64)
    Charge_capacity, Discharge_capacity, Charge_energy, Discharge_energy = \
        struct.unpack('<qqqq', bytes[38:70])

    # Timestamp (seconds) + milliseconds
    Timestamp_sec, = struct.unpack('<I', bytes[70:74])
    # bytes[74:78] seems to be reserved bytes (all 0)
    Msec,          = struct.unpack('<H', bytes[78:80])

    # Range
    Range,         = struct.unpack('<i', bytes[80:84])

    # Skip invalid index or rest status
    if Index == 0 or Status == 0:
        return []

    multiplier = multiplier_dict_v22[Range]

    # Generate local timezone timestamp
    ts = datetime.fromtimestamp(Timestamp_sec + Msec/1000, timezone.utc).astimezone()

    rec = [
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
        ts
    ]

    return rec

def _bytes_to_list_23(bytes):
    """Parse nda version 23 data record

    This version has mostly the same data structure as v22, but different current scaling factor when Range=0.
    1. Step field only occupies 2 bytes (uint16), immediately following Cycle.
    2. Date time stored as Unix epoch (seconds) + milliseconds, 4 bytes and 2 bytes respectively,
       at the position where original year and other fields were located.
    3. Range field shifted back 2 bytes to offset 80–83.
    """
    
    # v23 version specific multiplier dictionary, uses 1e-3 scaling when Range is 0
    multiplier_dict_v23 = multiplier_dict.copy()
    multiplier_dict_v23[0] = 1e-3

    # Basic field parsing
    Index, Cycle = struct.unpack('<II', bytes[2:10])
    Step,         = struct.unpack('<H',  bytes[10:12])
    Status, Jump  = struct.unpack('<BB', bytes[12:14])

    # Time, voltage, current
    Time, Voltage, Current = struct.unpack('<Qii', bytes[14:30])

    # Capacity / energy (int64)
    Charge_capacity, Discharge_capacity, Charge_energy, Discharge_energy = \
        struct.unpack('<qqqq', bytes[38:70])

    # Timestamp (seconds) + milliseconds
    Timestamp_sec, = struct.unpack('<I', bytes[70:74])
    # bytes[74:78] seems to be reserved bytes (all 0)
    Msec,          = struct.unpack('<H', bytes[78:80])

    # Range
    Range,         = struct.unpack('<i', bytes[80:84])

    # Skip invalid index or rest status
    if Index == 0 or Status == 0:
        return []

    multiplier = multiplier_dict_v23[Range]

    # Generate local timezone timestamp
    ts = datetime.fromtimestamp(Timestamp_sec + Msec/1000, timezone.utc).astimezone()

    rec = [
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
        ts
    ]

    return rec