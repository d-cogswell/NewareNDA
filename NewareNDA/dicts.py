# Data field names
rec_columns = [
    'Index', 'Cycle', 'Step', 'Status', 'Time', 'Voltage',
    'Current(mA)', 'Charge_Capacity(mAh)', 'Discharge_Capacity(mAh)',
    'Charge_Energy(mWh)', 'Discharge_Energy(mWh)', 'Timestamp']

# Define field precision
dtype_dict = {
    'Index': 'uint32',
    'Cycle': 'uint16',
    'Step': 'uint32',
    'Status': 'category',
    'Time': 'float32',
    'Voltage': 'float32',
    'Current(mA)': 'float32',
    'Charge_Capacity(mAh)': 'float32',
    'Discharge_Capacity(mAh)': 'float32',
    'Charge_Energy(mWh)': 'float32',
    'Discharge_Energy(mWh)': 'float32'
}

aux_dtype_dict = {
    'V': 'float32',
    'T': 'float32',
    't': 'float32'
}

# Dictionary mapping state integers to strings
state_dict = {
    1: 'CC_Chg',        # Constant current charging: fast charging the battery from low SOC with a fixed current
    2: 'CC_DChg',       # Constant current discharging: basic discharging method for capacity testing and rate performance testing
    3: 'CV_Chg',        # Constant voltage charging: voltage maintained at the upper limit, current decays with polarization to ensure no lithium precipitation due to overvoltage
    4: 'Rest',          # Resting/open circuit rest: asymptotic thermal-electrochemical equilibrium, used to measure OCV or eliminate polarization
    5: 'Cycle',         # Cycle: repeats a set of steps N times as set, statistics capacity decay, Coulomb efficiency, etc.
    7: 'CCCV_Chg',      # Constant current-constant voltage charging: industry standard CC→CV two-stage charging curve
    8: 'CP_DChg',       # Constant power discharging: loading with fixed power to evaluate the thermal-electrical coupling behavior of energy-type batteries under power-type conditions
    9: 'CP_Chg',        # Constant power charging: fast energy replenishment with fixed power, mostly used for high-power EV verification
    10: 'CR_DChg',      # Constant resistance discharging: examines current-power changes in voltage decay through a fixed load resistance
    13: 'Pause',        # Pause: manual/script temporary stop, not counted in capacity; convenient for changing lines or changing step sequences
    16: 'Pulse',        # Pulse: typical such as HPPC test, seconds of discharge/recharge pulse + rest, extract parameters such as Rs, τ
    17: 'SIM',          # Condition simulation: import driving or load waveform files, reproduce the real power curve in time series
    19: 'CV_DChg',      # Constant voltage discharging: constant voltage "pulling current" to the cell until reaching the cut-off current, mostly seen in material research or BMS failure protection scenarios
    20: 'CCCV_DChg',    # Constant current-constant voltage discharging: first set I to pull to the lower limit voltage, then constant voltage to pull small current, simulating the "power-down hold" process of power devices
    21: 'Control',      # Control/conditional jump: change the process according to IF/DO conditions (such as voltage threshold, time, temperature)
    22: 'OCV',          # Open circuit voltage sampling: measure Voc after long rest, essential in SoC modeling and aging diagnosis
    26: 'CPCV_DChg',    # Constant power-constant voltage discharging: first equal power output, turn to CV after voltage touches threshold, avoid over-discharge and maintain power stability
    27: 'CPCV_Chg'      # Constant power-constant voltage charging: use CV to finish after the power stage, which can efficiently fast charge and reduce thermal stress in the final stage
}

# Define field scaling based on instrument range settings
multiplier_dict = {
    -100000000: 10,
    -200000: 1e-2,
    -100000: 1e-2,
    -60000: 1e-2,
    -30000: 1e-2,
    -50000: 1e-2,
    -40000: 1e-2,
    -20000: 1e-2,
    -12000: 1e-2,
    -10000: 1e-2,
    -6000: 1e-2,
    -5000: 1e-2,
    -3000: 1e-2,
    -2000: 1e-2,
    -1000: 1e-2,
    -500: 1e-3,
    -100: 1e-3,
    -50: 1e-4,
    -25: 1e-4,
    -20: 1e-4,
    -10: 1e-4,
    -5: 1e-5,
    -2: 1e-5,
    -1: 1e-5,
    0: 0,
    1: 1e-4,
    2: 1e-4,
    5: 1e-4,
    10: 1e-3,
    20: 1e-3,
    50: 1e-3,
    100: 1e-2,
    200: 1e-2,
    250: 1e-2,
    500: 1e-2,
    1000: 1e-1,
    6000: 1e-1,
    10000: 1e-1,
    12000: 1e-1,
    20000: 1e-1,
    30000: 1e-1,
    40000: 1e-1,
    50000: 1e-1,
    60000: 1e-1,
    100000: 1e-1,
    200000: 1e-1,
}
