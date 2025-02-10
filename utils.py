'''
'''

import os
import pandas as pd
from pathlib import Path

HAUPTVERSUCHE_PATH = Path(r'D:\HiWi 2.0\Daten\Hauptversuche')

def get_HV_folder_name(hv_numbers, hve_path=Path(HAUPTVERSUCHE_PATH)):
    
    if type(hv_numbers) is int:
        hv_numbers = [hv_numbers]
    if type(hv_numbers) is list:
        folder_names = []
        for hv_number in hv_numbers:
            for folder_name in os.listdir(hve_path):
                if f'HV{hv_number}_' in folder_name:
                    folder_names.append(folder_name)
                    break
        return folder_names

def read_timeseries_data(hv_number):
    '''Reads and merges all timeseries datasets (from one single HV) necessary for the deviation pipeline'''
    
    folder_name = get_HV_folder_name(hv_number)[0]
    try:
        # Read drive, prog, cnc and tool datasets to merge them
        drive = pd.read_parquet(Path(r'D:\HiWi 2.0\Daten\Hauptversuche', folder_name, r'Daten\Zeitreihendaten\drive.parquet'))
        prog = pd.read_parquet(Path(r'D:\HiWi 2.0\Daten\Hauptversuche', folder_name, r'Daten\Zeitreihendaten\prog.parquet'))
        cnc = pd.read_parquet(Path(r'D:\HiWi 2.0\Daten\Hauptversuche', folder_name, r'Daten\Zeitreihendaten\cnc.parquet'))
        tool = pd.read_parquet(Path(r'D:\HiWi 2.0\Daten\Hauptversuche', folder_name, r'Daten\Zeitreihendaten\tool.parquet')).rename(columns={'Number': 'ToolID', 'Diameter': 'Tool_Diameter', 'Length': 'Tool_Length'}) 
    except:
        print(f'HV{hv_number} failed: Time series datasets not available')
        return pd.DataFrame()
    try:
        sim = pd.read_csv(Path(r'D:\HiWi 2.0\Daten\sim_voxout', f'vox_sim_HV{hv_number}.csv'))
    except:
        print(f'HV{hv_number} failed: Simulation dataset not available')
        return pd.DataFrame()

    # We get the torque feedback from drive, the operation from prof, the spindle speed from cnc, and the tool ID from tool
    df = pd.merge_asof(drive[['Timestamp', 'S1ActTrq']], prog[['Timestamp', 'Operation']], on='Timestamp', direction='backward')
    df = pd.merge_asof(df, cnc[['Timestamp', 'S1Actrev']], on='Timestamp', direction='backward')
    df = pd.merge_asof(df, tool[['Timestamp', 'ToolID', 'Tool_Diameter', 'Tool_Length']], on='Timestamp', direction='backward')
    df = pd.merge_asof(df, sim[['Timestamp', 'Removed_Volume V']], on='Timestamp', direction='backward')

    # Convert the torque column to N.m and the spindle speed column to RPM
    df["S1ActTrq"] = df["S1ActTrq"] / 1000 * 3.8
    df["S1Actrev"] = df["S1Actrev"] * (0.001/360) * 60

    return df

def lookup_outputdata(outputdata, hv_number, bezeichnung, column):
    hv_bezeichnung_filter = (outputdata['HV-Nummer'] == f'HV{hv_number}') & (outputdata['Bezeichnung'] == bezeichnung)
    filtered = outputdata[hv_bezeichnung_filter][column]
    if len(filtered) == 1:
        return outputdata[hv_bezeichnung_filter][column].iloc[0]
    raise ValueError('No/Multiple matches found')

def find_most_common_tool(df, operation):
    '''
    For a given operation, counts the frequency of each tool ID, returns the most frequent tool ID, as well as the number of different tool IDs
    Args:
        df (pd.DataFrame): dataframe with the oeration names and tool ids across time
        operation (str): name of the operation
    '''
    
    value_counts = df.loc[df['Operation'] == operation, 'ToolID'].value_counts()
    n_tools = len(value_counts)
    most_common_tool_id = value_counts.index[value_counts.argmax()]
    return most_common_tool_id, n_tools

def get_operation_interval(df, operation, spindle_speed_column='S1Actrev', min_speed=20):
    '''
    Searches the df given and returns the period where the operation is correct and the spindle speed is constant and greater than min_speed
    '''

    # Finding the interval in which the spindle speed is positive (greater than min_speed) and the operation is correct
    mask = (df[spindle_speed_column] > min_speed) & (df['Operation'] == operation)
    spindle_speed_intervals = find_all_intervals(mask)
    
    # Checking if there's really only one continuous interval that matches the criteria
    if len(spindle_speed_intervals) != 1:
        print('Something is wrong with the spindle speed and operation mask')
        return None
    
    # Building a separate df for mask
    interval = spindle_speed_intervals[0]
    # df_filtered = df[mask] should also work since len(spindle_speed_intervals) == 1
    df_filtered = df.iloc[interval[0]:interval[1]]

    # Getting only the (sub)interval where the spindle speed is constant near its maximum
    max_speed_mask = (df_filtered[spindle_speed_column] >= df_filtered[spindle_speed_column].max() - 1)
    max_speed_intervals = find_all_intervals(max_speed_mask) # should be [(start_index, end_index)]
    if len(max_speed_intervals) != 1:
        print('Something is wrong with the max speed mask')
        return None
    df_constant = df_filtered.iloc[max_speed_intervals[0][0]:max_speed_intervals[0][1]]
    return df_constant

def find_all_intervals(mask):
    '''
    Returns the indexes for all intervals in a dataframe where a given condition is respected
    '''

    intervals = []
    in_interval = False
    start_idx = None

    for i, value in enumerate(mask):
        if value and not in_interval:
            # Start of a new range
            in_interval = True
            start_idx = i
        elif not value and in_interval:
            # End of the current range
            in_interval = False
            intervals.append((start_idx, i-1))

    # If the last value was part of a range, close it
    if in_interval:
        intervals.append((start_idx, len(mask) - 1))

    # print("Non-zero index intervals:", intervals)
    return intervals