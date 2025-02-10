'''
'''

import pandas as pd
from pathlib import Path

import utils
import config
from get_friction_trq import get_friction_trq
from tsdb_data_handler import read_data

def get_trq_stats(hv_numbers, bezeichnungen, outputdata=None):
    '''
    Given a list of HV numbers and bezeichnungen, builds a df containing the torque stats (mean, std, etc.) for the operation corresponding to each HV number and bezeichnung
    '''
    if type(hv_numbers) == int:
        # In case a single number is given, turn it into a list
        hv_numbers = [hv_numbers]
    if type(outputdata) != pd.DataFrame:
        outputdata = pd.read_excel(config.QUALITYDATA_PATH, sheet_name='OutputData')

    list_trq_stats = []

    for hv_number in hv_numbers:

        # I don't know why HV141 and HV157 have multiple values, so we skip them
        if (hv_number == 141) or (hv_number == 157):
            print(f'HV{hv_number} has been skipped')
            continue

        # If there was a problem during manufacturing (i.e. Störung==1), skip the HV
        problem = utils.lookup_outputdata(outputdata, hv_number, 'Aussenkontur', column='Störung')
        if problem == 1:
            print(f'HV{hv_number} failed: Problem during manufacturing')
            continue
        
        try:
            # My preprocessing:
            df = utils.read_timeseries_data(hv_number)

            # Erkut's preprocessing:
            # folder_name = utils.get_HV_folder_name(hv_number)[0]
            # folder_path = Path(r'D:\HiWi 2.0\Daten\Hauptversuche', folder_name, r'Daten\Zeitreihendaten')
            # df, _ = read_data(folder_path)
            
            if df.empty:
                # If preprocessing returned an empty df, move on to the next HV
                continue
        except:
            continue

        # Subtract friction torque from S1ActTrq
        df['S1ActTrq_friction'] = get_friction_trq(df['S1Actrev'])
        df['S1ActTrq_corrected'] = df['S1ActTrq'] - df['S1ActTrq_friction']

        # Drop rows with no information about the operation or the tool
        df.dropna(inplace=True)

        for bezeichnung in bezeichnungen:

            # HV30 has all zeros for Aussenkontur, I'll deal with it later
            if (hv_number == 30) and (bezeichnung == 'Aussenkontur'):
                print(f'HV{hv_number} ({bezeichnung}) has been skipped')
                continue

            # Find the name of the operation corresponding to the bezeichnung
            operation = utils.lookup_outputdata(outputdata, hv_number, bezeichnung, column='Operationsname')

            # Ensure there is tool info for the given operation
            if df[df['Operation'] == operation]['ToolID'].empty:
                print(f'HV{hv_number} - {bezeichnung} failed: No tool info')
                continue

            # Ensure the spindle speed is constant and greater than zero in the interval
            df_constant = utils.get_operation_interval(df, operation, spindle_speed_column='S1Actrev', min_speed=100)

            # Ensure the removal volume is greater than zero in the interval
            df_constant = df_constant[df_constant['Removed_Volume V'] > 0]

            # Some operations use multiple tools, find out the most common tool in this operation
            try:
                tool_id, n_tools = utils.find_most_common_tool(df_constant, operation)
            except:
                print(f'HV{hv_number} - {bezeichnung} failed')
                continue

            # Get torque stats for that specific operation and tool ID
            operation_tool_filter = (df_constant['Operation'] == operation) & (df_constant['ToolID'] == tool_id)   # This should be redundant

            trq_stats = df_constant[operation_tool_filter]['S1ActTrq_corrected'].describe().rename(f'HV{hv_number}')
            trq_stats['Bezeichnung'] = bezeichnung
            trq_stats['n_tools'] = n_tools
            trq_stats['ToolID'] = tool_id
            # Tool length and diameter should be the same throughout df_constant[operation_tool_filter] 
            trq_stats['Tool_Length'] = df_constant[operation_tool_filter]['Tool_Length'].iloc[0]
            trq_stats['Tool_Diameter'] = df_constant[operation_tool_filter]['Tool_Diameter'].iloc[0]
            list_trq_stats.append(trq_stats)

    return pd.DataFrame(list_trq_stats).reset_index(names='HV-Nummer')

def build_df_trq_dev(hv_numbers, 
                     bezeichnungen, 
                     expected_values=config.EXPECTED_VALUES, 
                     df_trq_stats=None, 
                     df_outputdata=None, 
                     save_path=None):
    '''
    Builds a dataset that merges torque stats and quality data for each Bezeichnung
    '''

    if type(df_outputdata) != pd.DataFrame:
        # Read the output data sheet from Qualitätsdaten_Final.xlsx, then get only the desired Hauptversuche
        df_outputdata_full = pd.read_excel(config.QUALITYDATA_PATH, sheet_name='OutputData')
        df_outputdata = df_outputdata_full[df_outputdata_full['HV-Nummer'].str[2:].astype(int).isin(hv_numbers)]
    if type(df_trq_stats) != pd.DataFrame:    
        df_trq_stats = get_trq_stats(hv_numbers, bezeichnungen=bezeichnungen, outputdata=df_outputdata)
    
    # df_merged = pd.merge(df_trq_stats, df_outputdata, on=['HV-Nummer', 'Bezeichnung'])
    # for bezeichnung in bezeichnungen:
    #     for i, column in enumerate(expected_values[bezeichnung]['columns']):
    #         df_merged[f'{column}_deviation'] = df_merged[column] - expected_values[bezeichnung]['expected_values'][i]
    #         print(f'{column} - {expected_values[bezeichnung]['expected_values'][i]}')

    dfs = []
    for bezeichnung in bezeichnungen:
        trq_bezeichnung_filter = (df_trq_stats['Bezeichnung'] == bezeichnung)
        output_bezeichnung_filter = (df_outputdata['Bezeichnung'] == bezeichnung)
        df_merged_bezeichnung = pd.merge(df_trq_stats[trq_bezeichnung_filter], 
                                         df_outputdata[output_bezeichnung_filter], 
                                         on=['HV-Nummer', 'Bezeichnung'])
        
        # Calculate the deviations based on the dictionary of expected values
        for i, column in enumerate(expected_values[bezeichnung]['columns']):
            df_merged_bezeichnung[f'{column}_deviation'] = df_merged_bezeichnung[column] - expected_values[bezeichnung]['expected_values'][i]

        if save_path:
            # Save the df for that bezeichnung to its own csv
            df_merged_bezeichnung.to_csv(save_path + f'_{bezeichnung}.csv', index=False)
            # df_merged.to_excel(save_path[:-3] + 'xlsx', index=False)
            # df_merged_bezeichnung.to_excel(xl_writer, sheet_name=bezeichnung, index=False)

        dfs.append(df_merged_bezeichnung)

    if save_path:
        # Save an excel file where each sheet corresponds to the df of a different bezeichnung
        xl_writer = pd.ExcelWriter(save_path + '.xlsx', engine='xlsxwriter')
        for df, bezeichnung in zip(dfs, bezeichnungen):
            df.to_excel(xl_writer, sheet_name=bezeichnung, index=False)
        xl_writer.close()

    return dfs