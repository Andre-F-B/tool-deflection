

# Hauptversuche to be considered when building the dataset
HV_NUMBERS = [i for i in range(1, 181)]

# List of Bezeichnungen to be considered. The options are ['Aussenkontur', 'Steg', 'Tasche', 'Passung 10H7', 'Passung 8H7']
BEZEICHNUNGEN = ['Aussenkontur', 'Steg', 'Tasche', 'Passung 10H7', 'Passung 8H7']

# Path to save the final dataset
SAVE_PATH = r'C:\Users\Usuario\Desktop\TUD\HiWi 2.0\Projekt\Deviation\data\df_trq_dev'

# Path to Qualitätsdaten_Final.xlsx
QUALITYDATA_PATH = r'D:\HiWi 2.0\Daten\Qualitätsdaten_Final.xlsx'

FRICTION_MODEL_PATH = r'C:\Users\Usuario\Desktop\TUD\HiWi 2.0\Projekt\models\sp_friction_model'

EXPECTED_VALUES = {
    'Aussenkontur': {'columns': ['XMaß_Zmax [mm]', 'YMaß_Zmax [mm]'], 'expected_values': [60, 60]}, 
    'Steg': {'columns': ['XMaß_Zmin [mm]', 'YMaß_Zmin [mm]'], 'expected_values': [56, 56]}, 
    'Tasche': {'columns': ['XMaß_Zmax [mm]', 'YMaß_Zmax [mm]'], 'expected_values': [40, 40]}, 
    'Passung 10H7': {'columns': ['Durchmesser_Zmax [mm]'], 'expected_values': [10]}, 
    'Passung 8H7': {'columns': ['Durchmesser_Zmax [mm]'], 'expected_values': [8]}
}