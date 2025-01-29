'''
'''
#%%

import config
from trq_stats import get_trq_stats, build_df_trq_dev
from get_deflection import get_cutting_force, get_deflection

if __name__ == '__main__':
    
    build_df_trq_dev(hv_numbers=config.HV_NUMBERS, 
                    #  bezeichnungen=['Aussenkontur', 'Steg', 'Tasche', 'Passung 10H7', 'Passung 8H7'], 
                     bezeichnungen=config.BEZEICHNUNGEN,
                     save_path=config.SAVE_PATH)
                    #  save_path=None)          