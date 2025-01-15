'''
'''
#%%
from trq_stats import get_trq_stats, build_df_trq_dev
from get_deflection import get_cutting_force, get_deflection

if __name__ == '__main__':
    
    build_df_trq_dev(hv_numbers=[i for i in range(1, 181)], 
                     bezeichnungen=['Aussenkontur', 'Steg', 'Tasche', 'Passung 10H7', 'Passung 8H7'], 
                    # bezeichnungen=['Passung 8H7'],
                    #  save_path=r'C:\Users\Usuario\Desktop\TUD\HiWi 2.0\Projekt\Deviation\df_trq_dev')
                     save_path=None)