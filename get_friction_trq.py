'''
'''

import pickle

import config

def get_friction_trq(spindle_speed, model=None, return_std=False, model_path=config.FRICTION_MODEL_PATH):
    '''
    Given an array of spindle speeds, return an array of the corresponding friction torque
    Parameters:
        spindle_speed (pd.Series): array of spindle speeds in the original unit (0,001 degrees/second)
    '''

    if not model:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    S1ActTrq_friction, std = model.predict(spindle_speed.values.reshape(-1, 1), return_std=True)

    if return_std:
        return S1ActTrq_friction, std
    return S1ActTrq_friction