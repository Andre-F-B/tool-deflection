'''
'''

import numpy as np

def get_cutting_force(torque, radius, effective_radius=0.8):
    '''Takes the torque (N.m) and the radius (m) as input and outputs the cutting force (N)'''
    return torque / (radius * effective_radius)

def get_deflection(cutting_force, tool_length, elasticity, moment=None, diameter=None):
    '''
    Calculates the deflection (mm) based on cutting force (N), tool_length (mm), area moment of inertia (mm^4) and modulus of elasticity (MPa)
    Alternatively, the diameter (mm) can be used to calculate the area moment of inertia
    '''

    if moment == None:
        # If the moment is not given, calculate it, approximating the spindle as a solid circular cylinder
        moment = (np.pi * (diameter ** 4)) / 64

    return (cutting_force * (tool_length ** 3)) / (3 * elasticity * moment)