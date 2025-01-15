'''
'''

import numpy as np

def get_cutting_force(torque, radius, effective_radius=0.8):
    '''Takes the torque (N.m) and the radius (m) as input and outputs the cutting force (N)'''
    return torque / (radius * effective_radius)

def get_deflection(cutting_force, tool_length, diameter, elasticity):
    '''Calculates the deflection (mm) based on cutting force (N), tool_length (mm), diameter (mm) and modulus of elasticity (GPa)'''
    return (1e3) * (64 * cutting_force * (tool_length ** 3)) / (3 * np.pi * elasticity * (diameter ** 4))