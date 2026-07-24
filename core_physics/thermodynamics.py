"""thermodynamics module.

Handles core functionality and definitions."""
import torch
from .constants import *

def celsius_to_kelvin(T_celsius):
    """Executes celsius_to_kelvin operations."""
    return T_celsius + 273.15

def calc_henry_constant(T_celsius):
    """Executes calc_henry_constant operations."""
    T_K = celsius_to_kelvin(T_celsius)
    return torch.exp(ANTOINE_A - ANTOINE_B / (T_K + ANTOINE_C))

def calc_mixture_cp(c_urea, c_water, c_biuret):
    """Executes calc_mixture_cp operations."""
    return c_urea * CP_UREA + c_water * CP_WATER + c_biuret * CP_BIURET

def calc_p_nh3(T_celsius, P_sys, c_urea):
    """Executes calc_p_nh3 operations."""
    H = calc_henry_constant(T_celsius)
    x_nh3 = c_urea * 0.01
    return x_nh3 * H * P_sys