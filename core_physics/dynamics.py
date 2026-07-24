"""dynamics module.

Handles core functionality and definitions."""
import torch
from .constants import *
from .thermodynamics import celsius_to_kelvin, calc_mixture_cp, calc_p_nh3

def compute_odes(state, action, feed_conditions):
    """Executes compute_odes operations."""
    M, T, c_u, c_w, c_b = state.unbind(dim=-1)
    Q, P = action.unbind(dim=-1)
    F_in = feed_conditions['F_in']
    T_in = feed_conditions['T_in']
    c_u_in = feed_conditions['c_u_in']
    c_w_in = feed_conditions['c_w_in']
    T_K = celsius_to_kelvin(T)
    k_fwd = A_FWD * torch.exp(-E_FWD / (R_GAS * T_K))
    k_rev = A_REV * torch.exp(-E_REV / (R_GAS * T_K))
    p_nh3 = calc_p_nh3(T, P, c_u)
    r_B = k_fwd * c_u ** 2 - k_rev * c_b * p_nh3
    P_sat_w = torch.exp(11.9 - 3985.0 / (T_K - 39.0))
    evap_driving_force = torch.relu(P_sat_w - P)
    m_vap = 0.05 * M * c_w * evap_driving_force
    F_out = F_in - m_vap
    dM_dt = F_in - F_out - m_vap
    dc_u_dt = (F_in * c_u_in - F_out * c_u - r_B * M - c_u * dM_dt) / M
    dc_w_dt = (F_in * c_w_in - F_out * c_w - m_vap - c_w * dM_dt) / M
    dc_b_dt = (-F_out * c_b + r_B * M - c_b * dM_dt) / M
    Cp_mix = calc_mixture_cp(c_u, c_w, c_b)
    Cp_in = calc_mixture_cp(c_u_in, c_w_in, torch.zeros_like(c_u_in))
    H_in = F_in * Cp_in * T_in
    H_out = F_out * Cp_mix * T
    dT_dt = (H_in - H_out + Q - m_vap * DH_VAP_WATER - r_B * M * DH_REACTION) / (M * Cp_mix)
    return torch.stack([dM_dt, dT_dt, dc_u_dt, dc_w_dt, dc_b_dt], dim=-1)