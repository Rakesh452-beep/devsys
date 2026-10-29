"""Data loading and cleaning utilities for sensor calibration ML project."""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'sensor_calibration.csv')
CLEAN_PATH = os.path.join(BASE_DIR, 'data', 'sensor_calibration_clean.csv')

ID_COLS = ['die_id']
META_COLS = ['lot_id', 'wafer_id', 'derivative', 'full_scale_g', 'nominal_sens_LSB_per_g', 'package', 'characterized_3temp']
PCM_COLS = ['pcm_vth_n_mV', 'pcm_vth_p_mV', 'pcm_rs_ohm_sq', 'pcm_tox_A', 'pcm_mobility_rel']
POS_COLS = ['die_x', 'die_y', 'x_mm', 'y_mm', 'radius_mm']

FEATURE_BASE = ['stress_gauge_mV', 'asic_vref_mV', 'mems_resonance_kHz', 'mems_q_factor',
                'st_response_x_mg', 'offset_x_25C_mg', 'sens_x_25C_rel',
                'st_response_y_mg', 'offset_y_25C_mg', 'sens_y_25C_rel',
                'st_response_z_mg', 'offset_z_25C_mg', 'sens_z_25C_rel']

TCO_TARGETS = ['tco_x_true_mg_per_C', 'tco_y_true_mg_per_C', 'tco_z_true_mg_per_C']
TCS_TARGETS = ['tcs_x_true_ppm_per_C', 'tcs_y_true_ppm_per_C', 'tcs_z_true_ppm_per_C']
TCO2_TARGETS = ['tco2_x_true_mg_per_C2', 'tco2_y_true_mg_per_C2', 'tco2_z_true_mg_per_C2']
REFLOW_TARGETS = ['post_reflow_offset_shift_x_mg', 'post_reflow_offset_shift_y_mg', 'post_reflow_offset_shift_z_mg']
CAL_CODE_TARGETS = ['cal_offset_code_x', 'cal_offset_code_y', 'cal_offset_code_z',
                    'cal_tco_code_x', 'cal_tco_code_y', 'cal_tco_code_z',
                    'cal_tcs_code_x', 'cal_tcs_code_y', 'cal_tcs_code_z']

ALL_TARGETS = TCO_TARGETS + TCS_TARGETS + TCO2_TARGETS + REFLOW_TARGETS + CAL_CODE_TARGETS

EXTREME_LABELS = (['offset_x_m40C_mg', 'sens_x_m40C_rel', 'offset_x_85C_mg', 'sens_x_85C_rel',
                   'offset_y_m40C_mg', 'sens_y_m40C_rel', 'offset_y_85C_mg', 'sens_y_85C_rel',
                   'offset_z_m40C_mg', 'sens_z_m40C_rel', 'offset_z_85C_mg', 'sens_z_85C_rel'])

def load_raw(path=None):
    path = path or DATA_PATH
    return pd.read_csv(path)

def clean(df):
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates()
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    if 'latent_defect' in df.columns:
        df['latent_defect'] = df['latent_defect'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if 'characterized_3temp' in df.columns:
        df['characterized_3temp'] = df['characterized_3temp'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})

    for c in df.columns:
        if c not in ('die_id', 'lot_id', 'wafer_id', 'derivative', 'package'):
            df[c] = pd.to_numeric(df[c], errors='coerce')

    missing = df.isna().sum()
    missing = missing[missing > 0]

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    cap_cols = [c for c in PCM_COLS + POS_COLS + ['stress_gauge_mV', 'asic_vref_mV',
                'mems_resonance_kHz', 'mems_q_factor'] + FEATURE_BASE if c in df.columns]
    for c in cap_cols:
        lo, hi = df[c].quantile([0.005, 0.995])
        df[c] = df[c].clip(lo, hi)

    return df, {'rows_before': n0, 'rows_after': len(df), 'missing': missing.to_dict()}

def split_train_test(df):
    tr = df[df['characterized_3temp'] == True].copy()
    te = df[df['characterized_3temp'] == False].copy()
    return tr, te

if __name__ == '__main__':
    df = load_raw()
    df, info = clean(df)
    print('shape', df.shape)
    print('info', info)
    print('derivatives', df.derivative.value_counts().to_dict())
    print('characterized counts', df.characterized_3temp.value_counts().to_dict())
    print('missing total', int(df.isna().sum().sum()))
    tr, te = split_train_test(df)
    print('train', tr.shape, 'test', te.shape)
    df.to_csv(CLEAN_PATH, index=False)
    print('saved cleaned to', CLEAN_PATH)
