import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data import FEATURE_BASE, PCM_COLS, POS_COLS, ALL_TARGETS, TCO_TARGETS, TCS_TARGETS, TCO2_TARGETS, REFLOW_TARGETS, CAL_CODE_TARGETS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
METRICS_PATH = os.path.join(BASE_DIR, 'outputs', 'model_metrics.csv')

st.set_page_config(
    page_title='MEMS Calibration Studio',
    page_icon='\u2728',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --bg-dark: #050507; --bg-panel: rgba(17,17,20,0.6); --glass: rgba(255,255,255,0.03); --glass-border: rgba(255,255,255,0.08); --accent-cyan: #00f3ff; --accent-purple: #a855f7; --accent-pink: #f472b6; --text-main: #f8fafc; --text-muted: #94a3b8; }
    .stApp { background: radial-gradient(circle at 50% 50%, #0f0f1a 0%, #050507 100%); background-attachment: fixed; font-family: 'Inter', sans-serif; color: var(--text-main); }
    .stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle at 20% 30%, rgba(168, 85, 247, 0.15) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(0, 243, 255, 0.1) 0%, transparent 40%); pointer-events: none; z-index: -1; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .main { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1400px; }
    .app-header { background: linear-gradient(135deg, rgba(17,17,20,0.8) 0%, rgba(30,30,40,0.6) 100%); backdrop-filter: blur(20px); border: 1px solid var(--glass-border); border-radius: 1.5rem; padding: 2.5rem 2rem; margin-bottom: 2rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); position: relative; overflow: hidden; }
    .app-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)); }
    .app-header h1 { font-family: 'Space Grotesk', sans-serif; color: #ffffff; font-size: 2.25rem; font-weight: 700; margin: 0 0 0.5rem 0; letter-spacing: -0.02em; background: linear-gradient(135deg, #fff 0%, var(--accent-cyan) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .app-header p { color: var(--text-muted); margin: 0; font-size: 1rem; font-weight: 400; }
    .glass-card { background: var(--glass); backdrop-filter: blur(16px); border: 1px solid var(--glass-border); border-radius: 1rem; padding: 1.5rem; transition: all 0.3s ease; position: relative; overflow: hidden; }
    .glass-card:hover { border-color: rgba(168, 85, 247, 0.4); transform: translateY(-2px); box-shadow: 0 20px 40px -10px rgba(168, 85, 247, 0.15); }
    .glass-card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); }
    .metric-bento { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
    .bento-card { background: var(--glass); backdrop-filter: blur(16px); border: 1px solid var(--glass-border); border-radius: 1rem; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.5rem; transition: all 0.3s ease; }
    .bento-card:hover { border-color: rgba(0, 243, 255, 0.4); transform: translateY(-3px); box-shadow: 0 20px 40px -10px rgba(0, 243, 255, 0.15); }
    .bento-label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
    .bento-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0; }
    .bento-delta { font-size: 0.75rem; color: #10b981; font-weight: 500; }
    .bento-wide { grid-column: span 2; }
    .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.125rem; font-weight: 600; color: #ffffff; margin: 0 0 1.25rem 0; display: flex; align-items: center; gap: 0.75rem; }
    .section-title::before { content: ''; width: 4px; height: 24px; background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple)); border-radius: 2px; }
    .stButton > button { background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-cyan) 100%); color: white; border: none; border-radius: 0.75rem; padding: 0.75rem 1.5rem; font-weight: 600; font-family: 'Space Grotesk', sans-serif; width: 100%; transition: all 0.3s ease; box-shadow: 0 10px 20px -5px rgba(168, 85, 247, 0.3); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 20px 30px -10px rgba(168, 85, 247, 0.4); filter: brightness(1.1); }
    .stSelectbox > div > div { background: var(--glass); border: 1px solid var(--glass-border); border-radius: 0.75rem; color: var(--text-main); }
    .stNumberInput > div > div > input { background: var(--glass); border: 1px solid var(--glass-border); border-radius: 0.75rem; color: var(--text-main); }
    .stDataFrame { background: var(--glass); border: 1px solid var(--glass-border); border-radius: 1rem; overflow: hidden; }
    [data-testid="stSidebar"] { background: rgba(17, 17, 20, 0.4); backdrop-filter: blur(16px); border-right: 1px solid var(--glass-border); }
    .footer { margin-top: 3rem; padding: 2rem; border-top: 1px solid var(--glass-border); color: var(--text-muted); font-size: 0.875rem; text-align: center; }
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="app-header">
    <h1>\u2728 MEMS Calibration Studio</h1>
    <p>Predict temperature-dependent calibration parameters from sensor readings</p>
</div>
''', unsafe_allow_html=True)

if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
else:
    metrics_df = pd.DataFrame()

available_models = {}
if os.path.isdir(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith('_rf.pkl'):
            target = f.replace('_rf.pkl', '')
            available_models[target] = os.path.join(MODEL_DIR, f)

clean_path = os.path.join(BASE_DIR, 'data', 'sensor_calibration_clean.csv')
if os.path.exists(clean_path):
    feat_cols = [c for c in FEATURE_BASE + PCM_COLS + POS_COLS if c in pd.read_csv(clean_path, nrows=0).columns]
else:
    feat_cols = FEATURE_BASE + PCM_COLS + POS_COLS

missing_assets = []
if not available_models:
    missing_assets.append('trained models in models/*_rf.pkl')
if not os.path.exists(clean_path):
    missing_assets.append('data/sensor_calibration_clean.csv')
if missing_assets:
    st.warning(
        'Some deployment files are missing: ' + ', '.join(missing_assets) +
        '. The dashboard will load, but predictions or dataset-dependent features '
        'will be unavailable until these files are supplied.'
    )

st.sidebar.markdown('''
<div style="padding: 1.5rem 1rem;">
    <h2 style="color: #ffffff; font-size: 1.125rem; margin: 0 0 1.5rem 0; font-family: 'Space Grotesk', sans-serif;">Navigation</h2>
</div>
''', unsafe_allow_html=True)

page = st.sidebar.radio(
    'Go to',
    ['Home', 'Predictions', 'Model Performance', 'EDA'],
    label_visibility='collapsed'
)
if page == 'Home':
    if not metrics_df.empty:
        avg_r2 = metrics_df['r2'].mean()
        best_r2 = metrics_df['r2'].max()
        best_target = metrics_df.loc[metrics_df['r2'].idxmax(), 'target']
        n_models = len(metrics_df)
        st.markdown('''
        <div class="metric-bento">
            <div class="bento-card">
                <div class="bento-label">Models Trained</div>
                <div class="bento-value">''' + str(n_models) + '''</div>
            </div>
            <div class="bento-card">
                <div class="bento-label">Avg R² Score</div>
                <div class="bento-value">''' + f"{avg_r2:.3f}" + '''</div>
            </div>
            <div class="bento-card bento-wide">
                <div class="bento-label">Best R² Score</div>
                <div class="bento-value">''' + f"{best_r2:.3f}" + '''</div>
                <div class="bento-delta">''' + best_target + '''</div>
            </div>
            <div class="bento-card">
                <div class="bento-label">Dataset Size</div>
                <div class="bento-value">11,060</div>
                <div class="bento-delta">2,174 train / 8,886 test</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">\u1f4ca Overview</div>', unsafe_allow_html=True)
    st.write('This dashboard demonstrates an ML-based sensor calibration system for MEMS accelerometers.')
    st.write('')
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write('**Dataset:** 11,060 dies with 67 features including PCM, position, and sensor readings.')
        st.write('**Train/Test Split:** characterized_3temp == True (2,174 train) vs False (8,886 test)')
        st.write('**Target Groups:** TCO (3), TCS (3), TCO2 (3), Reflow shift (3), Cal codes (9)')
    with col_info2:
        st.write('**Features:** 23 columns from FEATURE_BASE, PCM, and position')
        st.write('**Algorithm:** Random Forest Regressor (100 estimators, max depth 15)')
        st.write('**Evaluation:** MAE, RMSE, R² on held-out test set')
    st.markdown('</div>', unsafe_allow_html=True)
    if not metrics_df.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">\u1f4c8 Model Performance Summary</div>', unsafe_allow_html=True)
        st.dataframe(metrics_df.round(4), use_container_width=True)
        st.write('')
        st.info('Key insight: Cal offset codes achieve R² > 0.98, TCS targets R² ~ 0.94, TCO targets R² ~ 0.82')
        st.markdown('</div>', unsafe_allow_html=True)
elif page == 'Predictions':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">\u1f4ca Calibration Parameter Prediction</div>', unsafe_allow_html=True)
    st.write('Enter sensor readings to predict calibration parameters.')
    st.write('')
    col1, col2 = st.columns(2)
    input_values = {}
    with col1:
        st.markdown('**Feature Set 1**')
        for c in feat_cols[:12]:
            input_values[c] = st.number_input(c, value=0.0, format='%.4f', key=f'inp_{c}_1')
    with col2:
        st.markdown('**Feature Set 2**')
        for c in feat_cols[12:]:
            input_values[c] = st.number_input(c, value=0.0, format='%.4f', key=f'inp_{c}_2')
    st.write('')
    target_group = st.selectbox(
        'Target group',
        ['TCO', 'TCS', 'TCO2', 'Reflow shift', 'Cal codes'],
        key='target_group_select'
    )
    group_map = {'TCO': TCO_TARGETS, 'TCS': TCS_TARGETS, 'TCO2': TCO2_TARGETS, 'Reflow shift': REFLOW_TARGETS, 'Cal codes': CAL_CODE_TARGETS}
    selected_targets = group_map[target_group]
    if st.button('Predict', key='predict_btn'):
        X_input = pd.DataFrame([input_values])
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">\u1f4c8 Predictions</div>', unsafe_allow_html=True)
        pred_cols = st.columns(min(len(selected_targets), 3))
        for i, target in enumerate(selected_targets):
            if target in available_models:
                model = joblib.load(available_models[target])
                pred = model.predict(X_input)[0]
                with pred_cols[i % 3]:
                    st.metric(target, f'{pred:.4f}')
            else:
                st.warning(f'No model for {target}')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
elif page == 'Model Performance':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">\u1f4c8 Model Performance Metrics</div>', unsafe_allow_html=True)
    if not metrics_df.empty:
        st.dataframe(metrics_df.round(4), use_container_width=True)
        st.write('')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df['target'],
            y=metrics_df['r2'],
            name='R²',
            marker_color='#00f3ff',
            marker_line_color='#a855f7',
            marker_line_width=1
        ))
        fig.update_layout(
            title='R² Score by Target',
            xaxis_title='Target',
            yaxis_title='R²',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font=dict(size=16, color='#ffffff'),
            height=450,
            margin=dict(l=20, r=20, t=60, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write('')
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=metrics_df['target'],
            y=metrics_df['mae'],
            name='MAE',
            marker_color='#f472b6',
            marker_line_color='#a855f7',
            marker_line_width=1
        ))
        fig2.update_layout(
            title='MAE by Target',
            xaxis_title='Target',
            yaxis_title='MAE',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font=dict(size=16, color='#ffffff'),
            height=450,
            margin=dict(l=20, r=20, t=60, b=100)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning('No metrics found. Run train.py first.')
    st.markdown('</div>', unsafe_allow_html=True)
elif page == 'EDA':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">\u1f50d Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.write('Saved EDA figures:')
    fig_files = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')] if os.path.exists(FIG_DIR) else []
    for fig_file in sorted(fig_files):
        img_path = os.path.join(FIG_DIR, fig_file)
        st.image(img_path, caption=fig_file, use_container_width=True)
        st.write('')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('''
<div class="footer">
    MEMS Calibration Studio &bull; Glassmorphism Dark Theme &bull; Powered by Streamlit &amp; Scikit-Learn
</div>
''', unsafe_allow_html=True)