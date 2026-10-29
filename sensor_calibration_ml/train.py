import os
import numpy as np
import pandas as pd
import joblib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data import load_raw, clean, split_train_test, FEATURE_BASE, PCM_COLS, POS_COLS, ALL_TARGETS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(MODEL_DIR, exist_ok=True)

df = load_raw()
df, info = clean(df)
tr, te = split_train_test(df)
print('Train:', tr.shape, 'Test:', te.shape)

feat_cols = [c for c in FEATURE_BASE + PCM_COLS + POS_COLS if c in df.columns]
print('Feature columns:', len(feat_cols))
print(feat_cols)

X_train = tr[feat_cols]
X_test = te[feat_cols]

results = []
models = {}
for target in ALL_TARGETS:
    if target not in df.columns:
        continue
    y_train = tr[target]
    y_test = te[target]
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((target, mae, rmse, r2))
    models[target] = model
    print(target + ': MAE=' + str(round(mae,4)) + ', RMSE=' + str(round(rmse,4)) + ', R2=' + str(round(r2,4)))
metrics_df = pd.DataFrame(results, columns=['target', 'mae', 'rmse', 'r2'])
metrics_df.to_csv(os.path.join(METRICS_DIR, 'model_metrics.csv'), index=False)
print('Metrics saved to', os.path.join(METRICS_DIR, 'model_metrics.csv'))

for target, model in models.items():
    joblib.dump(model, os.path.join(MODEL_DIR, target + '_rf.pkl'))
print('Models saved to', MODEL_DIR)
