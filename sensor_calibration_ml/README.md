# MEMS Sensor Calibration ML — Deployment Guide

## Overview

This project predicts temperature-dependent sensor calibration parameters using Random Forest models. The Streamlit dashboard provides a glassmorphism dark-themed UI for model predictions and performance visualization.

## Project Structure

```
sensor_calibration_ml/
├── dashboard/
│   └── app.py              # Streamlit dashboard (glassmorphism UI)
├── src/
│   └── data.py             # Data module with feature/target constants
├── data/
│   └── sensor_calibration_clean.csv
├── models/
│   └── *_rf.pkl            # Trained RandomForest models (21 files)
├── outputs/
│   ├── figures/            # EDA figures (PNG)
│   └── model_metrics.csv   # Model evaluation metrics
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku/Railway deployment
├── runtime.txt             # Python version (Heroku)
├── .streamlit/
│   └── config.toml         # Streamlit server config
└── .gitignore              # Git ignore rules
```

## Deployment Options

---

### Vercel compatibility

This project is a Streamlit app, not a Vercel Python serverless function. Vercel's
`PYTHON_ENTRYPOINT_NOT_FOUND` error is expected for this app; adding a Vercel
configuration file will not make Streamlit's persistent server and WebSocket
interface run there. Use Streamlit Community Cloud, Railway, or Render instead.

### Option 1: Streamlit Cloud (Recommended — Free)

1. Push your code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the `master` branch of this repository
5. Set the main file path to `sensor_calibration_ml/dashboard/app.py`
6. Click **Deploy**

Your app will be live at `https://your-app.streamlit.app`

---

### Option 2: Heroku

1. Install the Heroku CLI: [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

2. Login and create app:
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. Deploy:
   ```bash
   git add .
   git commit -m "Deploy MEMS calibration dashboard"
   git push heroku main
   ```

4. Open your app:
   ```bash
   heroku open
   ```

---

### Option 3: Railway

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Connect GitHub and select your repository
4. Railway auto-detects the `Procfile` and `requirements.txt`
5. Click **Deploy**

---

### Option 4: Render

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create a new **Web Service**
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run dashboard/app.py --server.port $PORT --server.headless true`
   - **Runtime:** Python 3

---

### Option 5: Docker

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8502", "--server.headless=true"]
```

Build and run:
```bash
docker build -t mems-calibration .
docker run -p 8502:8502 mems-calibration
```

Then visit: `http://localhost:8502`

---

### Option 6: Local Deployment (Production)

For local production use with a process manager:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with nohup (Linux/Mac)
nohup streamlit run dashboard/app.py --server.port 8502 --server.headless true &

# Or use PM2 (Node.js process manager, works with any script)
npm install -g pm2
pm2 start "streamlit run dashboard/app.py --server.port 8502 --server.headless true" --name mems-calibration
```

---

## Important Notes

### Model Files (.pkl)
The trained models (`models/*_rf.pkl`) are published using Git LFS. Keep Git LFS
enabled when cloning or deploying so the actual model files, rather than pointer
files, are available to the app.

### Data Files (.csv)
`data/sensor_calibration_clean.csv` is published using Git LFS along with the
models. The model metrics CSV and EDA chart images under `outputs/` are included
in the repository because the dashboard uses them for its Model Performance and
EDA pages. Other generated output files and the original source dataset remain
excluded.

### Port Configuration
- The app is configured to run on port **8502** by default
- For cloud platforms, the `$PORT` environment variable is used automatically
- Update `.streamlit/config.toml` if you need a different port

### Environment Variables
Create a `.env` file if you need secrets:
```bash
SECRET_KEY=your-secret-key
```

Streamlit secrets can also be configured in `.streamlit/secrets.toml`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in `.streamlit/config.toml` or use `$PORT` env var |
| Model files not found | Ensure `models/*.rf.pkl` are included in deployment |
| Dependencies missing | Run `pip install -r requirements.txt` |
| Streamlit warnings | Set `gatherUsageStats = false` in config.toml |

---

## Support

For issues with deployment, check:
1. Streamlit logs in your cloud provider dashboard
2. That `requirements.txt` lists all needed packages
3. That `models/` and `data/` directories are accessible at runtime