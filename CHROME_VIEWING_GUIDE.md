# 📊 Viewing ML Pipeline Results in Chrome

## 🚀 Quick Start

### Method 1: Use Your Existing Streamlit App
```bash
streamlit run src/streamlit_app.py
```
Then open Chrome and go to: `
http://localhost:8501`

### Method 2: MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Then open Chrome and go to: `http://localhost:5000`

## 🔧 Setup (if needed)

If you get MLflow tracker errors, run:
```bash
python src/setup_zenml.py
```

## 🎯 What You'll See

- **Streamlit App**: Your existing app with new "Results & MLflow" tab
- **MLflow UI**: Detailed experiment tracking, metrics, and artifacts
- **Chrome**: Both interfaces work perfectly in Chrome

## 📱 Chrome Extensions (Optional)

- **JSON Viewer**: Better JSON file viewing
- **Markdown Viewer**: View documentation
- **CSV Viewer**: Better dataset viewing

## 🎉 Success!

Your pipeline results will be visible in Chrome through either interface!
