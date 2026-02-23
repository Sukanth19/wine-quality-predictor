# 🍷 Wine Quality Detector

A web application to explore and analyse wine quality using statistical methods — no machine learning involved. Built with Python, Flask, Pandas, NumPy, Matplotlib, and Seaborn.

## 📸 Features

- 📊 **Exploratory Data Analysis** — quality distributions, correlation heatmaps, alcohol vs quality boxplots, feature histograms
- 🔍 **Wine Quality Checker** — enter your wine's chemical properties and get a quality score (3–9) based on z-score statistics
- 🍷 Supports both **Red** and **White** wine datasets
- 🎨 Dark-themed responsive UI

## 📁 Project Structure

```
wine-quality-predictor/
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── static/
│   ├── css/style.css
│   └── plots/          # auto-generated on first run
├── templates/
│   ├── index.html
│   ├── analysis.html
│   └── predict.html
├── app.py
├── analysis.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Sukanth19/wine-quality-predictor.git
cd wine-quality-predictor
```

### 2. Create and activate a virtual environment
```bash
python -m venv wine_env
source wine_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

## 📊 Dataset

Uses the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI Machine Learning Repository.

- `winequality-red.csv` — 1,599 red wine samples
- `winequality-white.csv` — 4,898 white wine samples

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality

## 🔍 How the Quality Checker Works

No machine learning is used. The app compares your wine's chemical values against the dataset using **z-scores** (standard deviations from the mean). Features known to positively correlate with quality (alcohol, sulphates, citric acid) raise the score; features that negatively correlate (volatile acidity, chlorides, density) lower it.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Flask | Web framework |
| Pandas | Data loading & stats |
| NumPy | Numerical computation |
| Matplotlib | Chart generation |
| Seaborn | Heatmaps |

## 📄 License

MIT
