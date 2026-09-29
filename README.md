# 🌡️ Weather Temperature Prediction

Predicts hourly temperature (°C) from weather conditions (humidity, wind, pressure, visibility, precipitation type) and recent temperature history, using and comparing five regression models. Includes a Jupyter notebook for the full analysis and a Streamlit app for live predictions.

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://weather-temperature-prediction-ft3synrs8mhkil4mot7vmy.streamlit.app)

🔗 **[Try it live](https://weather-temperature-prediction-ft3synrs8mhkil4mot7vmy.streamlit.app)**

A Streamlit dashboard where you enter current weather conditions and recent temperature history, and get a predicted temperature back instantly.

[App screenshot](https://github.com/user-attachments/assets/72b19f84-fd51-4edf-8605-98481576be54)

## Project Structure

```
.
├── data
│   └── weatherHistory.csv        # Dataset (not included if too large — see Dataset section)
├── notebook
│   └── weather_prediction.ipynb  # Full notebook: EDA, cleaning, feature engineering, modeling
├── streamlit_app
│   └── app5.py                   # Streamlit app for live predictions
├── README.md
├── best_weather_model.joblib     # Saved best model + scaler + feature list
└── requirements.txt              # Python dependencies
```

## Dataset

~96,000 hourly weather observations, with columns including:

| Column | Description |
|---|---|
| `Formatted Date` | Timestamp of the observation |
| `Temperature (C)` | Target variable |
| `Humidity` | Relative humidity (0–1) |
| `Wind Speed (km/h)` | Wind speed |
| `Wind Bearing (degrees)` | Wind direction |
| `Visibility (km)` | Visibility distance |
| `Pressure (millibars)` | Atmospheric pressure |
| `Precip Type` | rain / snow |

Source: [Kaggle — Weather in Szeged 2006–2016](https://www.kaggle.com/datasets/budincsevity/szeged-weather)

## Approach

1. **EDA** — distributions, missing values, correlation heatmap, seasonal and precipitation patterns
2. **Data Cleaning** — dropped rows with missing `Precip Type`, removed the zero-variance `Loud Cover` column, dropped `Apparent Temperature (C)` (0.99 correlated with the target — would leak the answer)
3. **Feature Engineering** — added recent-history features (`Temp_lag1`, `Temp_lag3`, `Humidity_lag1`, `Temp_roll3`), since temperature changes slowly hour to hour
4. **Train/Test Split** — chronological split (train on the past, test on the future) to avoid leaking neighboring hours between train and test
5. **Modeling** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and a Deep Learning model (MLP neural network)
6. **Evaluation** — RMSE, MAE, R², train-vs-test overfitting check, 5-fold cross-validation
7. **Model Saving** — best model (by test RMSE) saved with `joblib`, bundled with its scaler and feature list

## Results

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| Linear Regression | 0.703 | 0.512 | 0.9940 |
| Decision Tree | 0.888 | 0.648 | 0.9905 |
| Random Forest | 0.752 | 0.545 | 0.9932 |
| Gradient Boosting | 0.720 | 0.537 | 0.9938 |
| **Deep Learning (MLP)** | **0.642** | **0.481** | **0.9950** |

**Best model: Deep Learning (MLP)** — lowest RMSE, highest R², and no meaningful overfitting (train/test R² gap < 0.01, confirmed with 5-fold cross-validation, std ≈ 0.0016).

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Addychauhan/Weather-Temperature-Prediction.git
cd Weather-Temperature-Prediction
pip install -r requirements.txt
```

### Run the notebook

[Open in Google Colab](https://colab.research.google.com/github/Addychauhan/Weather-Temperature-Prediction/blob/main/notebook/weather_prediction.ipynb)

### Run the Streamlit app

```bash
streamlit run streamlit_app/app5.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

## Notes

- If you retrain the model with a different scikit-learn version than what's installed when running the app, you may see an `InconsistentVersionWarning`. Match your scikit-learn version to the one used for training, or retrain locally with `joblib.dump(...)` to resolve it.
- The app dynamically adapts to whatever features are stored in `best_weather_model.joblib`, so it works with either the single-lag-feature model or the multi-lag-feature model from the notebook.

## License

MIT — feel free to use, modify, and share.
