# Seattle Weather Classification using Artificial Neural Networks

This project uses a feedforward Artificial Neural Network (ANN) built with PyTorch to classify Seattle weather conditions based on numerical weather features such as temperature, wind speed, and precipitation.

---

## Dataset

- **Source**: [Kaggle – Seattle Weather by Ananth R. (2022)](https://www.kaggle.com/datasets/ananthr1/weather-prediction)
- The dataset contains historical weather data for Seattle and was cleaned prior to training.
- Target classes include: `drizzle`, `fog`, `rain`, `snow`, and `sun`.

---

## Project Structure

- `Seattle_Weather_Cleaned.csv`: Cleaned dataset used for training.
- `main.py`: Core training and evaluation script.
- `README.md`: This file.

---

## Model Overview

- Model: Feedforward ANN (`WeatherNet`) with two hidden layers and batch normalization.
- Optimizer: Adam
- Loss Function: Cross-Entropy Loss
- Evaluation: Accuracy, Classification Report, Confusion Matrix

---

## Results

The model was trained for 50 epochs and evaluated on a 30% holdout validation set.  
**Performance Metrics:**
- Accuracy: ~87% (may vary slightly per run)
- Precision, recall, and F1-scores reported per class.

Confusion matrix and classification report are plotted after training.

---

## How to Run

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/yourusername/seattle-weather-ann.git
   cd seattle-weather-ann
