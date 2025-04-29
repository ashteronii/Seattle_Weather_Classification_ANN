# Seattle Weather Classification using Artificial Neural Networks

This project uses a feedforward Artificial Neural Network (ANN) built with PyTorch to classify Seattle weather conditions based on numerical weather features such as temperature, precipitation, and wind speed.

---

## Dataset

- Ananth R. (2022). *WEATHER PREDICTION*. Kaggle. https://www.kaggle.com/datasets/ananthr1/weather-prediction
- The dataset contains historical weather data for Seattle and was cleaned prior to training.
- Target classes include: `drizzle`, `fog`, `rain`, `snow`, and `sun`.

---

## Project Structure

- `Seattle_Weather_Cleaned.csv`: Cleaned dataset used for training.
- `main.py`: Core training and evaluation script.
- `README.md`: This file.

---

## Model Overview

The model is a feedforward Artificial Neural Network (ANN) designed to classify Seattle weather conditions. It consists of:

- **Input Layer**: 4 features (temperature, precipitation, wind speed, and humidity).
- **Hidden Layer 1**: 16 neurons with Batch Normalization and ReLU activation.
- **Hidden Layer 2**: 8 neurons with Batch Normalization and ReLU activation.
- **Output Layer**: A softmax layer with 5 neurons corresponding to the target classes: `drizzle`, `fog`, `rain`, `snow`, and `sun`.

The model is trained using the **Adam optimizer** and the **Cross-Entropy Loss function**.

---

## Results

The model was trained for 50 epochs and evaluated on a 30% holdout validation set.  
**Performance Metrics:**
- Accuracy: ~83% (may vary slightly per run)
- Precision, recall, and F1-scores reported per class.

Confusion matrix and classification report are plotted after training.

---

## How to Run

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/yourusername/seattle-weather-ann.git
   cd seattle-weather-ann
2. Create and activate a virtual environment (optional, but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

---

## Dependencies

- Python 3.8+
- torch
- pandas
- scikit-learn
- matplotlib

