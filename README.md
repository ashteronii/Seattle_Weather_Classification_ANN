# Seattle Weather Classification using Artificial Neural Networks

This project uses a feedforward Artificial Neural Network (ANN) built with PyTorch to classify Seattle weather conditions based on numerical weather features such as temperature, precipitation, and wind speed.

---

## Dataset

- Ananth R. (2022). *WEATHER PREDICTION*. Kaggle. https://www.kaggle.com/datasets/ananthr1/weather-prediction
- The dataset contains historical weather data for Seattle.
- Target classes include: `drizzle`, `fog`, `rain`, `snow`, and `sun`.

---

## Project Structure

- `Seattle_Weather_Classification_ANN.py`: Model training and evaluation script.
- `Seattle_Weather_Cleaned.csv`: Cleaned dataset used for training.
- `Seatle_Weather_Cleaned.py`: Data cleaning script.
- `seattle-weather.csv`: Raw dataset from Kaggle.
- `README.md`: This file.

---

## Model Overview

The model is a feedforward Artificial Neural Network (ANN) designed to classify Seattle weather conditions. It consists of:

- **Input Layer**: 4 features (max temperature, min temperature, precipitation, and wind speed).
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
   ```
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn torch matplotlib
   ```
3. Run the script:
   ```bash
   python main.py
   ```

---

## Dependencies

- Python 3.8+
- torch
- pandas
- scikit-learn
- matplotlib

