# STNet
## Electronic Nose Gas Concentration Prediction System
This project leverages deep learning techniques to accurately predict NO and NO₂ gas concentrations using electronic nose sensor data. It integrates multiple time-series models (CNN, LSTM, TCN, Transformer) with attention mechanisms to enhance feature extraction, making it suitable for environmental monitoring, industrial safety, and other related applications.
## Project Background
Electronic noses are multi-sensor systems capable of real-time monitoring of environmental gas components. This project aims to use deep learning to infer NO and NO₂ concentrations from time-series sensor data, providing data-driven support for environmental monitoring.
Features
Multi-Model Support: Implements various deep learning architectures including CNN, LSTM, TCN, and Transformer for flexible model comparison and selection
Data Processing: Utilizes sliding window techniques for time-series feature extraction with normalization and standardization preprocessing
Attention Mechanism: Integrates CBAM attention modules to enhance the model's focus on critical features
Evaluation Metrics: Provides comprehensive evaluation using R² score, MAE, and other metrics with visualizations
Visualization: Generates training curves, scatter plots, and line graphs for result analysis

# Installation and Usage
## Environment Requirements
Python 3.8+
PyTorch 1.8+
NumPy, Pandas, Scikit-learn
Matplotlib, CSV
