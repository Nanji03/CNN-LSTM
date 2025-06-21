# CNN-LSTM for Multivariate Stock Price Forecasting

This notebook demonstrates a hybrid deep learning architecture — CNN-LSTM — for predicting Google’s stock closing prices based on 30-day windows of historical data. Unlike standard LSTMs, this model combines convolutional layers (to extract local temporal patterns) with recurrent LSTM layers (to capture long-term dependencies).

Objective:
- Use 5 key stock market features ("Open", "High", "Low", "Close", "Volume") as multivariate input
- Predict the next day’s "Close price" using CNN-LSTM
- Compare its performance using RMSE and MAE against baseline models

Model Architecture:
- TimeDistributed Conv1D: learns short-term local features from each sub-sequence
- MaxPooling1D + Flatten: reduces dimensionality and prepares data for LSTM
- LSTM Layer: captures long-term trends across sub-sequences
- Dropout: helps prevent overfitting
- Dense Output: generates the final price prediction

Data Preparation:
- Window Size: 30 days
- Sub-sequences: 5 (i.e., each sub-sequence = 6 days)
- Scaler: MinMaxScaler applied to all 5 features
- Train/Test Split: 90% for training, 10% for testing

Evaluation Results:
- RMSE: 4.21    
- MAE: 3.22    

While not outperforming the plain LSTM model, this experiment shows how hybrid models can be structured for multivariate time series forecasting.

Output:
The notebook includes a visualization comparing **predicted** vs **actual** closing prices over the test period.

Conclusion:
This CNN-LSTM architecture shows how combining local and sequential patterns can enhance forecasting performance. Further improvements could involve:
- Tuning convolutional kernel size
- Increasing LSTM units
- Removing MaxPooling
- Using alternative sequence lengths (e.g., 3×10 instead of 5×6)


