📈 Apple Stock Price Prediction

1️⃣ Problem Statement
Forecast Apple Inc. stock prices using historical data with a time series model.

2️⃣ Dataset
- File: Project DATASET.csv
- Columns: Date, Open, High, Low, Close, Volume
- 
3️⃣ EDA
- Converted Date to datetime & sorted
- check for null and missing value 
- Visualized closing prices for trends & seasonality
  
4️⃣ Model Choice
- SARIMA selected for capturing trend + seasonal patterns in financial time series.
  
5️⃣ Preprocessing
- Removed missing values
- Removed outlires 
- Target: Close Price
- Applied scaling

6️⃣ Model Training
- Train/test split
- Tuned SARIMA parameters
- Saved model

7️⃣ Streamlit App
- Interactive web app (app.py)
- Shows historical data + future predictions
- Visualizations for better insights
  
8️⃣ Deployment
- Deployed on Streamlit Cloud

9️⃣ Repository Structure
- Apple stock price prediction.ipynb → EDA & training
- app.py → Streamlit frontend
- Apple stock price prediction PPT→ Presentation 
- Project  DATASET.csv → Dataset
  
🔟 Achievements
✅ End-to-end forecasting pipeline
✅ SARIMA statistical modeling
✅ Real-time Streamlit app
✅ Successful cloud deployment


