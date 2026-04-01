# 🚗 Vehicle Virtual Sensor  
### Road Slope & Vehicle Mass Prediction using Machine Learning

---

## 📌 Introduction
This project builds a **virtual sensor** to estimate:
- Road slope  
- Vehicle mass  

using existing vehicle signals such as speed and engine torque.

Instead of installing expensive physical sensors, this approach leverages **data and machine learning** to infer hidden variables.

---

## 🚀 Demo
Run the Streamlit app locally: https://vehicle-virtual-sensor.streamlit.app/


👉 Demo will allow users to:
- Input vehicle signals  
- Predict slope and mass in real-time  

---

## 🚀 Features
- 📊 Exploratory Data Analysis (EDA)  
- 🧠 Feature Engineering based on domain understanding  
- 🤖 Machine Learning models:
  - Random Forest (classification)  
  - KNN (regression)  
- ⚙️ Data preprocessing & scaling  
- 📈 Model evaluation and comparison  
- 🌐 (Coming soon) Interactive demo with Streamlit  

---

## 📊 Dataset
- 11 vehicle signals  
- 9 input features  
- 2 targets:
  - `RoadSlope_100ms` (regression)  
  - `Vehicle_Mass` (classification: 38t vs 49t)  

⚠️ Note:
- No time-series → each row is independent  

---

## 🔍 Key Insights
- Torque-related features strongly correlate with road slope  
- Vehicle speed negatively correlates with vehicle mass  
- Feature engineering significantly improves model performance  

💡 Interpretation:
- Higher torque → uphill or heavier vehicle  
- Lower speed → heavier vehicle  

---

## ⚙️ Tech Stack
- Python  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Streamlit (for demo)

---

## 🤖 Models Used

### 🔹 Classification (Vehicle Mass)
- Random Forest (best performance)  
- Logistic Regression  
- SVM  

---

### 🔹 Regression (Road Slope)
- KNN Regressor (best performance)  
- SVR  
- Gradient Boosting  

---

## 📈 Results

| Task | Score |
|------|------|
| Classification | ~0.998 |
| Regression | ~0.79 |
| Overall | **0.865** |

---

## 📁 Project Structure

---

## 🧠 Key Learning
- Understanding data is more important than model complexity  
- Feature engineering plays a critical role  
- Data can replace hardware in real-world systems  

---

## 🎯 Future Improvements
- Improve regression performance  
- Add real-time prediction pipeline  
- Deploy online demo  
- Integrate dashboard (Power BI / web app)

---

## 👤 Author
- 

---
