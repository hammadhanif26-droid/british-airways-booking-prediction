# British Airways: Customer Booking Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A machine learning model that predicts whether a customer will **complete a flight booking**, and shows which factors drive that decision. I built it as part of the **British Airways Data Science Virtual Experience (Forage)**.

---

## 📌 Problem Statement

Airlines can't wait for customers to book; they need to reach them before they travel. The aim is to use booking-session data to predict which customers are likely to complete a booking, so marketing can target the right customers and markets.

## 📊 Dataset

50,000 customer booking records with 13 features and the target `booking_complete`. Only about **15%** of sessions end in a completed booking, so the target is imbalanced.

| Feature | Description |
|---|---|
| `num_passengers` | Number of passengers travelling |
| `sales_channel` | Channel used for the booking (Internet / Mobile) |
| `trip_type` | Round trip, one way or circle trip |
| `purchase_lead` | Days between booking date and travel date |
| `length_of_stay` | Days spent at the destination |
| `flight_hour` / `flight_day` | Departure hour and day of week |
| `route` | Origin → destination route |
| `booking_origin` | Country the booking was made from |
| `wants_extra_baggage` / `wants_preferred_seat` / `wants_in_flight_meals` | Add-on preferences |
| `flight_duration` | Total flight duration (hours) |

> The dataset (`customer_booking.csv`) is provided by British Airways through Forage and is included in this repo.

## 📂 Repository Structure

```
british-airways-booking-prediction/
├── British_Airways_Customer_Booking_Prediction.ipynb   # EDA → features → model → evaluation
├── customer_booking.csv                                # 50,000 booking records
└── README.md
```

## 🔬 Approach

1. **EDA:** compared booking rates by channel, trip type, day of week, route, booking origin and add-on preferences.
2. **Feature engineering:** split routes into origin and destination airports, and added `total_extras_wanted`, `is_last_minute`, `is_weekend_flight` and `long_trip`.
3. **Leak-free encoding:** split train/test *before* encoding. High-cardinality columns are frequency-encoded using the training set only, and low-cardinality columns are one-hot encoded.
4. **Model:** a Random Forest with depth and leaf-size limits to reduce overfitting, and `class_weight='balanced'` for the 85/15 imbalance.
5. **Validation:** 5-fold stratified cross-validation, then a single evaluation on the held-out 20% test set.

## 📈 Results

| Metric | Result |
|---|---|
| Cross-validated ROC-AUC | 0.769 ± 0.005 |
| **Test ROC-AUC** | **0.777** (vs. 0.50 random) |
| **Recall (bookings)** | **76%** of true bookers correctly flagged |
| Precision (bookings) | 28% |

## 💡 Key Insights

- **Geography drives the predictions.** Booking origin and route-related features make up about **65%** of the model's feature importance. For example, Malaysia-origin bookings convert at ~34%, compared with ~5% for Australia.
- Length of stay, flight duration and purchase lead contribute moderately.
- Day of week, trip type and individual add-ons contribute very little.

**Recommendation:** focus proactive marketing on high-converting origin markets and routes rather than on trip attributes.

**Next steps:** tune the decision threshold to match the business's precision/recall trade-off, try gradient boosting (XGBoost / LightGBM), and add time-based features.

## 🚀 How to Run

```bash
git clone https://github.com/hammadhanif26-droid/british-airways-booking-prediction.git
cd british-airways-booking-prediction
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook British_Airways_Customer_Booking_Prediction.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter
