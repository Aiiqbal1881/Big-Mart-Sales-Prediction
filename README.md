# Big-Mart-Sales-Prediction
# Big Mart Sales Prediction

## Overview
This project aims to predict sales for Big Mart stores using Machine Learning techniques. The dataset contains information about various products, stores, and their attributes. The goal is to build a regression model to forecast sales based on historical data.

## Technologies Used
- **Python**
- **Machine Learning (ML)**
- **XGBoost Regressor**
- **train_test_split** (from scikit-learn)
- **Pandas, NumPy, Matplotlib, Seaborn** (for data preprocessing and visualization)

## Dataset
The dataset consists of the following key features:
- `Item_Identifier`: Unique product ID
- `Item_Weight`: Weight of the product
- `Item_Fat_Content`: Low Fat or Regular
- `Item_Visibility`: The percentage of total display area of all products in a store allocated to this product
- `Item_Type`: The category to which the product belongs
- `Item_MRP`: Maximum Retail Price
- `Outlet_Identifier`: Unique store ID
- `Outlet_Establishment_Year`: Year the store was established
- `Outlet_Size`: Size of the store (Small/Medium/Large)
- `Outlet_Location_Type`: Tier-wise location of the store
- `Outlet_Type`: Type of store (e.g., Supermarket/Grocery store)
- `Item_Outlet_Sales`: Sales of the product in a store (Target Variable)

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/big-mart-sales-prediction.git
cd big-mart-sales-prediction
pip install -r requirements.txt
```

## Model Training & Prediction
The project follows these key steps:

1. **Data Preprocessing**
   - Handling missing values
   - Feature engineering
   - Encoding categorical variables
   - Scaling numerical variables

2. **Train-Test Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

3. **Model Training using XGBoost Regressor**
   ```python
   from xgboost import XGBRegressor
   model = XGBRegressor()
   model.fit(X_train, y_train)
   ```

4. **Model Evaluation**
   ```python
   from sklearn.metrics import R Square Error
   y_pred = model.predict(X_test)
   print(f"R Squared Error: {r2_score}")
## Usage
Run the following command to train and evaluate the model:
```bash
python main.py
```

## Contributing
If you’d like to contribute, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
