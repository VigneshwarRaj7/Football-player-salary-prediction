# Salary Prediction Regression Model

## Overview
This project implements a **Salary Prediction Model** using **Linear Regression** and **Polynomial Regression**. The dataset consists of football players' salaries and attributes such as age, club, league, nationality, position, and appearances.

## Dataset
- **Source:** `data/SalaryPrediction.csv`
- **Number of Rows:** 3,900
- **Number of Columns:** 8
- **Target Variable:** `Wage` (Salary)
- **Features:**
  - Age
  - Club
  - League
  - Nation
  - Position
  - Apps (Appearances)
  - Caps (Caps)

## Dependencies
Install required Python packages:
```sh
pip install pandas numpy scikit-learn matplotlib pydotplus
```

## Data Preprocessing
1. **Load dataset:**
   ```python
   df = pd.read_csv('data/SalaryPrediction.csv')
   ```
2. **Display dataset info:**
   ```python
   print(df.shape)
   print(df.head())
   print(df.describe())
   ```
3. **One-hot encode categorical variables:**
   ```python
   X = pd.get_dummies(df.drop('Wage', axis=1))
   y = df['Wage']
   ```
4. **Feature scaling and correlation analysis:**
   ```python
   scatter_matrix(df)
   plt.show()
   ```

## Model Training
### **Linear Regression**
1. **Split dataset into train/test:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
   ```
2. **Train Linear Regression model:**
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
3. **Model evaluation:**
   ```python
   from sklearn.metrics import mean_squared_error
   y_hat = model.predict(X_test)
   print("R-squared:", model.score(X, y))
   print("RMSE:", mean_squared_error(y_test, y_hat, squared=False))
   ```

### **Polynomial Regression**
1. **Add polynomial features:**
   ```python
   X['AgeSquared'] = np.square(df['Age'])
   X['AppsSquared'] = np.square(df['Apps'])
   X['CapsSquared'] = np.square(df['Caps'])
   ```
2. **Train Polynomial Regression model:**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
3. **Evaluate model performance:**
   ```python
   y_hat = model.predict(X_test)
   print("R-squared:", model.score(X, y))
   print("RMSE:", mean_squared_error(y_test, y_hat, squared=False))
   ```

## Results
| Model | R-squared | RMSE |
|--------|-----------|------|
| Linear Regression | 0.6405 | 1,312,531 |
| Polynomial Regression | 0.6405 | 1,312,531 |

## Conclusion
- The **Linear Regression model** performs well but has room for improvement.
- **Polynomial Regression** did not significantly improve performance.
- Further improvements could include **feature selection, normalization, and hyperparameter tuning**.

## Next Steps
- Implement **Decision Tree Regression** for non-linear relationships.
- Test **Random Forest** and **XGBoost** for better performance.
- Optimize hyperparameters to minimize RMSE.

## License
This project is licensed under the MIT License.

---
Developed by **Vigneshwar Raj**

