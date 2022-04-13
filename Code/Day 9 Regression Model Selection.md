## Importing Libraries


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

## Importing the dataset


```python
df = pd.read_csv("Datas.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

## Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

# Training the Random Forest Regression model on the whole dataset


```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
```




    RandomForestRegressor(n_estimators=10, random_state=0)



## Predicting the Test set Results


```python
y_pred = regressor.predict(X_test)
print(y_pred)
```

    [454.333 440.702 442.089 ... 431.079 438.905 460.95 ]
    

## Evaluating the Model Performance


```python
from sklearn.metrics import r2_score
RandomForest = r2_score(y_test, y_pred)
```

# Training the Decision Tree Regression model on the whole dataset


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
regressor_1 = DecisionTreeRegressor(random_state = 0)
regressor_1.fit(X_train, y_train)
```




    DecisionTreeRegressor(random_state=0)



## Predicting the Test set Results



```python
y_pred = regressor_1.predict(X_test)
print(y_pred)
```

    [451.77 440.59 439.74 ... 430.21 441.61 459.1 ]
    

## Evaluating the Model Performance


```python
from sklearn.metrics import r2_score
DecisionTree = r2_score(y_test, y_pred)
```

# Support Vector Regression

### Importing the Dataset


```python
df = pd.read_csv("Datas.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y),1)
print(y)
```

    [[463.26]
     [444.37]
     [488.56]
     ...
     [429.57]
     [435.74]
     [453.28]]
    

### Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
print(y_train)
```

    [[-0.79803286]
     [ 0.55355384]
     [ 0.93662073]
     ...
     [ 1.47045054]
     [ 0.9336876 ]
     [ 1.18007059]]
    

## Training the SVR model on the Training set


```python
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X_train, y_train)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    




    SVR()



## Predicting the Test set Results


```python
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
```

## Evaluating the Model Performance


```python
from sklearn.metrics import r2_score
Svr = r2_score(y_test, y_pred)
```

# Polynomial Regression


### Importing the Dataset


```python
df = pd.read_csv("Datas.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

### Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

### Feature Scaling and Polynomial regression model Training 


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
```




    LinearRegression()



### Predicting the Test set Results


```python
y_pred = poly_reg.predict(poly.transform(X_test))
```

### Evaluating the Model Performance


```python
from sklearn.metrics import r2_score
ply_reg = r2_score(y_test, y_pred)
```

# Multiple Linear Regression 

## Importing the Dataset


```python
df = pd.read_csv("Datas.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
```

## Splitting the Dataset


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

## Training the Multiple Linear Regression model on the Training set


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```




    LinearRegression()



## Predicting the Test set results


```python
y_pred = regressor.predict(X_test)
```

## Evaluating the Model Performance


```python
from sklearn.metrics import r2_score
lin_reg = r2_score(y_test, y_pred)
```

# The best model


```python
dic = {lin_reg :"Multiple Linear Regression",
      ply_reg : "Polynomial Regression",
      Svr : "Support Vector Regression",
      DecisionTree: "Decision Tree Regression",
      RandomForest : "Random Forest Regression"
      }
```


```python
print("The model with highest performance is ", dic.get(max(dic)))
```

    The model with highest performance is  Random Forest Regression
    
