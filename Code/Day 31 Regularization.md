```python
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
# Suppress Warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')
```


```python
# read dataset
dataset = pd.read_csv('./melbourne_housing_price.csv')
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbotsford</td>
      <td>68 Studley St</td>
      <td>2</td>
      <td>h</td>
      <td>NaN</td>
      <td>SS</td>
      <td>Jellis</td>
      <td>3/09/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8014</td>
      <td>144.9958</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>85 Turner St</td>
      <td>2</td>
      <td>h</td>
      <td>1480000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>3/12/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.7996</td>
      <td>144.9984</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>25 Bloomburg St</td>
      <td>2</td>
      <td>h</td>
      <td>1035000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8079</td>
      <td>144.9934</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abbotsford</td>
      <td>18/659 Victoria St</td>
      <td>3</td>
      <td>u</td>
      <td>NaN</td>
      <td>VB</td>
      <td>Rounds</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8114</td>
      <td>145.0116</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>5 Charles St</td>
      <td>3</td>
      <td>h</td>
      <td>1465000.0</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>4/03/2017</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8093</td>
      <td>144.9944</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
dataset.nunique()
```




    Suburb             351
    Address          34009
    Rooms               12
    Type                 3
    Price             2871
    Method               9
    SellerG            388
    Date                78
    Distance           215
    Postcode           211
    Bedroom2            15
    Bathroom            11
    Car                 15
    Landsize          1684
    BuildingArea       740
    YearBuilt          160
    CouncilArea         33
    Lattitude        13402
    Longtitude       14524
    Regionname           8
    Propertycount      342
    dtype: int64




```python
# let's use limited columns which makes more sense for serving our purpose
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Distance</th>
      <th>CouncilArea</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>SS</td>
      <td>Jellis</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>1480000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1035000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>u</td>
      <td>VB</td>
      <td>Rounds</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1465000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.shape
```




    (34857, 15)




```python
dataset.isna().sum()
```




    Suburb               0
    Rooms                0
    Type                 0
    Method               0
    SellerG              0
    Regionname           3
    Propertycount        3
    Distance             1
    CouncilArea          3
    Bedroom2          8217
    Bathroom          8226
    Car               8728
    Landsize         11810
    BuildingArea     21115
    Price             7610
    dtype: int64




```python
# Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
# like 0 for Propertycount, Bedroom2 will refer to other class of NA values
# like 0 for Car feature will mean that there's no car parking feature with house
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

# other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
# using Lasso and Ridge Regression
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
```


```python
dataset.dropna(inplace=True)
```


```python
dataset.shape
```




    (27244, 15)




```python
dataset = pd.get_dummies(dataset, drop_first=True)
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Propertycount</th>
      <th>Distance</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>Price</th>
      <th>Suburb_Aberfeldie</th>
      <th>...</th>
      <th>CouncilArea_Moorabool Shire Council</th>
      <th>CouncilArea_Moreland City Council</th>
      <th>CouncilArea_Nillumbik Shire Council</th>
      <th>CouncilArea_Port Phillip City Council</th>
      <th>CouncilArea_Stonnington City Council</th>
      <th>CouncilArea_Whitehorse City Council</th>
      <th>CouncilArea_Whittlesea City Council</th>
      <th>CouncilArea_Wyndham City Council</th>
      <th>CouncilArea_Yarra City Council</th>
      <th>CouncilArea_Yarra Ranges Shire Council</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>160.2564</td>
      <td>1480000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0000</td>
      <td>1035000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0000</td>
      <td>1465000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>94.0</td>
      <td>160.2564</td>
      <td>850000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>142.0000</td>
      <td>1600000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 745 columns</p>
</div>




```python
X = dataset.drop('Price', axis=1)
y = dataset['Price']
```


```python
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)
```


```python

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X, train_y)
```


```python

reg.score(test_X, test_y)
```




    0.13853683161569663




```python
reg.score(train_X, train_y)
```




    0.6827792395792723



Here training score is 68% but test score is 13.85% which is very low

Normal Regression is clearly overfitting the data, let's try other models
## Using Lasso (L1 Regularized) Regression Model


```python
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
```




    Lasso(alpha=50, max_iter=100, tol=0.1)




```python
lasso_reg.score(test_X, test_y)
```




    0.6636111369404488




```python

lasso_reg.score(train_X, train_y)
```




    0.6766985624766824



## Using Ridge (L2 Regularized) Regression Model


```python
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
```




    Ridge(alpha=50, max_iter=100, tol=0.1)




```python
ridge_reg.score(test_X, test_y)
```




    0.6670848945194976




```python
ridge_reg.score(train_X, train_y)
```




    0.6622376739684327


