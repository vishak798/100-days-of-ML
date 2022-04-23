# Eclat

## Importing the libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Datapreprocessing


```python
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])
```

## Training the Eclat model on the dataset


```python
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift = 3, min_length = 2, max_length = 2)
```

## Visualising the results

### Displaying the first results coming directly from the output of the apriori function


```python
results = list(rules)
```

### Putting the results well organised into a Pandas DataFrame


```python
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
```

### Displaying the results sorted by descending supports


```python
resultsinDataFrame.nlargest(n = 10, columns = 'Support')
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
      <th>Product 1</th>
      <th>Product 2</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>herb &amp; pepper</td>
      <td>ground beef</td>
      <td>0.015998</td>
    </tr>
    <tr>
      <th>7</th>
      <td>whole wheat pasta</td>
      <td>olive oil</td>
      <td>0.007999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pasta</td>
      <td>escalope</td>
      <td>0.005866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mushroom cream sauce</td>
      <td>escalope</td>
      <td>0.005733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tomato sauce</td>
      <td>ground beef</td>
      <td>0.005333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pasta</td>
      <td>shrimp</td>
      <td>0.005066</td>
    </tr>
    <tr>
      <th>0</th>
      <td>light cream</td>
      <td>chicken</td>
      <td>0.004533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fromage blanc</td>
      <td>honey</td>
      <td>0.003333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>light cream</td>
      <td>olive oil</td>
      <td>0.003200</td>
    </tr>
  </tbody>
</table>
</div>


