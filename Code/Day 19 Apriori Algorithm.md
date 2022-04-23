# Association Rule Mining Apriori Algorithm

## Importing the libraries


```python
!pip install apyori
```

    Collecting apyori
      Downloading apyori-1.1.2.tar.gz (8.6 kB)
    Building wheels for collected packages: apyori
      Building wheel for apyori (setup.py): started
      Building wheel for apyori (setup.py): finished with status 'done'
      Created wheel for apyori: filename=apyori-1.1.2-py3-none-any.whl size=5974 sha256=dd479fc9b5c0bd74f47ce63ecf98d427142db958dc75eca07ea4478e12730caf
      Stored in directory: c:\users\visha\appdata\local\pip\cache\wheels\32\2a\54\10c595515f385f3726642b10c60bf788029e8f3a1323e3913a
    Successfully built apyori
    Installing collected packages: apyori
    Successfully installed apyori-1.1.2
    


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
```

## Importing the dataset and Data Preprocessing


```python
df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
# Apriori Algorithm needs lists of items as input so we have to convert pandas dataframe to list
transactions = []
for i in range(0, len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,df.shape[1])])
```

## Training the Apriori model on the dataset


```python
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
```

## Visualising the results

### Displaying the first results coming directly from the output of the apriori function


```python
results = list(rules)
```


```python
results
```




    [RelationRecord(items=frozenset({'chicken', 'light cream'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]),
     RelationRecord(items=frozenset({'escalope', 'mushroom cream sauce'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mushroom cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)]),
     RelationRecord(items=frozenset({'escalope', 'pasta'}), support=0.005865884548726837, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, lift=4.700811850163794)]),
     RelationRecord(items=frozenset({'honey', 'fromage blanc'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), confidence=0.2450980392156863, lift=5.164270764485569)]),
     RelationRecord(items=frozenset({'herb & pepper', 'ground beef'}), support=0.015997866951073192, ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef'}), confidence=0.3234501347708895, lift=3.2919938411349285)]),
     RelationRecord(items=frozenset({'tomato sauce', 'ground beef'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomato sauce'}), items_add=frozenset({'ground beef'}), confidence=0.3773584905660377, lift=3.840659481324083)]),
     RelationRecord(items=frozenset({'light cream', 'olive oil'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'olive oil'}), confidence=0.20512820512820515, lift=3.1147098515519573)]),
     RelationRecord(items=frozenset({'olive oil', 'whole wheat pasta'}), support=0.007998933475536596, ordered_statistics=[OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'olive oil'}), confidence=0.2714932126696833, lift=4.122410097642296)]),
     RelationRecord(items=frozenset({'shrimp', 'pasta'}), support=0.005065991201173177, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'shrimp'}), confidence=0.3220338983050847, lift=4.506672147735896)])]



### Putting the results well organised into a Pandas DataFrame


```python
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
```

### Displaying the results non sorted


```python
resultsinDataFrame
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
      <th>Left Hand Side</th>
      <th>Right Hand Side</th>
      <th>Support</th>
      <th>Confidence</th>
      <th>Lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>light cream</td>
      <td>chicken</td>
      <td>0.004533</td>
      <td>0.290598</td>
      <td>4.843951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mushroom cream sauce</td>
      <td>escalope</td>
      <td>0.005733</td>
      <td>0.300699</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pasta</td>
      <td>escalope</td>
      <td>0.005866</td>
      <td>0.372881</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fromage blanc</td>
      <td>honey</td>
      <td>0.003333</td>
      <td>0.245098</td>
      <td>5.164271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>herb &amp; pepper</td>
      <td>ground beef</td>
      <td>0.015998</td>
      <td>0.323450</td>
      <td>3.291994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tomato sauce</td>
      <td>ground beef</td>
      <td>0.005333</td>
      <td>0.377358</td>
      <td>3.840659</td>
    </tr>
    <tr>
      <th>6</th>
      <td>light cream</td>
      <td>olive oil</td>
      <td>0.003200</td>
      <td>0.205128</td>
      <td>3.114710</td>
    </tr>
    <tr>
      <th>7</th>
      <td>whole wheat pasta</td>
      <td>olive oil</td>
      <td>0.007999</td>
      <td>0.271493</td>
      <td>4.122410</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pasta</td>
      <td>shrimp</td>
      <td>0.005066</td>
      <td>0.322034</td>
      <td>4.506672</td>
    </tr>
  </tbody>
</table>
</div>



### Displaying the results sorted by descending lifts


```python
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
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
      <th>Left Hand Side</th>
      <th>Right Hand Side</th>
      <th>Support</th>
      <th>Confidence</th>
      <th>Lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>fromage blanc</td>
      <td>honey</td>
      <td>0.003333</td>
      <td>0.245098</td>
      <td>5.164271</td>
    </tr>
    <tr>
      <th>0</th>
      <td>light cream</td>
      <td>chicken</td>
      <td>0.004533</td>
      <td>0.290598</td>
      <td>4.843951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pasta</td>
      <td>escalope</td>
      <td>0.005866</td>
      <td>0.372881</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pasta</td>
      <td>shrimp</td>
      <td>0.005066</td>
      <td>0.322034</td>
      <td>4.506672</td>
    </tr>
    <tr>
      <th>7</th>
      <td>whole wheat pasta</td>
      <td>olive oil</td>
      <td>0.007999</td>
      <td>0.271493</td>
      <td>4.122410</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tomato sauce</td>
      <td>ground beef</td>
      <td>0.005333</td>
      <td>0.377358</td>
      <td>3.840659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mushroom cream sauce</td>
      <td>escalope</td>
      <td>0.005733</td>
      <td>0.300699</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>herb &amp; pepper</td>
      <td>ground beef</td>
      <td>0.015998</td>
      <td>0.323450</td>
      <td>3.291994</td>
    </tr>
    <tr>
      <th>6</th>
      <td>light cream</td>
      <td>olive oil</td>
      <td>0.003200</td>
      <td>0.205128</td>
      <td>3.114710</td>
    </tr>
  </tbody>
</table>
</div>


