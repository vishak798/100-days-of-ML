# KFold Cross Validation 
#### Cross-validation is a statistical method used to estimate the performance (or accuracy) of machine learning models. It is used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited. In cross-validation, you make a fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.

## Importing the libraries


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt
```

## Importing the Dataset


```python
from sklearn.datasets import load_digits
digits = load_digits()
```

## Testing without Cross validation


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)
```

### Logistic Regression


```python
lr = LogisticRegression(solver = "liblinear", multi_class = "ovr")
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```




    0.9481481481481482



### SVM


```python
svm = SVC(gamma = "auto")
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
```




    0.40185185185185185



### Random Forest


```python
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```




    0.9629629629629629



## Using KFold


```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf
```




    KFold(n_splits=3, random_state=None, shuffle=False)




```python
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
```

    [3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]
    

## Use KFold for our digits example


```python
#Function for getting score of different models
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
```


```python
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_random_forest = []
for train_index, test_index in folds.split(digits.data, digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index],digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver="liblinear", multi_class="ovr"), X_train, X_test, y_train, y_test))
    
    scores_svm.append(get_score(SVC(kernel="rbf",gamma = "scale"), X_train, X_test, y_train, y_test))
    scores_random_forest.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
```


```python
print(scores_logistic)
print(scores_random_forest)
print(scores_svm)
```

    [0.8948247078464107, 0.9532554257095158, 0.9098497495826378]
    [0.9332220367278798, 0.9565943238731218, 0.9181969949916527]
    [0.9649415692821369, 0.9799666110183639, 0.9649415692821369]
    

## cross_val_score function


```python
from sklearn.model_selection import cross_val_score
```

### Logistic regression model performance using cross_val_score


```python
cross_val_score(LogisticRegression(solver="liblinear", multi_class="ovr"), digits.data, digits.target, cv=3)
```




    array([0.89482471, 0.95325543, 0.90984975])



### svm model performance using cross_val_score


```python
cross_val_score(SVC(gamma="scale"), digits.data, digits.target, cv=3)
```




    array([0.96494157, 0.97996661, 0.96494157])



### random forest performance using cross_val_score


```python
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)
```




    array([0.92320534, 0.94323873, 0.91318865])


