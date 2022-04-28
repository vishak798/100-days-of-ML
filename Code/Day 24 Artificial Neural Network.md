# Artificial Neural Network

## Importing the Libraries


```python
import pandas as pd
import numpy as np
import tensorflow as tf
```


```python
tf.__version__
```




    '2.8.0'



## Part 1 - Data Preprocessing

### Importing the dataset


```python
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values
```


```python
print(X)
```

    [[619 'France' 'Female' ... 1 1 101348.88]
     [608 'Spain' 'Female' ... 0 1 112542.58]
     [502 'France' 'Female' ... 1 0 113931.57]
     ...
     [709 'France' 'Female' ... 0 1 42085.58]
     [772 'Germany' 'Male' ... 1 0 92888.52]
     [792 'France' 'Female' ... 1 0 38190.78]]
    


```python
print(y)
```

    [1 0 1 ... 1 1 0]
    

### Encoding categorical data

Label Encoding the "Gender" column


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```


```python
print(X)
```

    [[619 'France' 0 ... 1 1 101348.88]
     [608 'Spain' 0 ... 0 1 112542.58]
     [502 'France' 0 ... 1 0 113931.57]
     ...
     [709 'France' 0 ... 0 1 42085.58]
     [772 'Germany' 1 ... 1 0 92888.52]
     [792 'France' 0 ... 1 0 38190.78]]
    

One Hot Encoding the "Geography" column


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
```


```python
print(X)
```

    [[1.0 0.0 0.0 ... 1 1 101348.88]
     [0.0 0.0 1.0 ... 0 1 112542.58]
     [1.0 0.0 0.0 ... 1 0 113931.57]
     ...
     [1.0 0.0 0.0 ... 0 1 42085.58]
     [0.0 1.0 0.0 ... 1 0 92888.52]
     [1.0 0.0 0.0 ... 1 0 38190.78]]
    

### Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Part 2 - Building the ANN

### Initializing the ANN


```python
ann = tf.keras.models.Sequential()
```

### Adding the input layer and the first hidden layer


```python
ann.add(tf.keras.layers.Dense(units=6, activation="relu")) #relu is the rectifier activation function (Rectified Linear Unit)
```

### Adding the second hidden layer


```python
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
```

### Adding the output layer


```python
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

## Part 3 - Training the ANN

### Compiling the ANN


```python
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### Training the ANN on the Training set


```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

    Epoch 1/100
    250/250 [==============================] - 0s 486us/step - loss: 0.5356 - accuracy: 0.7421
    Epoch 2/100
    250/250 [==============================] - 0s 478us/step - loss: 0.4529 - accuracy: 0.7918
    Epoch 3/100
    250/250 [==============================] - 0s 490us/step - loss: 0.4387 - accuracy: 0.7960
    Epoch 4/100
    250/250 [==============================] - 0s 494us/step - loss: 0.4310 - accuracy: 0.8012
    Epoch 5/100
    250/250 [==============================] - 0s 486us/step - loss: 0.4217 - accuracy: 0.8055
    Epoch 6/100
    250/250 [==============================] - 0s 482us/step - loss: 0.4083 - accuracy: 0.8160
    Epoch 7/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3939 - accuracy: 0.8301
    Epoch 8/100
    250/250 [==============================] - 0s 474us/step - loss: 0.3826 - accuracy: 0.8375
    Epoch 9/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3733 - accuracy: 0.8431
    Epoch 10/100
    250/250 [==============================] - 0s 474us/step - loss: 0.3664 - accuracy: 0.8465
    Epoch 11/100
    250/250 [==============================] - 0s 542us/step - loss: 0.3608 - accuracy: 0.8514
    Epoch 12/100
    250/250 [==============================] - 0s 510us/step - loss: 0.3571 - accuracy: 0.8534
    Epoch 13/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3535 - accuracy: 0.8568
    Epoch 14/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3517 - accuracy: 0.8556
    Epoch 15/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3495 - accuracy: 0.8553
    Epoch 16/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3479 - accuracy: 0.8580
    Epoch 17/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3467 - accuracy: 0.8589
    Epoch 18/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3457 - accuracy: 0.8580
    Epoch 19/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3449 - accuracy: 0.8600
    Epoch 20/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3444 - accuracy: 0.8600
    Epoch 21/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3440 - accuracy: 0.8600
    Epoch 22/100
    250/250 [==============================] - 0s 494us/step - loss: 0.3435 - accuracy: 0.8600
    Epoch 23/100
    250/250 [==============================] - 0s 498us/step - loss: 0.3427 - accuracy: 0.8612
    Epoch 24/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3423 - accuracy: 0.8600
    Epoch 25/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3416 - accuracy: 0.8622
    Epoch 26/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3413 - accuracy: 0.8619
    Epoch 27/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3402 - accuracy: 0.8622
    Epoch 28/100
    250/250 [==============================] - 0s 502us/step - loss: 0.3399 - accuracy: 0.8627
    Epoch 29/100
    250/250 [==============================] - 0s 498us/step - loss: 0.3397 - accuracy: 0.8621
    Epoch 30/100
    250/250 [==============================] - 0s 506us/step - loss: 0.3396 - accuracy: 0.8645
    Epoch 31/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3390 - accuracy: 0.8624
    Epoch 32/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3392 - accuracy: 0.8624
    Epoch 33/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3384 - accuracy: 0.8633
    Epoch 34/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3384 - accuracy: 0.8639
    Epoch 35/100
    250/250 [==============================] - 0s 498us/step - loss: 0.3380 - accuracy: 0.8625
    Epoch 36/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3377 - accuracy: 0.8658
    Epoch 37/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3374 - accuracy: 0.8636
    Epoch 38/100
    250/250 [==============================] - 0s 502us/step - loss: 0.3371 - accuracy: 0.8634
    Epoch 39/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3368 - accuracy: 0.8658
    Epoch 40/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3365 - accuracy: 0.8651
    Epoch 41/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3366 - accuracy: 0.8643
    Epoch 42/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3354 - accuracy: 0.8662
    Epoch 43/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3362 - accuracy: 0.8641
    Epoch 44/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3359 - accuracy: 0.8641
    Epoch 45/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3358 - accuracy: 0.8630
    Epoch 46/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3352 - accuracy: 0.8639
    Epoch 47/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3354 - accuracy: 0.8648
    Epoch 48/100
    250/250 [==============================] - 0s 494us/step - loss: 0.3351 - accuracy: 0.8636
    Epoch 49/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3348 - accuracy: 0.8641
    Epoch 50/100
    250/250 [==============================] - 0s 506us/step - loss: 0.3350 - accuracy: 0.8625
    Epoch 51/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3345 - accuracy: 0.8643
    Epoch 52/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3352 - accuracy: 0.8640
    Epoch 53/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3346 - accuracy: 0.8625
    Epoch 54/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3345 - accuracy: 0.8630
    Epoch 55/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3344 - accuracy: 0.8635
    Epoch 56/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3340 - accuracy: 0.8639
    Epoch 57/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3340 - accuracy: 0.8646
    Epoch 58/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3337 - accuracy: 0.8648
    Epoch 59/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3336 - accuracy: 0.8649
    Epoch 60/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3336 - accuracy: 0.8637
    Epoch 61/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3331 - accuracy: 0.8648
    Epoch 62/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3334 - accuracy: 0.8643
    Epoch 63/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3332 - accuracy: 0.8646
    Epoch 64/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3333 - accuracy: 0.8640
    Epoch 65/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3333 - accuracy: 0.8641
    Epoch 66/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3330 - accuracy: 0.8641
    Epoch 67/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3332 - accuracy: 0.8646
    Epoch 68/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3327 - accuracy: 0.8645
    Epoch 69/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3328 - accuracy: 0.8648
    Epoch 70/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3324 - accuracy: 0.8637
    Epoch 71/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3323 - accuracy: 0.8656
    Epoch 72/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3327 - accuracy: 0.8655
    Epoch 73/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3324 - accuracy: 0.8654
    Epoch 74/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3319 - accuracy: 0.8644
    Epoch 75/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3322 - accuracy: 0.8643
    Epoch 76/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3321 - accuracy: 0.8654
    Epoch 77/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3319 - accuracy: 0.8659
    Epoch 78/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3322 - accuracy: 0.8650
    Epoch 79/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3317 - accuracy: 0.8639
    Epoch 80/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3313 - accuracy: 0.8645
    Epoch 81/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3312 - accuracy: 0.8656
    Epoch 82/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3313 - accuracy: 0.8649
    Epoch 83/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3313 - accuracy: 0.8644
    Epoch 84/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3312 - accuracy: 0.8655
    Epoch 85/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3309 - accuracy: 0.8649
    Epoch 86/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3304 - accuracy: 0.8662
    Epoch 87/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3307 - accuracy: 0.8645
    Epoch 88/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3304 - accuracy: 0.8640
    Epoch 89/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3306 - accuracy: 0.8645
    Epoch 90/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3308 - accuracy: 0.8643
    Epoch 91/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3304 - accuracy: 0.8643
    Epoch 92/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3305 - accuracy: 0.8641
    Epoch 93/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3305 - accuracy: 0.8637
    Epoch 94/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3301 - accuracy: 0.8654
    Epoch 95/100
    250/250 [==============================] - 0s 478us/step - loss: 0.3297 - accuracy: 0.8639
    Epoch 96/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3304 - accuracy: 0.8626
    Epoch 97/100
    250/250 [==============================] - 0s 490us/step - loss: 0.3299 - accuracy: 0.8639
    Epoch 98/100
    250/250 [==============================] - 0s 482us/step - loss: 0.3304 - accuracy: 0.8640
    Epoch 99/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3296 - accuracy: 0.8630
    Epoch 100/100
    250/250 [==============================] - 0s 486us/step - loss: 0.3299 - accuracy: 0.8639
    




    <keras.callbacks.History at 0x215aba4eca0>



## Part 4 - Making the predictions and evaluating the model

### Predicting the result of a single observation

Use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: $ 60000

Number of Products: 2

Does this customer have a credit card ? Yes

Is this customer an Active Member: Yes

Estimated Salary: $ 50000

So, should we say goodbye to that customer ?


```python
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```

    [[False]]
    

Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

### Predicting the Test set results


```python
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

    [[0 0]
     [0 1]
     [0 0]
     ...
     [0 0]
     [0 0]
     [0 0]]
    

### Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

    [[1520   75]
     [ 199  206]]
    




    0.863


