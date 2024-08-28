## Ali Akbar Naqvi
## Internship Project 3
## KNN Image Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv("train.csv").head(3500)
test_data=pd.read_csv("test.csv").head(3500)

##print(train_data)
##print(test_data)

y_train = train_data['label']
x_train = train_data.drop('label',axis=1)

plt.figure(figsize=(7,7))
some_digit=120
some_digit_image= x_train.iloc[some_digit].to_numpy().reshape(28,28)
plt.imshow(np.reshape(some_digit_image,(28,28)),cmap=plt.cm.gray)
print(y_train[some_digit])
plt.show()

sns.countplot(x=train_data['label'])
plt.show()