## Ali Akbar Naqvi
## Internship Project 3
## KNN Image Classifier

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

#print(train_data)
#print(test_data)

y_train = train_data['label']
x_train = train_data.drop('label',axis=1)

x_train,x_test,y_train,y_test= train_test_split(x_train,y_train, test_size=0.3, random_state=42)


Scaler= StandardScaler()
Scaler.fit(x_train,y_train)
x_train= Scaler.transform(x_train)

K_Range = range(1,31)
K_Score=[]

for K in K_Range:
    KNN=KNeighborsClassifier(n_neighbors=K)
    KNN.fit(x_train,y_train)
    y_predict = KNN.predict(x_test)
    score= accuracy_score(y_test,y_predict)
    K_Score.append(score)
#print("Accuracy of each value of K :", K_Score)

K=20
KNN=KNeighborsClassifier(n_neighbors=K)
KNN.fit(x_train,y_train)
y_predict = KNN.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))

plt.plot(K_Range,K_Score,marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

with open('KNNImageClassifier.pkl', 'wb') as file:
    pickle.dump(KNN,file)