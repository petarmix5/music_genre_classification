# music_genre_classification
Klasifikacija glazbenih djela po žanrovima


Ovaj projekt je napravljen u sklopu kolegija "Inteligentni sustavi 1". 
Potrebno je otvoriti u Google Colaboratory alatu.

<pre>
<code>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/deep-learning-music-genre-classification/Data/features_30_sec.csv")
df.head()



plt.scatter(df['label'], df['tempo'])
plt.xlabel("Žanrovi")
plt.ylabel("Tempo")




X = df.drop(['filename','label'],axis=1)
print(X)
y = df['label']

print(y)
print("X shape is {}".format(X.shape))
print()
print("y shape is {}".format(y.shape))



y





**Odvajanje testnih podataka i podataka za treniranje**


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

len(X_train)

len(X_test)

len(y_train)

len(y_test)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



SVM metoda - SUPPORT VECTOR MACHINES

from sklearn import svm
model_SVM = svm.SVC()
model_SVM.fit(X_train, y_train)


# Predviđanje

y_pred_SVM = model_SVM.predict(X_test)

Točnost

from sklearn.metrics import accuracy_score
print("Točnost je: {:.4f}".format(accuracy_score(y_test, y_pred_SVM))) 

Matrica konfuzije za SVM

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_SVM)

X_train.values

y_train.values





RFC (Random Forest Classifier)
##RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=5)
forest.fit(X_train,y_train)

print("Točnost je: {:.4f}".format(forest.score(X_test,y_test)))

y_pred_RFC = forest.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RFC))

from sklearn.metrics import recall_score, f1_score
recall_score(y_test,y_pred_RFC, average='micro')

x = [y_pred_RFC,y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

forest.classes_

forest.n_features_

X_train.shape[1]

temp_X = X['tempo']
print(temp_X)

temp_y = y
print(temp_y)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_RFC),index = forest.classes_, columns=forest.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


confusion_matrix(y_test,y_pred_RFC)

plt.figure(figsize=(8,5))
plt.scatter(temp_X, temp_y)
plt.title("Prikaz tempa o pojedinom žanru", size=15)
plt.xlabel("Tempo",size=14)
plt.ylabel("Žanrovi",size=14)

plt.show()




K-nearest Neighbors
## K-nearest Neighbors metoda

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors = 10)
modelKNN.fit(X_train,y_train)

Predviđanje za KNN metodu

y_pred_KNN = modelKNN.predict(X_test)

Preciznost za KNN metodu

from sklearn.metrics import accuracy_score


print("Točnost za model KNN je: {:.4f}".format(accuracy_score(y_test,y_pred_KNN)))

Matrica konfuzije za KNN metodu

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_KNN)

Drugačiji prikaz matrice konfuzije 

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_KNN),index = modelKNN.classes_, columns=modelKNN.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


plt.scatter(y_pred_KNN,y_test)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

x = [y_pred_KNN, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()




Gaussian Naive Bayes

## Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X_train,y_train)

Predviđanje za Naive Bayes

y_pred_NB = modelNB.predict(X_test)

Točnost za metodu Gaussian Naive Bayes

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_NB)))

Matrica konfuzije za metodu Gaussian Naive Bayes

confusion_matrix(y_test,y_pred_NB)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_NB),index = modelNB.classes_, columns=modelNB.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_NB, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()



Logistička regresija
## Logistička regresija

from sklearn.linear_model import LogisticRegression


logR = LogisticRegression(max_iter=1000)
logR.fit(X_train,y_train)

Predviđanje za logističku regresiju

y_pred_logR = logR.predict(X_test)

Točnost

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_logR)))

Matrica konfuzije za logističku regresiju

confusion_matrix(y_test,y_pred_logR)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_logR),index = logR.classes_, columns=logR.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_logR, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()






Povećanje broja testnih podataka

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

###Support Vector Machine

#SUPPORT VECTOR MACHINE
from sklearn import svm
model_SVM = svm.SVC()
model_SVM.fit(X_train, y_train)

Predviđanje za SVM

y_pred_SVM = model_SVM.predict(X_test)

Točnost za SVM

from sklearn.metrics import accuracy_score
print("Točnost je: {:.4f}".format(accuracy_score(y_test, y_pred_SVM)))

Matrica konfuzije za SVM

confusion_matrix(y_test,y_pred_SVM)

x = [y_pred_SVM, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()






###RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=5)
forest.fit(X_train,y_train)

print("Točnost je: {:.4f}".format(forest.score(X_test,y_test)))

y_pred_RFC = forest.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RFC))

from sklearn.metrics import recall_score, f1_score
print("{:.4f}".format(recall_score(y_test,y_pred_RFC, average='micro')))

from sklearn.metrics import confusion_matrix
import seaborn as sn




plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_RFC),index = forest.classes_, columns=forest.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_RFC,y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

forest.classes_

temp_X = X['tempo']
print(temp_X)

temp_y = y
print(temp_y)

plt.figure(figsize=(8,5))
plt.scatter(temp_X, temp_y)
plt.title("Prikaz tempa o pojedinom žanru", size=15)
plt.xlabel("Tempo",size=14)
plt.ylabel("Žanrovi",size=14)

plt.show()





### K-nearest neighbors metoda

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors = 10)
modelKNN.fit(X_train,y_train)

modelKNN.classes_

Predviđanje za KNN metodu 

y_pred_KNN = modelKNN.predict(X_test)

Točnost za KNN metodu


from sklearn.metrics import accuracy_score
print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_KNN)))

Matrica konfuzije za KNN

confusion_matrix(y_test,y_pred_KNN)

x = [y_pred_KNN, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()


### Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X_train,y_train)

Predviđanje za GaussianNB

y_pred_NB = modelNB.predict(X_test)

Točnost za GaussianNB

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_NB)))

Matrica konfuzije za GaussianNB

confusion_matrix(y_test,y_pred_NB)

x = [y_pred_NB, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()





###Logistička regresija 

logR = LogisticRegression(max_iter=1000)
logR.fit(X_train,y_train)

Predviđanje za Logističku regresiju 

y_pred_logR = logR.predict(X_test)

Točnost za Logističku regresiju

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_logR)))

Matrica konfuzije za Logističku regresiju

confusion_matrix(y_test,y_pred_logR)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_logR),index = logR.classes_, columns=logR.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_logR, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()


###Logistička regresija 

logR = LogisticRegression(max_iter=1000)
logR.fit(X_train,y_train)

Predviđanje za Logističku regresiju 

y_pred_logR = logR.predict(X_test)

Točnost za Logističku regresiju

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_logR)))

Matrica konfuzije za Logističku regresiju

confusion_matrix(y_test,y_pred_logR)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_logR),index = logR.classes_, columns=logR.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_logR, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

Napravili smo novu bazu podataka koja sadrži 1000 redaka odnosno sadrži sve podatke vezane uz žanrove, ali samo 5 značajki u X i jedna značajka odnosno **label** u y varijabli. 

novabaza = df

X = novabaza[['harmony_mean','zero_crossing_rate_mean','tempo','rolloff_mean','perceptr_mean']]
X = X[::10]
print(X)
y = novabaza['label']
y = y[::10]
y



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)







#### Support Vector Machine

from sklearn import svm
model_SVM = svm.SVC()
model_SVM.fit(X_train, y_train)

Predviđanje za SVM

y_pred_SVM = model_SVM.predict(X_test)

Točnost za SVM

from sklearn.metrics import accuracy_score
print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_SVM)))

Matrica konfuzije za SVM

confusion_matrix(y_test,y_pred_SVM)

x = [y_pred_SVM, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()








###RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=5)
forest.fit(X_train,y_train)

print("Točnost je: {:.4f}".format(forest.score(X_test,y_test)))

y_pred_RFC = forest.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RFC))

from sklearn.metrics import recall_score, f1_score
print("{:.4f}".format(recall_score(y_test,y_pred_RFC, average='micro')))

from sklearn.metrics import confusion_matrix
import seaborn as sn




plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_RFC),index = forest.classes_, columns=forest.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_RFC,y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

forest.classes_

temp_X = X['tempo']
print(temp_X)

temp_y = y
print(temp_y)









#### K-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors = 10)
modelKNN.fit(X_train,y_train)

Predviđanje za KNN metodu 

y_pred_KNN = modelKNN.predict(X_test)

Točnost za KNN metodu


from sklearn.metrics import accuracy_score
print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_KNN)))

Matrica konfuzije za KNN

confusion_matrix(y_test,y_pred_KNN)

x = [y_pred_KNN, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()








#### Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X_train,y_train)

Predviđanje za GaussianNB

y_pred_NB = modelNB.predict(X_test)

Točnost za GaussianNB

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_NB)))

Matrica konfuzije za GaussianNB

confusion_matrix(y_test,y_pred_NB)

x = [y_pred_NB, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()







###Logistička regresija 

logR = LogisticRegression(max_iter=1000)
logR.fit(X_train,y_train)

Predviđanje za Logističku regresiju 

y_pred_logR = logR.predict(X_test)

Točnost za Logističku regresiju

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_logR)))

Matrica konfuzije za Logističku regresiju

confusion_matrix(y_test,y_pred_logR)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_logR),index = logR.classes_, columns=logR.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_logR, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()











###Logistička regresija 

logR = LogisticRegression(max_iter=1000)
logR.fit(X_train,y_train)

Predviđanje za Logističku regresiju 

y_pred_logR = logR.predict(X_test)

Točnost za Logističku regresiju

print("Točnost je: {:.4f}".format(accuracy_score(y_test,y_pred_logR)))

Matrica konfuzije za Logističku regresiju

confusion_matrix(y_test,y_pred_logR)

from sklearn.metrics import confusion_matrix
import seaborn as sn
plt.figure(figsize=(8,5))
df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred_logR),index = logR.classes_, columns=logR.classes_)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.show()


x = [y_pred_logR, y_test]
plt.hist(x, bins = 10)
plt.xlabel('Predviđene vrijednosti')
plt.ylabel('Prave vrijednosti')
plt.show()

#Prikaz rezultata

tablica = [('20.33 %','22.44 %','3.33 %'),
           ('63.67 %','62.44 %', '30.00%' ),
           ('24.00 %','24.89 %', '13.33 %'),
           ('38.33 %','36.89 %', '26.67 %'),
           ('36.36 %','36.67 %', '13.33 %')]

details = pd.DataFrame(tablica, columns=[ 'Baza cijela', 'Testni podaci 45 %', 'Nova baza'], index=['SVM', 'RFC','KNN', 'GNB','LOGR'])

details




</code>
</pre>
