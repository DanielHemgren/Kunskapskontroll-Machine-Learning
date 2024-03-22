import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline



def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

#Data för träning,validering,test

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

#Split 60-10 av tot data 70 dvs 10 test skiljs av
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)
#Split 50-10 där 10 samplas av i olika batches till validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)




# Standardizing the data (as you generally always should do when using SVM models).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Only transforming the validation data. 
X_test_scaled = scaler.transform(X_test)  # Only transforming the test data. 

#TSNE

X.shape


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap=plt.cm.get_cmap("jet", 10), s=10)
plt.colorbar(ticks=range(10))
plt.title('t-SNE Visualization of MNIST')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()

#Modeller instansieras ur några modellklasser i Scikit-learn


random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
lin_clf = LinearSVC(max_iter=100, tol=20, random_state=42)



 #lista skapas list_comprehenda över listan vid träning.
estimators = [random_forest_clf, extra_trees_clf, lin_clf]

#loop konstruktion av modellering

#Modeller tränas på de 50000 i train.
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)


#Kontroll av att decision tree modellerna har inga problem att "overfit to perfection" för ett visst set. Detta är inte 
#generaliserbart så klart. SVM lyckas också mappa datat relativt bra på 0.86.
[estimator.score(X_train, y_train) for estimator in estimators]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#cross validation of bigger set accuracy
for estimator in estimators:
    print("Cross-validating", estimator.__class__.__name__)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='accuracy')
    print("Accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation of accuracy:", scores.std())



#Kontroll av modeller på de 10000 i val. Här ses att en viss generaliserbarhet överlever till hela subsetet.

[estimator.score(X_val, y_val) for estimator in estimators]

[estimator.score(X_test, y_test) for estimator in estimators]





#Voting classifier skapas och tränas

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("lin_clf", lin_clf)
]

voting_clf = VotingClassifier(named_estimators)

voting_clf.fit(X_train, y_train)





print("Cross-validating", voting_clf.__class__.__name__)
scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')

# Print the results
print("Accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())

#Validering av votingmodellen på de 10000 i val.

voting_clf.score(X_val, y_val)





voting_clf.score(X_test, y_test)

y_pred_voting = voting_clf.predict(X_test)
accuracy_score(y_test, y_pred_voting)

display_confusion_matrix(y_test, y_pred_voting)

cm2 = confusion_matrix(y_test, y_pred_voting)

# Compute probabilities
class_counts = np.sum(cm2, axis=1)  # Total number of samples in each class
probabilities = cm2 / class_counts[:, np.newaxis]  # Divide each count by the total number of samples in the corresponding class

print("Confusion Matrix:")
print(cm2)
print("\nProbabilities:")
print(probabilities)


from sklearn.metrics import log_loss

# Confusion matrix
cm = np.array([[969, 0, 3, 0, 1, 1, 2, 1, 6, 0],
               [0, 1137, 5, 2, 1, 0, 2, 4, 1, 0],
               [4, 1, 940, 2, 2, 1, 5, 5, 7, 0],
               [1, 2, 17, 983, 0, 4, 2, 10, 8, 7],
               [3, 0, 1, 2, 874, 0, 3, 3, 2, 18],
               [5, 2, 0, 16, 1, 897, 3, 0, 11, 2],
               [4, 1, 0, 0, 3, 8, 944, 0, 1, 0],
               [2, 4, 13, 0, 7, 0, 0, 1021, 2, 6],
               [2, 5, 9, 11, 6, 7, 3, 4, 916, 6],
               [5, 5, 2, 15, 15, 1, 1, 10, 8, 974]])

# Convert confusion matrix to probabilities
probabilities = cm / cm.sum(axis=1, keepdims=True)

# True class labels (you need to provide the true labels)
y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Compute cross entropy using the true class labels and probabilities
cross_entropy = log_loss(y_true, probabilities)

print("Cross-Entropy (Log Loss):", cross_entropy)


# Confusion matrix
cm = np.array([[969, 0, 3, 0, 1, 1, 2, 1, 6, 0],
               [0, 1137, 5, 2, 1, 0, 2, 4, 1, 0],
               [4, 1, 940, 2, 2, 1, 5, 5, 7, 0],
               [1, 2, 17, 983, 0, 4, 2, 10, 8, 7],
               [3, 0, 1, 2, 874, 0, 3, 3, 2, 18],
               [5, 2, 0, 16, 1, 897, 3, 0, 11, 2],
               [4, 1, 0, 0, 3, 8, 944, 0, 1, 0],
               [2, 4, 13, 0, 7, 0, 0, 1021, 2, 6],
               [2, 5, 9, 11, 6, 7, 3, 4, 916, 6],
               [5, 5, 2, 15, 15, 1, 1, 10, 8, 974]])

# Compute precision and recall
precision = np.diag(cm) / np.sum(cm, axis=0)  # Precision = TP / (TP + FP)
recall = np.diag(cm) / np.sum(cm, axis=1)  # Recall = TP / (TP + FN)

# Print probabilities (precision and recall)
print("Precision:", precision)
print("Recall:", recall)


cm = np.array([[969, 0, 3, 0, 1, 1, 2, 1, 6, 0],
               [0, 1137, 5, 2, 1, 0, 2, 4, 1, 0],
               [4, 1, 940, 2, 2, 1, 5, 5, 7, 0],
               [1, 2, 17, 983, 0, 4, 2, 10, 8, 7],
               [3, 0, 1, 2, 874, 0, 3, 3, 2, 18],
               [5, 2, 0, 16, 1, 897, 3, 0, 11, 2],
               [4, 1, 0, 0, 3, 8, 944, 0, 1, 0],
               [2, 4, 13, 0, 7, 0, 0, 1021, 2, 6],
               [2, 5, 9, 11, 6, 7, 3, 4, 916, 6],
               [5, 5, 2, 15, 15, 1, 1, 10, 8, 974]])

# Labels
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Compute precision and recall
precision = np.diag(cm) / np.sum(cm, axis=0)  # Precision = TP / (TP + FP)
recall = np.diag(cm) / np.sum(cm, axis=1)  # Recall = TP / (TP + FN)

# Print precision and recall in neat columns
print("Label   Precision   Recall")
print("-" * 26)
for label, prec, rec in zip(labels, precision, recall):
    print(f"{label:5d}   {prec:.4f}      {rec:.4f}")


#fungerar inte!!

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
loss = log_loss(y_test, y_pred_voting, labels=labels)

#testfasen av den kombinerade modellen på de 10000 i test
#modellen uppnår bra accuracy även på testdatat och väljs därför
#Jag är i och med detta övertygad om att modellen har en bra balans mellan bias och variance.
#Confusion matrix är "balanced" och inget sticker ut mycket. 18 gånger predikteras 9 på sanna 4:or det sticker ut mest.
# modellen uppvisar hög och lågt varierande accuracy över olika set. 
#cross validation metoder och gridsearch på hyperparametrar kan öka resultatet men det får anses "good enough" för ändamålet.
# balanserade klasser gör att accuracy är lämpligt

accuracy = voting_clf.score(X_test, y_test)
print(f'Accuracy for voting_clf:{accuracy}')


# Sparar modellen med joblib
joblib.dump(voting_clf,'voting_clf3.joblib')

