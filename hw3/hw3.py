from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Xy=pd.read_csv('cses4_cut.csv')  

cols = [0]                                      # dropping the 1st column because its just a row number, index
Xy.drop(Xy.columns[cols],axis=1,inplace=True)   # dropping the 1st column because its just a row number, index


def consider_those_as_missing_too(number):    # Im gonna assume REFUSED and UNKNOWN responses as MISSING too
    return {str(number):(number,number-1,number-2) }

missings=[9,99,999,9999,99999]
a={}
for x in missings:
    a={**a,**consider_those_as_missing_too(x)}
#a becomes
#{'9': (9, 8, 7),       #if maximum of a column is 9 then all 7 8 9 will be treated as 9
 #'99': (99, 98, 97),   #if maximum of a column is 99 then all 97 98 99 will be treated as 99
 #'999': (999, 998, 997),
 #'9999': (9999, 9998, 9997),
 #'99999': (99999, 99998, 99997)}
 # based on the maximum value of the column, those cells which has those the dict.values(98,997..) will be updated as NONE
 
df1 = Xy.max()     # maximum value of each column as pandas series    
df1=df1.to_dict()     # converting pandas series to dictionary

for x,y in df1.items(): 
    if x in('D2002','age','voted'):  # there is no missing value in those columns
        continue
    else:
        Xy.loc[Xy[x] == a.get(str(y))[0], [x]] = None # converting all MISSING, REFUSED, I DONT KNOW to None/NAN
        Xy.loc[Xy[x] == a.get(str(y))[1], [x]] = None # converting all MISSING, REFUSED, I DONT KNOW to None/NAN
        Xy.loc[Xy[x] == a.get(str(y))[2], [x]] = None # converting all MISSING, REFUSED, I DONT KNOW to None/NAN
        
def true_false_encode (row):  # making a mapping True=1,  False=0
   if row['voted'] == True:
      return 1
   else:
      return 0
    
Xy['voted_label'] = Xy.apply (lambda row: true_false_encode(row), axis=1) # making a new target column with elements (1,0)
X = Xy.drop(['voted_label','voted'], axis=1)
y = Xy['voted_label']

categoricals=[]  # all inputs are categorical but age 
for col in X.columns:
    if col!='age':
        categoricals.append(col)
        
X=pd.get_dummies(X, columns=categoricals, dummy_na=True) # one hot encoding - missing values treated as another input
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=1)


################################### TRYING DIFFERENT MODELS AND COMPARING THEM
################################### TRYING DIFFERENT MODELS AND COMPARING THEM
################################### TRYING DIFFERENT MODELS AND COMPARING THEM
################################### AT THIS STAGE ALL 552 INPUTS INCLUDED NO FEATURE SELECTION
model = GaussianNB()
model.fit(Xtrain, ytrain) 
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Naive Bayes Gaussian')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = KNeighborsClassifier(n_neighbors=1)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('K-Neighbors-Classifier N=1')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = KNeighborsClassifier(n_neighbors=2)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('K-Neighbors-Classifier N=2')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_scores = model.predict_proba(Xtest) #probability outcomes
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Logistic Regression')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = RandomForestClassifier(max_depth=5, random_state=0)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Random Forest Classifier D=5')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = RandomForestClassifier(max_depth=10, random_state=0)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Random Forest Classifier D=10')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')


model = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=1, max_iter=1000,alpha=0.001)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Neural Network 10 10 10')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = MLPClassifier(hidden_layer_sizes=(5,5,5),random_state=1, max_iter=1000,alpha=0.001)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Neural Network 5 5 5')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

################################### NOW APPLYING FEATURE SELECTION BY CONTRIBUTION OF EACH VARIABLE USING LOGREG
################################### NOW APPLYING FEATURE SELECTION BY CONTRIBUTION OF EACH VARIABLE USING LOGREG
###################################  NUMBER OF INPUTS REDUCED TO 222

# #Selecting the Best important features according to Logistic Regression using SelectFromModel
feature_selector = SelectFromModel(estimator=LogisticRegression(random_state=0, max_iter=1000))
Xtrain=feature_selector.fit_transform(Xtrain, ytrain) # removing features from train data
Xtest=feature_selector.transform(Xtest)               # retain  same remaining features for test data

###################################  THEN TRYING DIFFERENT MODELS AND COMPARING THEM
###################################  THEN TRYING DIFFERENT MODELS AND COMPARING THEM
###################################  THEN TRYING DIFFERENT MODELS AND COMPARING THEM


model = GaussianNB()
model.fit(Xtrain, ytrain) 
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Naive Bayes Gaussian')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = KNeighborsClassifier(n_neighbors=1)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('K-Neighbors-Classifier N=1')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = KNeighborsClassifier(n_neighbors=2)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('K-Neighbors-Classifier N=2')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_scores = model.predict_proba(Xtest) #probability outcomes
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Logistic Regression')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = RandomForestClassifier(max_depth=5, random_state=0)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Random Forest Classifier D=5')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = RandomForestClassifier(max_depth=10, random_state=0)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Random Forest Classifier D=10')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')


model = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=1, max_iter=1000,alpha=0.001)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Neural Network 10 10 10')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')

model = MLPClassifier(hidden_layer_sizes=(5,5,5),random_state=1, max_iter=1000,alpha=0.001)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
y_model_train = model.predict(Xtrain) 
plot_confusion_matrix(model, Xtest, ytest)
plt.title('Neural Network 5 5 5')
plt.show()
print(accuracy_score(ytest, y_model), 'TEST ACCURACY')
print(accuracy_score(ytrain, y_model_train), 'TRAIN ACCURACY')




