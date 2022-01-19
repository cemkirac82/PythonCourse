import pandas as pd
import numpy as np
import re
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt



Xy=pd.read_csv('real_estate.csv')
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})



#for a in Xy.columns:                     #CHECKING UNIQUE VALUES OF COLUMNS
#    if a in ('target','price'):          #CHECKING UNIQUE VALUES OF COLUMNS
#        continue                         #CHECKING UNIQUE VALUES OF COLUMNS
#    else:                                #CHECKING UNIQUE VALUES OF COLUMNS
#        print(Xy[a].unique())            #CHECKING UNIQUE VALUES OF COLUMNS


del Xy['r']          #irrelevant data
del Xy['ad_link']    #irrelevant data
del Xy['ad_date']    #irrelevant data
del Xy['ad_id']    #irrelevant data        
del Xy['naturalgas_avb']    #ALL NULLS --- no need
del Xy['real_estate_type']    #ALL for sale --- no need
Xy.replace('Belirtilmemiş', None, inplace=True) # Belirtilmemiş means not specified,  means equal to None
Xy.replace('Bilinmiyor', None, inplace=True) # Bilinmiyor means unknown,  means equal to None

#Transforming some categoricals to numerical because in reality they are actually numbers
# for example number of rooms 'Between 5-10' we dont know exact number but we can replace with 7.5 for numerical use
Xy.replace('5-10 arası', '7.5', inplace=True) 
Xy.replace('30 ve üzeri', '35.0', inplace=True) 
Xy.replace('31 ve üzeri', '35.0', inplace=True) 
Xy.replace('Yüksek Giriş', '1', inplace=True) 
Xy.replace('Giriş Katı', '1', inplace=True) 
Xy.replace('Bahçe Katı', '1', inplace=True) 
Xy.replace('Zemin Kat', '0.5', inplace=True) 
Xy.replace('Bodrum Kat', '0.3', inplace=True) 
Xy.replace('Çatı Katı', '10', inplace=True) 
Xy.replace('21-25 arası', '23', inplace=True) 
Xy.replace('26-30 arası', '28.0', inplace=True) 
Xy.replace('11-15 arası', '13', inplace=True) 
Xy.replace('16-20 arası', '18', inplace=True) 
Xy.replace('10 Üzeri', '12', inplace=True) 
Xy.replace('Giriş Altı Kot 1', '0.3', inplace=True) 
Xy.replace('Giriş Altı Kot 2', '0.3', inplace=True) 
Xy.replace('Giriş Altı Kot 3', '0.3', inplace=True) 
Xy.replace('Giriş Altı Kot 4', '0.3', inplace=True) 
Xy.replace('Stüdyo (1+0)', '1', inplace=True) 
Xy.replace('Yok', '0', inplace=True)
Xy.replace('Var', '1', inplace=True)
Xy.replace('6 Üzeri', '6.0', inplace=True)
Xy.replace('Evet', '1', inplace=True)
Xy.replace('Hayır', '0', inplace=True)
Xy.replace('Müstakil', '1', inplace=True)

target_array=np.array(Xy['price'],str)  #transfer target column to numpy array
target_array=np.char.strip(np.char.replace(target_array, '.', "", count=None), chars=' TL').astype(float) #transform to number
Xy.insert(0, 'target', target_array) #transfer back to pandas
del Xy['price']    #we dont need the old target anymore
Xy = Xy.apply(lambda x:x.fillna(x.value_counts().index[0]))  #filling nan with most frequent


# converting below  list into integer & all numbers inside string to floats
#['3+1' '6+2' '2+1' '4+1' '3+2' '1+1' '4+2' '5+1' '5+2' '4.5+1' '3.5+1' '1'
# '7+1' '7+2' '1.5+1' '2+2' '8+2' '6+1' '2.5+1' '12'] number_of_rooms
# converted to this-----> [1,2,3,4,5....]
for col in Xy.columns:
    try:
        Xy[col]=Xy[col].apply(lambda x:eval(x))
    except:
        continue

categoricals=[]  # collecting categoricals names
for col,tp in Xy.dtypes.items():
    if tp=='object':
        categoricals.append(col)
Xy=pd.get_dummies(Xy, columns=categoricals, dummy_na=True) # one hot encoding 


#creating a new feature
#the in between floors are usually more expensive in Turkey
# if a building is lets say 10 floors on total floor 1,2,9,10 are the cheapest  3...8 are most expensive
#therefore we are calculating a ration of the floor of the flat divided by total number of floors
Xy['floor ratio']=Xy['which_floor']/Xy['total_numberof_floors']


numericals=[]  # collecting only floats , not booleans(1-0 excluded)
for col,tp in Xy.dtypes.items():
    if tp=='float64':
        numericals.append(col)
numeric_Xy=Xy[numericals]


numericals_and_boolans=[]  # collecting only floats + booleans(1-0 incl)
for col,tp in Xy.dtypes.items():
    if tp=='float64':
        numericals_and_boolans.append(col)
    if tp=='uint8':
        numericals_and_boolans.append(col)
        
#MIN MAX SCALING
scaled_features = MinMaxScaler().fit_transform(numeric_Xy.values)
scaled_features_df = pd.DataFrame(scaled_features, index=numeric_Xy.index, columns=numeric_Xy.columns)
        
#visualizing numericals vs target (excluding booleans and categoricals)
#Xy[numericals].scatterplot();
var_x=[]
for x in scaled_features_df.columns:
    if x!='target':
        var_x.append(x)
   
#PLOTTING SNS PAIRPLOT SCATTERS
b=0
for a in range(len(var_x)//3):
    d=0
    g=sns.pairplot(scaled_features_df,y_vars=['target'],x_vars=var_x[b:b+3], height=5 )
    g.axes[0,d].set_xlim((0,0.5))
    g.axes[0,d].set_ylim((0,0.5))
    g.axes[0,d+1].set_xlim((0,0.5))
    g.axes[0,d+1].set_ylim((0,0.5))
    g.axes[0,d+2].set_xlim((0,0.5))
    g.axes[0,d+2].set_ylim((0,0.5))
    if b+2>len(var_x):
        b=len(var_x)
    else:
        b+=3
    plt.show()

#PLOTTING CORRELATION HEATMAP
Correlations = Xy.corr()
heatmap_corr = numeric_Xy.corr()
cor_target = abs(Correlations["target"])
sns.heatmap(heatmap_corr, annot=True)




#FEATURES WITH VERY LOW CORRELATION TO TARGET ARE BEING ELIMINATED 


relevant_features = cor_target[cor_target>0.1]
model_features=[]
for x,y in relevant_features.items():
    model_features.append(x)
Xy=Xy[model_features]

del Xy['square_meter_net']    #highly correlated with another feature --> square_meter_total
del Xy['bathrooms']    #highly correlated with another feature-->  square_meter_total
del Xy['number_of_rooms']    #highly correlated with another feature --> square_meter_total

#MLP REGRESSION with sklearn 
Xy_tr, Xy_te = train_test_split(Xy, test_size=0.4 , random_state=5, shuffle=True)
print('MLP Regressor')
Xy_train=Xy_tr.drop('target',axis=1)
y_train=Xy_tr['target']
Xy_test=Xy_te.drop('target',axis=1)
y_test=Xy_te['target']
# 1598 trn,  1066 test 2664  totaldata points
regression_1 = MLPRegressor(hidden_layer_sizes=(32,16,10,5), random_state=1, max_iter=5000,alpha=0.001).fit(Xy_train, y_train)
regression_2 = MLPRegressor(hidden_layer_sizes=(32,16,8,6,4), random_state=1, max_iter=5000,alpha=0.001).fit(Xy_train, y_train)
regression_3 = MLPRegressor(hidden_layer_sizes=(32,16,8), random_state=1, max_iter=5000,alpha=0.001).fit(Xy_train, y_train)
print('mlp 1 regressor TRAIN R SQUARED',regression_1.score(Xy_train, y_train))
print('mlp 1 regressor TEST R SQUARED',regression_1.score(Xy_test, y_test))
print('mlp 2 regressor TRAIN R SQUARED',regression_2.score(Xy_train, y_train))
print('mlp 2 regressor TEST R SQUARED',regression_2.score(Xy_test, y_test))
print('mlp 3 regressor TRAIN R SQUARED',regression_3.score(Xy_train, y_train))
print('mlp 3 regressor TEST R SQUARED',regression_3.score(Xy_test, y_test))

#RANDOM FOREST REGRESSION with sklearn 
print('Random Forest Regressor')
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(Xy_train, y_train)
regr2 = RandomForestRegressor(max_depth=3, random_state=0)
regr2.fit(Xy_train, y_train)
regr3 = RandomForestRegressor(max_depth=4, random_state=0)
regr3.fit(Xy_train, y_train)
regr4 = RandomForestRegressor(max_depth=5, random_state=0)
regr4.fit(Xy_train, y_train)
print('rf md=2 TRAIN R SQUARED',regr.score(Xy_train, y_train))
print('rf md=2 TRAIN R SQUARED',regr.score(Xy_test, y_test))
print('rf md=3 TRAIN R SQUARED',regr2.score(Xy_train, y_train))
print('rf md=3 TRAIN R SQUARED',regr2.score(Xy_test, y_test))
print('rf md=4 TRAIN R SQUARED',regr3.score(Xy_train, y_train))
print('rf md=4 TRAIN R SQUARED',regr3.score(Xy_test, y_test))
print('rf md=5 TRAIN R SQUARED',regr4.score(Xy_train, y_train))
print('rf md=5 TRAIN R SQUARED',regr4.score(Xy_test, y_test))

#LINEAR REGRESSION with sklearn 
correlated_dropper=DropCorrelatedFeatures(variables=None , method='pearson' ,threshold=0.7 )
Xy_train=correlated_dropper.fit_transform(Xy_train) 
Xy_test=correlated_dropper.transform(Xy_test)
regression = LinearRegression().fit(Xy_train, y_train)
print('TRAIN R SQUARED',regression.score(Xy_train, y_train))
print('TEST R SQUARED',regression.score(Xy_test, y_test))

#LINEAR REGRESSION with statsmodels
X2 = sm.add_constant(Xy_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())