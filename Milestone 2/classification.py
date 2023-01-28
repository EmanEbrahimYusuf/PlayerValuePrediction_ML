import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
########################################################
def spliPostion(data):
    max_values = np.max([
        data['positions'].apply(lambda x: len(str(x).split(','))).max(),
    ])
    return data['positions'].str.split(',').apply(lambda x: x + ['0'] * (max_values - len(x)))

def convert_rate(col):
    dataframe[col]=le.fit_transform(dataframe[col])
    joblib.dump(le,'Milestone 2/predect/model/'+col+'_predect')
    dataframe[col]=dataframe[col].apply(lambda x:x+1)

# dic
model_params = {
    "LogisticRegression": {
    'model': LogisticRegression(),
    'params': {
        'C': [1, 5, 10, 15]
    }
        },
    "GaussianNB": {
        'model': GaussianNB(),
        'params': {

        }
    },
    "MultinomialNB": {
        'model': MultinomialNB(),
        'params': {
            'alpha': [1, 2, 3, 4],
            'fit_prior': [True, False],

        }
    },
    "DecisionTreeClassifier": {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ["gini", "entropy"],
            'splitter': ["best", "random"],
            'max_depth': [5, 10, 15, 20],
            'max_features': ["auto", "sqrt", "log2"]
        }
    }
    
}

#############################################################
dataframe=pd.read_csv("Milestone 2\player-classification.csv")
print(dataframe.shape)
print(dataframe.head(3))

print(dataframe.info())
print(dataframe.describe())
print(dataframe.columns)

# drop nulll & unusfull columns 
nullcols=['id', 'name', 'full_name', 'birth_date','traits','national_team','national_rating','national_team_position','national_jersey_number','club_join_date','contract_end_year','tags','traits']
dataframe.drop(nullcols,axis=1,inplace=True)
print(dataframe.shape)

print(dataframe.isna().sum())

c= dataframe['positions'].str.split(',').max()
c # max num of position can player play on it in data is 

# the lare number player can play in it is 3
play_postion = spliPostion(dataframe)
dataframe['position_first'] = play_postion.apply(lambda x: x[0])
dataframe['position_second'] = play_postion.apply(lambda x: x[1])
dataframe['position_third'] = play_postion.apply(lambda x: x[2])
dataframe.drop('positions', axis=1, inplace=True)



transform=['nationality','preferred_foot','body_type','club_team''club_position','position_first','position_second','position_third']

le=LabelEncoder()

dataframe['nationality']=le.fit_transform(dataframe['nationality'])
joblib.dump(le,'Milestone 2/predect/model/nationality_transform')

dataframe['preferred_foot']=le.fit_transform(dataframe['preferred_foot'])
joblib.dump(le,'Milestone 2/predect/model/preferred_foot_transform')


dataframe['body_type']=le.fit_transform(dataframe['body_type'])
joblib.dump(le,'Milestone 2/predect/model/body_type_transform')

dataframe['club_team']=le.fit_transform(dataframe['club_team'])
joblib.dump(le,'Milestone 2/predect/model/club_team_transform')

dataframe['club_position']=le.fit_transform(dataframe['club_position'])
joblib.dump(le,'Milestone 2/predect/model/club_position_transform')

dataframe['position_first']=le.fit_transform(dataframe['position_first'])
joblib.dump(le,'Milestone 2/predect/model/position_first_transform')

dataframe['position_second']=le.fit_transform(dataframe['position_second'])
joblib.dump(le,'Milestone 2/predect/model/position_second_transform')

dataframe['position_third']=le.fit_transform(dataframe['position_third'])
joblib.dump(le,'Milestone 2/predect/model/position_third_transform')

dataframe['upper_work_rate'] = dataframe['work_rate'].apply(lambda x: x.split('/')[0])
dataframe['down_work_rate'] = dataframe['work_rate'].apply(lambda x: x.split('/')[1])

dataframe['height_cm'] = dataframe['height_cm'].apply(lambda x:x/100)

print(dataframe.head(3))


minmax=MinMaxScaler()
dataframe['release_clause_euro']=minmax.fit_transform(dataframe[['release_clause_euro']])
joblib.dump(minmax,'Milestone 2/predect/model/minmax_release_clause_euro')
dataframe['wage']=minmax.fit_transform(dataframe[['wage']])
joblib.dump(minmax,'Milestone 2/predect/model/wage_release_clause_euro')


# rawan pre
columnsIter2 = pd.DataFrame(dataframe, columns=['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB'],index=[dataframe.index])

for(columnName1, columnData1) in columnsIter2.iteritems():
    dataframe[columnName1] = dataframe[columnName1].str.replace('+', '.')
    dataframe[columnName1]= pd.to_numeric(dataframe[columnName1])

#for null numerical values:
columnsIter = pd.DataFrame(dataframe, columns=['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB','wage','release_clause_euro','club_rating','club_jersey_number',],index=[dataframe.index])
for (columnName, columnData) in columnsIter.iteritems():
    dataframe[columnName] = dataframe[columnName].fillna(dataframe[columnName].mean())
print(dataframe.head(3))


convert_rate('upper_work_rate')
convert_rate('down_work_rate')

dataframe['num_of_work_rate']=dataframe['upper_work_rate']/dataframe['down_work_rate']
dataframe['num_of_work_rate'].head()

dataframe= dataframe.drop(['work_rate'],axis=1)

modelLevel = LabelEncoder()

dataframe['PlayerLevel']=modelLevel.fit_transform(dataframe['PlayerLevel'])
joblib.dump(modelLevel,'Milestone 2/predect/model/Leve_predect')


for col in dataframe.columns:
    if dataframe[col].isnull:
        
        dataframe[col].fillna(dataframe[col].mean(),inplace=True)

cleandata=dataframe
cleandata.shape
cleandata.head()

corr=cleandata.corr()
print(corr)

top_feature = corr.index[abs(corr['PlayerLevel'])>0.0]
df=dataframe[top_feature]

x=df.drop(['PlayerLevel'],axis=1)
y=df['PlayerLevel']


dataframe=shuffle(dataframe)
x,xt,y,yt=train_test_split(dataframe.drop(['PlayerLevel'],axis=1),dataframe['PlayerLevel'],train_size=.8,random_state=0)

import time
from sklearn.model_selection import GridSearchCV
scorre=[]
for mn,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    tic=time.time()
    clf.fit(dataframe.drop(['PlayerLevel'],axis=1),dataframe['PlayerLevel'])
    toc=time.time()
    scorre.append({
        'model name ': mn,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_,
        'trainTime':toc-tic
    })

df=pd.DataFrame(scorre)
print(df)

model=DecisionTreeClassifier(criterion='gini',max_depth=15,max_features='auto',splitter='best')
tic=time.time()
model.fit(x,y)
toc=time.time()
joblib.dump(model,'predect/model/Class_predect')
print(toc-tic)

tic=time.time()
model.score(xt,yt)
toc=time.time()
print(toc-tic)