#!/usr/bin/env python
# coding: utf-8

# In[2]:


# loading required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import aix360
from aix360.algorithms.rbm import FeatureBinarizer
from aix360.algorithms.rbm import LogisticRuleRegression
import shap
shap.initjs()
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from functools import reduce
from sklearn.metrics import accuracy_score

def plot_distribution(dataset, cols=2, width=30, height=30, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column].dropna())
            plt.xticks(rotation=25)

def data_explainability(sensor_data, label):
    plot_distribution(sensor_data, cols=5, width=50, height=50, hspace=0.5, wspace=0.5)
    sensor_data.describe()
    sensor_data.boxplot(by=label, figsize= (10,10))
    sns.pairplot(sensor_data[sensor_data.columns])
    # scatterplots for joint relationships and histograms for univariate distributions
    g = sns.PairGrid(sensor_data[sensor_data.columns])
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, n_levels=6)
    # Plot the distribution of each feature
    
    
def logistic_feature_imp(sensor_data):
    X=sensor_data[sensor_data.columns[1:]]
    Y= sensor_data.class_label.astype('int64')
    model_logistic = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model_logistic.fit(X_train, y_train)
    # feature importance in logistic model and accuracy of model
    predicted_y_test= model_logistic.predict(X_test)
    print("Accuracy of logistic regression model on test data:",metrics.accuracy_score(y_test, predicted_y_test))
    # as data is equally balanced we can take accuracy as performance mesure
    print('coefficients/ feature importance', model_logistic.coef_)
    p= model_logistic.coef_[0:1,:]
    dd= pd.DataFrame({'VarImp': p.flatten()}, index=X.columns)
    return(dd)    

# random forest with recursive feature engineering
def rfe_random_forest(sensor_data, n):
    X=sensor_data[sensor_data.columns[1:]]
    Y= sensor_data.class_label.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model_rf= RandomForestClassifier()
    rfe_rf= RFE(model_rf, n)
    rfe_rf= rfe_rf.fit(X_train,y_train)
    res = dict(zip(sensor_data.columns[1:], rfe_rf.ranking_))
    print('imporance of sensors in classification:- ' + str(res))
    print('\n\n model accuracy of ref+RF', metrics.accuracy_score(y_test, rfe_rf.predict(X_test)))
    dataset_with_featureImp= pd.DataFrame(rfe_rf.ranking_, index= X.columns)
    return(dataset_with_featureImp)

### feature imporatnce from RF ( white box model)
def rf_var_imp (sensor_data):
    X=sensor_data[sensor_data.columns[1:]]
    Y= sensor_data.class_label.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model_rf= RandomForestClassifier()
    model_rf.fit(X_train,y_train)
    predicted_y_test= model_rf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of RF:",metrics.accuracy_score(y_test, model_rf.predict(X_test)))
    Var_imp_df= pd.DataFrame({'VarImp': model_rf.feature_importances_}, index=X.columns)
    print(sns.barplot(x= 'VarImp', y= Var_imp_df.index, data= Var_imp_df).set_title('relative importance of sensors'))
    print(Var_imp_df)
    return(Var_imp_df)

### explainability through xgboost
def xgboost_imp ( sensor_data) :
    
    xgc = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                            objective='binary:logistic', random_state=42)
    X=sensor_data[sensor_data.columns[1:]]
    Y= sensor_data.class_label.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    xgc.fit(X_train, y_train)
    xgc_predictions=xgc.predict(X_test)
    print(' testing accuracy of xgboost= ', (metrics.accuracy_score(xgc_predictions, y_test)))

    #plotting variable importance based on feature weight, split mean gain, sample coverage
    fig = plt.figure(figsize = (16, 12))
    title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

    ax1 = fig.add_subplot(2,2, 1)
    xgb.plot_importance(xgc, importance_type='weight', ax=ax1)
    t=ax1.set_title("Feature Importance - Feature Weight")

    ax2 = fig.add_subplot(2,2, 2)
    xgb.plot_importance(xgc, importance_type='gain', ax=ax2)
    t=ax2.set_title("Feature Importance - Split Mean Gain")

    ax3 = fig.add_subplot(2,2, 3)
    xgb.plot_importance(xgc, importance_type='cover', ax=ax3)
    t=ax3.set_title("Feature Importance - Sample Coverage")
     
    df1 = pd.DataFrame(list(zip(xgc.get_booster().get_score(importance_type="gain").keys(), xgc.get_booster().get_score(importance_type="gain").values())),
               columns =['Name', 'val'])

    df1 = pd.DataFrame(list(zip(xgc.get_booster().get_score(importance_type="gain").keys(), xgc.get_booster().get_score(importance_type="gain").values())),
                   columns =['Name', 'val_gain'])
    df2 = pd.DataFrame(list(zip(xgc.get_booster().get_score(importance_type="weight").keys(), xgc.get_booster().get_score(importance_type="gain").values())),
                   columns =['Name', 'val_weight'])
    df3 = pd.DataFrame(list(zip(xgc.get_booster().get_score(importance_type="cover").keys(), xgc.get_booster().get_score(importance_type="gain").values())),
                   columns =['Name', 'val_cover'])
    df=[df1,df2,df3]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Name'],
                                                how='outer'), df)
    print(df_merged)
    df_merged.sort_values(by=['val_gain'], ascending=False, inplace= True)
    return(df_merged)

def logstic_rule_reg (sensor_data):
    fb = FeatureBinarizer(negations=True, returnOrd=True)
    dfTrain, dfTrainStd = fb.fit_transform(sensor_data[sensor_data.columns[1:]])
    y=sensor_data['class_label']
    lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
    # Train, print, and evaluate model
    lrr.fit(dfTrain, y, dfTrainStd)
    print('Training accuracy:', accuracy_score(y, lrr.predict(dfTrain, dfTrainStd)),'\n')
    #print('Test accuracy:', accuracy_score(yTest, lrr.predict(dfTest, dfTestStd)))
    print('Probability of Y=1 is predicted as logistic(z) = 1 / (1 + exp(-z))')
    print('where z is a linear combination of the following rules/numerical features:')
    print(lrr.explain())
    return(lrr.explain)
    list_columns= list(sensor_data.columns)
    for col in list_columns:
        list_columns= list(sensor_data.columns)
        lrr.visualize(sensor_data, fb, [col])
        
def shap_dt_explanation(sensor_data):
    X=sensor_data[sensor_data.columns[1:]]
    Y= sensor_data.class_label
    # taking DT model
    clf= DecisionTreeClassifier()
    clf.fit(X, Y)
    explainer = shap.KernelExplainer(clf.predict_proba, X)
    shap_values = explainer.shap_values(X.iloc[0:10], nsamples=100)
    return(explainer, shap_values)

def shap_plot( sensor_data, n, expl, sha):
    X=sensor_data.iloc[:, 1:]
    l=shap.force_plot(expl.expected_value[0], sha[0][n,:], X.iloc[n,:])
    return(l)
 

