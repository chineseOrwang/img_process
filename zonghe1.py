# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:07:12 2018

@author: 王健学
""" 

import time
import numpy as np
import pandas as pd

def test_model(model):
    print('默认分类器:',model)
    #根据模型定参数
    params=dict()
    name=str(model).split('(')[0]
    if name=='KNeighborsClassifier':
        name='KNN'
        n_neighbors_range=list(range(3,20))
        leaf_range=list(range(1,10))
        weight_options=['uniform','distance']
        algorithm_options=['auto','ball_tree','kd_tree','brute']
        params=dict(n_neighbors = n_neighbors_range
                    ,weights = weight_options
                    ,algorithm=algorithm_options
                    ,leaf_size=leaf_range
                    )
    if name=='LogisticRegression':
        name='LogR'
        Cs = list((1,10,100,1000))
        max_iters = list((100,300,500,700,900,1100))
        solvers = ['newton-cg','lbfgs','liblinear','sag']
        params=dict(C=Cs,max_iter=max_iters,solver=solvers)
    if name=='GaussianNB':
        pass
    if name=='BernoulliNB':
        pass
    if name=='SVC':
        params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovo', 'ovr']},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovo', 'ovr']},
                    {'kernel': ['poly'], 'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovo', 'ovr']},
                    {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovo', 'ovr']}]
    if name=='DecisionTreeClassifier':
        name='DecisionTree'
        criterion_options=['gini','entropy']
        splitter_options=['best','random']
        max_depth_range=list(range(1,8))
        max_features_option=['auto','sqrt','log2']
#        max_depth_range=list(range(10,101))     #样本或特征少，不使用
#        min_samples_split_range=list(range(2,3))     #样本或特征少，不使用
#        min_samples_leaf_range=list(range(1,2))     #样本或特征少，不使用
        params=dict(criterion=criterion_options
                    ,splitter=splitter_options
                    ,max_features=max_features_option
                    ,max_depth=max_depth_range
#                    ,min_samples_split=min_samples_split_range
#                    ,min_samples_leaf=min_samples_leaf_range
                    )
    if name=='RandomForestClassifier':
        name='RandomForest'
        n_estimators_options=list(range(10,20))
        criterion_options=['entropy','gini']
        max_depth_range=list(range(1,5))
        max_features_option=['auto','sqrt','log2']
        params = dict(n_estimators=n_estimators_options
                            ,criterion =criterion_options
                            ,max_depth =max_depth_range
                            ,max_features=max_features_option
                            )
    if name=='AdaBoostClassifier':
        name='Adaboost'
        n_estimators_options=list(range(40,61,10))
        learning_rate_options=[0.5,1.0,1.5]
        params = dict(n_estimators=n_estimators_options
                            ,learning_rate=learning_rate_options
                            )
    if name=='GradientBoostingClassifier':
        name='GradientBoosting'
        n_estimators_options=list(range(5,51,10))
        #learning_rate_range=list(range(0.05,0.31,0.05))
        learning_rate_option=[0.05,0.1,0.15,0.2,0.3]       
        criterion_options=['friedman_mse','mse','mae']
        max_depth_range=list(range(1,4,1))
        #max_features_option=['auto','sqrt','log2']
        #max_features_option=list(range(2,5,1))
        #subsample_range=list(range(0.6,1,0.1))
        subsample_option=[0.6,0.8,1]
        params = dict(n_estimators=n_estimators_options
                      ,learning_rate=learning_rate_option
                      ,criterion =criterion_options
                      ,max_depth =max_depth_range
#                      ,max_features=max_features_option
                      ,subsample=subsample_option
                      )
    if name=='XGBClassifier':
        name='XGboost'
        params= {"max_depth": [10,30,50],
                 "min_child_weight" : [1,3,6],
                 "n_estimators": [200],
                 "learning_rate": [0.05, 0.1,0.16]
                }
    #调参开始
    from sklearn.grid_search import GridSearchCV
    grid_model=GridSearchCV(model,params,cv=10,scoring='accuracy',verbose=1)
    return grid_model,name


from sklearn.model_selection import train_test_split
if __name__=='__main__':
    '''数据准备部分'''
    #准备数据
    data = pd.read_excel('shuangmei_2.xlsx')
    X,Y = data.iloc[:,1:20],data.iloc[:,0]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=3)
    #数据预处理
    from sklearn import preprocessing
    scaler=preprocessing.StandardScaler().fit(X_train)   #可能加入pipline,但我不会
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    '''模型准备'''
    from sklearn import neighbors     #KNN
    from sklearn.linear_model import LogisticRegression      #Logistic
    from sklearn.naive_bayes import GaussianNB,BernoulliNB    #朴素贝叶斯
    from sklearn.svm import SVC       #支持向量机
    from sklearn.tree import DecisionTreeClassifier     #决策树
    from sklearn.ensemble import RandomForestClassifier     #随机森林
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier    #GBDT
    import xgboost as xgb              #xgboost
    
    models=(
            neighbors.KNeighborsClassifier()      #1 KNN
            ,LogisticRegression()                 #2 Logistic
            #3 softmax
            ,GaussianNB()                         #4 贝叶斯一
            ,BernoulliNB()                        #4 贝叶斯二
            #4 贝叶斯三
            ,SVC()                                #5 支持向量机
            ,DecisionTreeClassifier()             #6 决策树
            ,RandomForestClassifier()             #7 随机森林
            ,AdaBoostClassifier()                 #8 Adaboost
            ,GradientBoostingClassifier(random_state=1)   #9 GBDT
            ,xgb.XGBClassifier()                      #10 XGBoost
           )
    '''每个模型调参及预测与评价'''
#    pred_result=pd.DataFrame()      #未知结果使用
    pred_result=pd.DataFrame({"real value":np.transpose(Y_test)})     #已知结果加入对比
    evaluate_result=pd.DataFrame(columns=("accuracy","precision","recall","f1-score","time"))
    f=open('canshu.txt','w')    #清空输出文件
    print('',file=f)
    f.close()
    for model in models:
        print('------------------------------分类器开始----------------------------------')
        starttime=time.time()
        grid_model,name=test_model(model)
        grid_model.fit(X_train,Y_train)        #可能加入pipline,但我不会
        #收集信息
        best_score = grid_model.best_score_
        best_params = grid_model.best_params_
        best_estimator = grid_model.best_estimator_
        '''模型训练及预测,经GridSearchCV搜索的分类器已储存在best_estimator'''
        best_estimator.fit(X_train,Y_train)        #可能加入pipline,但我不会
        #训练和储存模型
        '''预测、评价及储存输出数据'''
        from sklearn import metrics
        from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
        from sklearn.cross_validation import cross_val_score
        from sklearn.metrics import precision_score
        Y_pred=best_estimator.predict(X_test)
        #评价模型
        accuracy_score = accuracy_score(Y_test,Y_pred)
        confusion_matrix = confusion_matrix(Y_test,Y_pred,labels=[0,1,2])
#        precision = metrics.precision_score(Y_test,Y_pred)#二分类
#        recall = metrics.recall_score(Y_test,Y_pred)#二分类
#        f1_score = metrics.f1_score(Y_test,Y_pred)#二分类
        precision = metrics.precision_score(Y_test,Y_pred,average="macro")#多分类
        recall = metrics.recall_score(Y_test,Y_pred,average="macro")#多分类
        f1_score = metrics.f1_score(Y_test,Y_pred,average="macro")#多分类
        classification_report = classification_report(Y_test,Y_pred)
        classification_report_split=classification_report.split()   #提取矩阵中的值
        endtime = time.time()
        '''
        输出文件
        '''
        
        out="\n 最优分类器:"+str(best_estimator)+\
              "\n The confusion matrix is:\n"+str(confusion_matrix)
        f=open('canshu.txt','a')
        print(out+'\n\n',file=f)
        f.close()

        #储存数据
        pred_result[str(name)]=np.transpose(Y_pred)
        evaluate_result.loc[str(name)]=[accuracy_score
                            ,precision
                            ,recall
                            ,f1_score
                            ,endtime-starttime
                            ]
        print('-----------------------------分类器结束-----------------------------------')
    #文件输出
    writer=pd.ExcelWriter('HZB.xlsx')
    evaluate_result.to_excel(writer,sheet_name="evaluate")
    pred_result.to_excel(writer,sheet_name="predict")
    writer.save()