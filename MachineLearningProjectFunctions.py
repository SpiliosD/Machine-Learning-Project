############################################
## Functions for machine learning project ##
############################################

##Modules##
from imblearn.over_sampling import RandomOverSampler 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC   
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import random 
from sklearn.metrics import precision_recall_fscore_support

##Features##
#Features preprocessing
def datapreproc(xtrain,xtest,ytrain,PCAVar = 0.6):
    scaler = StandardScaler()
    x_trainsc = scaler.fit_transform(xtrain)
    x_testsc = scaler.transform(xtest)
    #Resampling training dataset
    rus = RandomOverSampler(random_state=123)
    x_trainscres, y_trainres = rus.fit_resample(x_trainsc, ytrain)
    #PCA of training features
    pca = PCA(PCAVar)
    pca.fit(x_trainscres)
    x_trainpcascres = pca.transform(x_trainscres)
    x_testpcasc = pca.transform(x_testsc)
    
    return x_trainpcascres,y_trainres, x_testpcasc, scaler, pca

###Machine learning algorithms### 
##Algorithms comparison##
def algcomp(features,x,y,iterations = 5000):
    
    #Split the data in training and test datasets
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
    #Data preprocessing
    x_train,y_train,x_test,scaler, pca = datapreproc(x_train,x_test,y_train) 
    
    #Supporting vector classification
    clf_svm = SVC(probability=True)
    #Logistic regression 
    clf_lr = LogisticRegression(max_iter=iterations)
    #Gaussian Naive Bayes
    clf_gnb = GaussianNB()
    #Decission tree classification
    clf_dt = tree.DecisionTreeClassifier()
    #K nearest neighbours
    clf_knn = KNeighborsClassifier()
    
    clf_svm.fit(x_train,y_train)
    clf_lr.fit(x_train,y_train) 
    clf_gnb.fit(x_train,y_train) 
    clf_dt.fit(x_train,y_train) 
    clf_knn.fit(x_train,y_train) 
    
    y_svm = clf_svm.predict_proba(x_test)[:,1]
    y_lr = clf_lr.predict_proba(x_test)[:,1]
    y_gnb = clf_gnb.predict_proba(x_test)[:,1]
    y_dt = clf_dt.predict_proba(x_test)[:,1]
    y_knn = clf_knn.predict_proba(x_test)[:,1]
    
    false_positive_rate_svm, true_positive_rate_svm, threshold_svm = roc_curve(y_test, y_svm)
    false_positive_rate_lr, true_positive_rate_lr, threshold_lr = roc_curve(y_test, y_lr)
    false_positive_rate_gnb, true_positive_rate_gnb, threshold_gnb = roc_curve(y_test, y_gnb)
    false_positive_rate_dt, true_positive_rate_dt, threshold_dt = roc_curve(y_test, y_dt)
    false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test, y_knn)
    
    plt.figure()
    plt.plot(false_positive_rate_svm, true_positive_rate_svm,'r', label ='SVM')
    plt.plot(false_positive_rate_lr, true_positive_rate_lr, 'g', label ='Logistic regression')
    plt.plot(false_positive_rate_gnb, true_positive_rate_gnb, 'b', label ='Gaussian Naive Bayes')
    plt.plot(false_positive_rate_dt, true_positive_rate_dt, 'c', label ='Decisicion tree classification')
    plt.plot(false_positive_rate_knn, true_positive_rate_knn, 'm', label ='K nearest neighbour')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(features)
    plt.legend()
       
    print('The area under the ROC curve for the')
    print('SVM on', features ,'is', roc_auc_score(y_test, y_svm))
    print('Logistic Regression on', features ,'is', roc_auc_score(y_test, y_lr))
    print('Gaussian Naive Bayes classification on', features ,'is', roc_auc_score(y_test, y_gnb))
    print('Decision Tree classification on', features ,'is', roc_auc_score(y_test, y_dt))
    print('NN on', features ,'is', roc_auc_score(y_test, y_knn))

        
def featurescomp(feature1, feature2, x1train,y1train,x1test,y1test,x2train,y2train,x2test,y2test):
    x1train,y1train,x1test,scaler1,pca1 = datapreproc(x1train,x1test,y1train)
    x2train,y2train,x2test,scaler2,pca2 = datapreproc(x2train,x2test,y2train)
    clf_svm1 = SVC(probability=True)
    clf_svm2 = SVC(probability=True)
    
    clf_svm1.fit(x1train,y1train)
    clf_svm2.fit(x2train,y2train)
    
    y1svm = clf_svm1.predict_proba(x1test)[:,1]
    y2svm = clf_svm2.predict_proba(x2test)[:,1]
    
    false_positive_rate_svm1, true_positive_rate_svm1, threshold_svm1 = roc_curve(y1test, y1svm)
    false_positive_rate_svm2, true_positive_rate_svm2, threshold_svm2 = roc_curve(y2test, y2svm)
    
    plt.figure()
    plt.plot(false_positive_rate_svm1, true_positive_rate_svm1,'r', label = feature1)
    plt.plot(false_positive_rate_svm2, true_positive_rate_svm2,'g', label = feature2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title('Features comarison')
    plt.legend()
    
    print('The area under the ROC curve for')
    print(feature1 ,'is', roc_auc_score(y1test, y1svm))
    print(feature2 ,'is', roc_auc_score(y2test, y2svm))
    
    return clf_svm1,scaler1,pca1, clf_svm2,scaler2,pca2
    

def modelval(model,scaler,pca, number_of_examples):
    
    Data = pd.read_pickle('Data_Chroma_STFTval.pkl')
    Data = Data.dropna()
    
    if number_of_examples > len(Data):
        print('Please select no more than', len(Data), 'examples')
    else:
        n = random.sample(range(0,len(Data)), number_of_examples)
        DataR = pd.DataFrame()
        
        k = 0
        for i in range(len(Data)):
            if Data.iloc[i,1] == 1:
                k+=1
                
        for i in range(len(n)):
           DataR = DataR.append(Data.iloc[n[i]])
        
        x = DataR.iloc[:,2:]
        y = DataR.iloc[:,1]
        
        x = scaler.transform(x)
        x = pca.transform(x)
        
        y_pred = model.predict(x)
        
        Res = precision_recall_fscore_support(y,y_pred)
        Pr = Res[0]
        Re = Res[1]
        F1 = Res[2]
        
        print('Employing the trained model to', number_of_examples, 
              'examples form the validation dataset results to a precision of', round(Pr[1],2),
              ', a recall of', round(Re[1],2),
              ',and in a F1 score of', round(F1[1],2), 'for the detection of bird sounds.')
        print('In the validation dataset', k/len(Data), 'of the audio recordings contains bird sounds.')
        
        return Pr[1],Re[1],F1[1]
        
    
    