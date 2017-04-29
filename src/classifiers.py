import numpy as np
from sklearn.naive_bayes import GaussianNB
from  sklearn.linear_model import LogisticRegressionCV
import soft_impute
from ppca import PPCA
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import general_functions

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier

def softImpute(data,nCompSoft=30,**kargs):

        imputeFnc = soft_impute.SoftImpute(J=nCompSoft, lambda_=0.0)
        fit = imputeFnc.fit(data)
        imputeFnc.predict(data,copyto=True)
        return data
def PPCA(data,nCompPCA=30,**kargs):
        #tmp[np.where(np.isnan(bkgrnd_wNaN.as_matrix()))]=0
        ppca = PPCA(data)
        ppca.fit(d=nCompPCA, verbose=True,min_obs=1)
        means = np.nanmean(ppca.raw,axis=0)
        stds=np.nanstd(ppca.raw, axis=0)
        data[np.where(np.isnan(data))]=(ppca.data*stds+means)[np.where(np.isnan(data))]
        #print tmp.astype('float').shape
        return data

def LogisticReg(trainData,trainCategory,penalty='l1',feature_sel=0,score_func=mutual_info_classif, kBest=20):

		clasifier=LogisticRegressionCV(penalty=penalty,solver='liblinear')
		if feature_sel:
				feature_sel = SelectKBest(score_func=score_func, k=kBest)
				pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
				pipeline.fit(trainData,trainCategory)
				return pipeline
		clasifier.fit(trainData,trainCategory)
		return clasifier

def NaiveBayes(trainData,trainCategory,score_func=mutual_info_classif,kBest=20):
	feature_sel = SelectKBest(score_func=score_func, k=kBest)
	pipeline = Pipeline([('select', feature_sel),('classifier', GaussianNB())])
	pipeline.fit(trainData,trainCategory)
	return pipeline

def SVM(trainData,trainCategory,C=1.0, kernel='rbf' , degree=3, gamma='auto',feature_sel=0,score_func=mutual_info_classif, kBest=20): #kernel 'rbf','poly'
	clasifier=SVC(C=C,kernel='rbf',degree=degree,gamma=gamma)
	if feature_sel:
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
		pipeline.fit(trainData,trainCategory)
		return pipeline
	clasifier.fit(trainData,trainCategory)
	return clasifier


def RandomForest(trainData,trainCategory, n_estimators=10,feature_sel=0,score_func=mutual_info_classif, kBest=20 ): 
	clasifier=RandomForestClassifier(n_estimators=n_estimators)
	if feature_sel:
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
		pipeline.fit(trainData,trainCategory)
		return pipeline
	clasifier.fit(trainData,trainCategory)
	return clasifier


def ExtraTrees(trainData,trainCategory, n_estimators=10,feature_sel=0,score_func=mutual_info_classif, kBest=20 ): 
	clasifier=ExtraTreesClassifier(n_estimators=n_estimators)
	if feature_sel:
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
		pipeline.fit(trainData,trainCategory)
		return pipeline
	clasifier.fit(trainData,trainCategory)
	return clasifier


def AdaBoost(trainData,trainCategory, n_estimators=100,feature_sel=0,score_func=mutual_info_classif, kBest=20 ): 
	clasifier=AdaBoostClassifier(n_estimators=n_estimators)
	if feature_sel:
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
		pipeline.fit(trainData,trainCategory)
		return pipeline
	clasifier.fit(trainData,trainCategory)
	return clasifier


def GradientBoosting(trainData,trainCategory, n_estimators=100,max_depth=3,feature_sel=0,score_func=mutual_info_classif, kBest=20 ): 
	clasifier=GradientBoostingClassifier(n_estimators=n_estimators,max_depth=3)
	if feature_sel:
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', clasifier)])
		pipeline.fit(trainData,trainCategory)
		return pipeline
	clasifier.fit(trainData,trainCategory)
	return clasifier

def prediction(clf,testData,testCategory):
	predict=clf.predict(testData)
	#print f1_score(testCategory,predict)
	print classification_report(testCategory,predict)
	return clf


num_cross_validation_folds=5
val=20
# Read in the data
data, category = general_functions.read_in_data()

# Convert inf to nan for imputation
for i in range(len(data)):
    for j in range(len(data[i])):
        if np.isinf(data[i][j]):
            data[i][j] = float('nan')
data=softImpute(np.array(data))
category=np.array(category)

cross_val_fold = KFold(n_splits=num_cross_validation_folds, shuffle=True)
for training_indices, testing_indices in cross_val_fold.split(data):
    nB=NaiveBayes(data[training_indices], category[training_indices])
    print 'Naive Bayes'
    prediction(nB,data[testing_indices],category[testing_indices])
    gB=GradientBoosting(data[training_indices], category[training_indices])
    print 'Gradient Boosting'
    prediction(gB,data[testing_indices],category[testing_indices])
    aB=AdaBoost(data[training_indices], category[training_indices])
    print 'Ada Boosting'
    prediction(aB,data[testing_indices],category[testing_indices])
    randForest=RandomForest(data[training_indices], category[training_indices])
    print 'Random Forest'
    prediction(randForest,data[testing_indices],category[testing_indices])
    exTree=ExtraTrees(data[training_indices], category[training_indices])
    print 'Extra random Forest'
    prediction(exTree,data[testing_indices],category[testing_indices])
    svmRBF=SVM(data[training_indices], category[training_indices])
    print 'SVM rbf kernel'
    prediction(svmRBF,data[testing_indices],category[testing_indices])
    print 'SVM poly kernel'
    svmPOLY=SVM(data[training_indices], category[training_indices],kernel='poly')
    prediction(svmPOLY,data[testing_indices],category[testing_indices])

# Need to look more carefully at these. The are also quite slow!     
    # logRl1=LogisticReg(data[training_indices], category[training_indices],penalty='l1')
    # prediction(logRl1,data[testing_indices],category[testing_indices])
    # logRl2=LogisticReg(data[training_indices], category[training_indices],penalty='l2')
    # prediction(logRl2,data[testing_indices],category[testing_indices])
    # logRl1_featSel=LogisticReg(data[training_indices], category[training_indices],penalty='l1',feature_sel=1)
    # prediction(logRl1_featSel,data[testing_indices],category[testing_indices])
    # logRl2_featSel=LogisticReg(data[training_indices], category[training_indices],penalty='l2',feature_sel=1)
    # prediction(logRl2_featSel,data[testing_indices],category[testing_indices])


#mutInf=sklearn.feature_selection.mutual_info_classif(data,category)
