import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from  sklearn.linear_model import LogisticRegressionCV
import soft_impute
#from ppca import PPCA
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import general_functions
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
import multiProcFuncs


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

def LogisticReg(trainData,trainCategory,penalty='l1',feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0,k_Range=None):

		classifier=LogisticRegressionCV(penalty=penalty,solver='liblinear')
		parameters={'penalty':['l1','l2']}
		if feature_sel:
				parameters={'classifier__penalty':['l1','l2']}
				if k_Range is not None:
					parameters['select_k']=k_Range
				feature_sel = SelectKBest(score_func=score_func, k=kBest)
				pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])
		else:
			pipeline=classifier
		if gridSearch:
			pipeline = GridSearchCV(pipeline, parameters)
		pipeline.fit(trainData,trainCategory)

		if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
		return pipeline

def NaiveBayes(trainData,trainCategory,feature_sel=1,score_func=mutual_info_classif,kBest=20,gridSearch=0,k_Range=None ):
	feature_sel = SelectKBest(score_func=score_func, k=kBest)
	pipeline = Pipeline([('select', feature_sel),('classifier', GaussianNB())])
	if gridSearch and k_Range is not None:
		parameters={'select__k':k_Range}
		pipeline = GridSearchCV(pipeline, parameters)
	pipeline.fit(trainData,trainCategory)
	if gridSearch and k_Range is not None:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline

def SVM(trainData,trainCategory,C=1.0, kernel='rbf' , degree=3, gamma='auto',feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0,k_Range=None ): #kernel 'rbf','poly'
	classifier=SVC(C=C,kernel='rbf',degree=degree,gamma=gamma)
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                           #    "gamma": np.logspace(-2, 2, 5)})
	param_grid=[{'kernel': ['rbf'], 'gamma': np.logspace(-2, 2, 5),
                     'C': [1, 10, 100, 1000]},{'kernel': ['poly'], 'C': [1, 10, 100, 1000],'degree':[1,3,5]}]
	if feature_sel:
		param_grid=[{'classifier__kernel': ['rbf'], 'classifier__gamma': np.logspace(-2, 2, 5),
                     'classifier__C': [1, 10, 100, 1000]},{'classifier__kernel': ['poly'], 'classifier__C': [1, 10, 100, 1000],'classifier__degree':[1,3,5]}]	

		if k_Range is not None:
			param_grid[0]['select__k']=k_Range
			param_grid[1]['select__k']=k_Range
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])
	else:
		pipeline=classifier
	if gridSearch:
			pipeline = GridSearchCV(pipeline, param_grid=param_grid)
	pipeline.fit(trainData,trainCategory)
	if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline


def RandomForest(trainData,trainCategory, n_estimators=10,feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0,k_Range=None  ): 
	classifier=RandomForestClassifier(n_estimators=n_estimators)
	parameters={'n_estimators':[5,30]}
	if feature_sel:
		parameters={'classifier__n_estimators':[5,30]}
		if k_Range is not None:
			parameters['select__k']=k_Range
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])
	else:
		pipeline=classifier
	if gridSearch:
			pipeline = GridSearchCV(pipeline, parameters)
			
	pipeline.fit(trainData,trainCategory)
	if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline


def ExtraTrees(trainData,trainCategory, n_estimators=10,feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0 ,k_Range=None ): 
	classifier=ExtraTreesClassifier(n_estimators=n_estimators)
	parameters={'n_estimators':[5,30]}
	if feature_sel:
		parameters={'classifier__n_estimators':[5,30]}
		if k_Range is not None:
			parameters['select__k']=k_Range
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])
	else:
		pipeline=classifier
	if gridSearch:
			pipeline = GridSearchCV(pipeline, parameters)

	pipeline.fit(trainData,trainCategory)
	if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline


def AdaBoost(trainData,trainCategory, n_estimators=100,feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0,k_Range=None  ): 
	classifier=AdaBoostClassifier(n_estimators=n_estimators)
	parameters={'learning_rate' : [.01,.1,.5],'n_estimators' : [10,100,500]}
	if feature_sel:
		parameters={'classifier__learning_rate' : [.01,.1,.5],'classifier__n_estimators' : [10,100,500]}
		if k_Range is not None:
			parameters['select__k']=k_Range
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])

	else:
		pipeline=classifier
	if gridSearch:
			pipeline = GridSearchCV(pipeline, parameters)
	pipeline.fit(trainData,trainCategory)
	if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline


def GradientBoosting(trainData,trainCategory, n_estimators=100,max_depth=3,feature_sel=1,score_func=mutual_info_classif, kBest=20,gridSearch=0,k_Range=None ): 
	classifier=GradientBoostingClassifier(n_estimators=n_estimators,max_depth=3)
	parameters={'learning_rate' : [.01,.1,.5],'n_estimators' : [10,100,500],'max_depth':[1,3,5]}
	if feature_sel:
		parameters={'classifier__learning_rate' : [.01,.1,.5],'classifier__n_estimators' : [10,100,500],'classifier__max_depth':[1,3,5]}
		if k_Range is not None:
			parameters['select__k']=k_Range
		feature_sel = SelectKBest(score_func=score_func, k=kBest)
		pipeline = Pipeline([('select', feature_sel),('classifier', classifier)])

	else:
		pipeline=classifier
	if gridSearch:
			pipeline = GridSearchCV(pipeline, parameters)

	pipeline.fit(trainData,trainCategory)
	if gridSearch:
			pipeline.estimator.set_params(**pipeline.best_params_)
			return pipeline.estimator
	return pipeline

def prediction(clf,testData,testCategory):
	predict=clf.predict(testData)
	#print f1_score(testCategory,predict)
	#print classification_report(testCategory,predict)
	#print (precision_score(testCategory,predict,pos_label=0),recall_score(testCategory,predict,pos_label=0),precision_score(testCategory,predict,pos_label=1),recall_score(testCategory,predict,pos_label=1))
	tmp=np.array([precision_score(testCategory,predict,pos_label=0),recall_score(testCategory,predict,pos_label=0),precision_score(testCategory,predict,pos_label=1),recall_score(testCategory,predict,pos_label=1)])
	print tmp
	return tmp


def runClassifier(clf):
	num_cross_validation_folds=25
	val=20
	K_max = 90
	K_min = 5
	K_space = 5

	k_values = range(K_min, K_max, K_space)


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




	
	#locals()["myfunction"]

	
	counter=-1
	tmpClf=locals()[clf](data, category,kBest=val,gridSearch=1,k_Range=k_values)
	classiferStatistics=np.zeros([num_cross_validation_folds,4])
	for training_indices, testing_indices in cross_val_fold.split(data):
		counter+=1
		tmpClf.fit(data[training_indices], category[training_indices])
		classiferStatistics[counter]+= prediction(tmpClf,data[testing_indices],category[testing_indices])
	print clf,val,np.mean(classiferStatistics[j],axis=0)
	    # nB=NaiveBayes(data[training_indices], category[training_indices],kBest=val)
	    # prediction(nB,data[testing_indices],category[testing_indices])
	    # print 'Naive Bayes'
	    
	    # gB=GradientBoosting(data[training_indices], category[training_indices])
	    # print 'Gradient Boosting'
	    # prediction(gB,data[testing_indices],category[testing_indices])
	    # aB=AdaBoost(data[training_indices], category[training_indices])
	    # print 'Ada Boosting'
	    # prediction(aB,data[testing_indices],category[testing_indices])
	    # randForest=RandomForest(data[training_indices], category[training_indices])
	    # print 'Random Forest'
	    # prediction(randForest,data[testing_indices],category[testing_indices])
	    # exTree=ExtraTrees(data[training_indices], category[training_indices])
	    # print 'Extra random Forest'
	    # prediction(exTree,data[testing_indices],category[testing_indices])

	f=open('../data/classifiers'+clf+'.pkl','w')
	pickle.dump([clf,tmpClf,classiferStatistics],f)
	f.close()

	# Need to look more carefully at these. The are also quite slow!     
		    #svmRBF=SVM(data[training_indices], category[training_indices])
		    # print 'SVM rbf kernel'
		    # prediction(svmRBF,data[testing_indices],category[testing_indices])
		    # print 'SVM poly kernel'
		    # svmPOLY=SVM(data[training_indices], category[training_indices],kernel='poly')
		    # prediction(svmPOLY,data[testing_indices],category[testing_indices])

	    # logRl1=LogisticReg(data[training_indices], category[training_indices],penalty='l1')
	    # prediction(logRl1,data[testing_indices],category[testing_indices])
	    # logRl2=LogisticReg(data[training_indices], category[training_indices],penalty='l2')
	    # prediction(logRl2,data[testing_indices],category[testing_indices])
	    # logRl1_featSel=LogisticReg(data[training_indices], category[training_indices],penalty='l1',feature_sel=1)
	    # prediction(logRl1_featSel,data[testing_indices],category[testing_indices])
	    # logRl2_featSel=LogisticReg(data[training_indices], category[training_indices],penalty='l2',feature_sel=1)
	    # prediction(logRl2_featSel,data[testing_indices],category[testing_indices])


	#mutInf=sklearn.feature_selection.mutual_info_classif(data,category)




classifiers=['RandomForest','ExtraTrees','NaiveBayes','GradientBoosting','AdaBoost','LogisticReg','SVM'] # 
#for j,clf in enumerate(classifiers):

multiProcFuncs.parmap(runClassifier,classifier)

