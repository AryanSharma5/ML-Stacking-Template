# ARYAN SHARMA

import numpy as np
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1score

#-------------------------------------------------
#-------------------------------------------------

def transfomer(y, func=None):
	if func is None:
		return y
	else:
		return func(y)

#--------------------------------------------------
#-------------------STACKING-----------------------

def Stacking(models, X_train, Y_train, X_test,
			 regression=True, transfomer_target=None, metric=None
			 , n_folds=4, Stratified=False, shuffle=False,
			 verbose=0, random_state=101, transfomer_pred=None):
	
	if regression and verbose>0:
		print('task : [regression]')		
	elif not regression and verbose>0:
		print('task : [classification]')

	if metric is None and regression:
		metric=mean_absolute_error
	elif metric is None and not regression:
		metric=accuracy_score

	if verbose>0:
		print(f'metric : {metric.__name__}\n')

	if Stratified and not regression:
		kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state=random_state)
	else:
		kf = KFold(len(y_train), n_folds, shuffle=shuffle, random_state=random_state)		

	S_train = np.zeros((X_train.shape[0], len(models)))
	S_test = np.zeros((X_test.shape[0], len(models)))
	

	for model_counter, model in enumerate(models):
		if verbose>0:
			print(f'model {model_counter} : {model.__class__.__name__}')

		S_test_temp = np.zeros((X_test.shape[0], len(kf)))

		for fold_counter, (tr_index, te_index) in enumerate(kf):
			x_tr = X_train[tr_index]
			y_tr = y_train[tr_index]
			x_te = X_train[te_index]
			y_te = y_train[te_index]

			model = model.fit(X_tr, transformer(y_tr, func=transfomer_target))

			S_train[te_index, model_counter] = transformer(model.predict(X_te), func=transfomer_pred)
			S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func=transfomer_pred)

			if verbose>0:
				print(f'   fold {fold_counter} : {metric(y_te, S_train[te_index, model_counter])}')

		if regression:
			S_test[:, model_counter] = np.mean(S_test_temp, axis=1)
		else:
			S_test[:, model_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

		if verbose>0:
			print('    ----')
			print(f'   MEAN : {metric(y_train, S_train[:, model_counter])}\n')

	return (S_train, S_test)		