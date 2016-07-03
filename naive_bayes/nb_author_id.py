#!/usr/bin/python

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



from sklearn.naive_bayes import GaussianNB
clf= GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


t0 = time()
pred=clf.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score 
print "Accuracy Score of Naive bayes Algo is ", accuracy_score(pred, labels_test)



