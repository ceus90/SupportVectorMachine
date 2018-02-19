from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

# prepare dataset
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# svm classification
clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0).fit(X_train, y_train)
y_predicted = clf.predict(X_test)

# performance
print "Classification report for %s" % clf
print
print metrics.classification_report(y_test, y_predicted)
print
print "Confusion matrix"
print metrics.confusion_matrix(y_test, y_predicted)

