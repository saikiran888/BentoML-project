import bentoml
from sklearn import svm
from sklearn import datasets


iris=datasets.load_iris()
X,y =iris.data,iris.target


clf=svm.SVC(gamma='scale')
clf.fit(X,y)

saved_model = bentoml.sklearn.save_model("iris_clf",clf)
print(f"model save:{saved_model}")

#iris_clf:w7sijt4mzcsdxvgy