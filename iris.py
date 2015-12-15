from sklearn.datasets import load_iris
import numpy as np

#Load iris data

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][data['target']]

print features.shape,feature_names

plength = features[:,2]
is_setosa = (labels =='setosa')
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print 'Max Setosa : ',max_setosa
print 'Min Non Setosa : ',min_non_setosa

if features[:,2] < 2:
 print "Iris setosa"
else :
 print "Virginica or Versicolour"
