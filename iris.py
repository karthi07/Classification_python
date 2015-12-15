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

#if features[:,2] < 2:
# print "Iris setosa"
#else :
# print "Virginica or Versicolour"

#classify virginica and versicolour

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels=='Virginica')

best_acc = -1.0
for fi in xrange(features.shape[1]):
  thresh = features[:,fi]
  thresh.sort()
  for t in thresh:
    pred = (features[:,fi]>t)
    acc = (pred == virginica).mean()
    if acc > best_acc:
      best_acc = acc
      best_fi = fi
      best_t = t


print best_t,best_fi,best_acc
      
