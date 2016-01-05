import numpy as np


#Load dataset from file

ifile = open('seeds.tsv')

features = []
labels = []

for line in ifile:
  token = line.strip().split('\t')
  features.append([ float(tv) for tv in token[:-1]])
  labels.append(token[-1])

features = np.array(features)
labels = np.array(labels)

print (features.shape)
print (features[:6])
print (labels.shape)
print (labels[:6])

#KNN Model

def learn_model(k,features,labels):
  return k,features.copy(),labels.copy()


#Find the Class of max nieghbors ( for KNN )
def plurality(xs):
  from collections import defaultdict
  counts = defaultdict(int)
  for x in xs:
    counts[x] += 1
  maxv = max(counts.values())
  for k,v in counts.items():
    if v == maxv:
      return k


#Calculate distance
def apply_model(features,model):
  k,train_feats,labels = model  
  results = []
  for f in features:
    label_dist = []
    for t,ell in zip(train_feats,labels):
      label_dist.append( (np.linalg.norm(f-t), ell) )
    label_dist.sort(key=lambda d_ell:d_ell[0])
    label_dist = label_dist[:k]
    results.append(plurality([ell for _,ell in label_dist]))
  return np.array(results)


def accuracy(features, labels, model):
  preds = apply_model(features, model)
  return np.mean(preds == labels)


# Cross Validate


def cross_validate(features, labels):
  error = 0.0
  for fold in range(10):
    training = np.ones(len(features), bool)
    training[fold::10] = 0
    testing = ~training
    model = learn_model(1,features[training], labels[training])
    test_error = accuracy(features[testing],labels[testing],model) 
    error += test_error   
  return error / 10.0


error = cross_validate(features, labels)
print('Ten fold Cross Validated error is {0:.1%}'.format(error))


features -= features.mean(0)
features /= features.std(0)
error = cross_validate(features, labels)
print('Ten fold cross validated error after z-scoring is {0:.1%}'.format(error))
