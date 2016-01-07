import scipy as sp
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer


DIR = r"data/toy"
posts = [open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]

new_post = 'imaging databases'

vectorizer = CountVectorizer(min_df=1)

X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print('#sample is %d, #features is %d' % (num_samples,num_features))	

new_post_vec = vectorizer.transform([new_post])

def dist(v1,v2):
  delta=v1-v2
  return sp.linalg.norm(delta.toarray())


#Cal doc Similarity with bag of words

best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0, num_samples):
  post = posts[i]
  if post == new_post:
    continue
  post_vec = X_train.getrow(i)
  d = dist(post_vec, new_post_vec)
  print "post %i with dist =%.2f : %s" % (i,d,post)
  if d < best_dist:
    best_dist = d
    best_i = i
    best_doc = post

print ("Best post is %i with dist =%.2f" % (best_i,best_dist))



