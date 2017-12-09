import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

fname = 'reducedsample.txt'
if len(sys.argv) > 1:
    fname = sys.argv[1]

def process_line(l):
    return [int(n) for n in l.strip().split()]
    # dat = l.strip().split()
    # x = dat[:dim]
    # y = dat[dim:]
    # return (x,y)

def process_words(wl):
    return wl.strip().split()

with open(fname) as f:
    dim = int(next(f))
    words = process_words(next(f))
    dat = [process_line(l) for l in f]
    # x,y = [],[]
    # for l in f:
    #     xdat,ydat = process_line(l,dimensionality)
    #     x.append(xdat)
    #     y.append(ydat)

dat = np.array(dat)
x = dat[:,:dim]
y = dat[:,dim:]
print(x.shape,y.shape)
# x = np.array(x)
# y = np.array(y)

print('Fitting Classifier: Random Forest Classifier')
clf = RFC(n_estimators=128,max_depth=50,random_state=0)
clf.fit(x,y)
print('Accuracy:',clf.score(x,y))