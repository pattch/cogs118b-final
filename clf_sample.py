import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn import svm

fname = 'reducedsample.txt'
if len(sys.argv) > 1:
    fname = sys.argv[1]

def process_line(l):
    return [float(n) for n in l.strip().split()]

def process_words(wl):
    return wl.strip().split()

with open(fname) as f:
    dim = int(next(f))
    words = process_words(next(f))
    dat = [process_line(l) for l in f]

dat = np.array(dat)
# Get x
x_ = dat[:,:dim]
x_ = np.array(x_,dtype=np.float64)
# Get y
y_ = dat[:,dim:]
# Train Test Split
x,xt,y,yt = train_test_split(x_,y_,test_size=0.1,random_state=0)

# for i in range(x.shape[0]):
#     n = norm[i]
#     if sum(n):
#         x[i] /= n

print('Shape of the Data',x.shape,y.shape)

for i in range(1,11):
    clf = RFC(n_estimators=128,max_depth=i*10,random_state=0)
    print('Fitting Classifier: Random Forest Classifier')
    print('Max Depth',str(i*10))
    clf.fit(x,y)
    print('Training Accuracy:',clf.score(x,y))
    print('Testing Accuracy:',clf.score(xt,yt))

# for g in np.linspace(0.01,2,10):
#     srbf = svm.SVC(kernel='rbf',gamma=g)
#     print('Fitting Classifier: SVM')
#     print('Gamma',str(g))
#     srbf.fit(x,y)
#     print('Training Accuracy:',srbf.score(x,y))
#     print('Testing Accuracy:',srbf.score(xt,yt))

# for i in range(1,5):
#     clf = KNN(n_neighbors=i)
#     print('Fitting Classifier: K Nearest Neighbors Classifier')
#     print('Neighbors',str(i))
#     clf.fit(x,y)
#     print('Training Accuracy:',clf.score(x,y))
#     print('Testing Accuracy:',clf.score(xt,yt))
