from load_lyrics import load_bag_of_words as load
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier

test_file,lyrics_file = 'test_lyrics.txt','lyrics1000shuf.csv'

def classify(fname):
    print('Loading',fname)
    x,y,genres,words = load(fname)
    print('Fitting Classifier: Random Forest Classifier')
    clf = RFC(n_estimators=128,max_depth=50,random_state=0)
    clf.fit(x,y)
    print('Accuracy:',clf.score(x,y))

    print('Fitting Classifier: Decision Tree Classifier')
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x,y)
    print('Accuracy:',clf.score(x,y))

print('Classifying using Test File')
classify(test_file)
print('Classifying using Subset of Dataset')
classify(lyrics_file)