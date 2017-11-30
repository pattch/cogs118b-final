import numpy as np
import pandas

def load(fname):
    genres,words = set(),set()
    with open(fname) as f:
        dat = []
        for l in f:
            p = l.strip().split(':')
            dat.append(p)
            genres.add(p[0])
            for word in p[1].split(' '):
                words.add(word)
        dat = np.array(dat)
    genres,words = list(genres),list(words)
    return (dat,genres,words)

def load_one_hot_lyrics(fname):
    dat,genres,words = load(fname)
    x,y = dat[:,1],dat[:,0]
    y = np.array(pandas.get_dummies(y))

    return (x,y,genres,words)