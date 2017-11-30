# COGS 118B Final Project - Classifying Genre based on Lyrics with NLP Models

## Requirements

So far, the project is using Python 3.6.x, with several python modules as dependencies. To use the helper functions, you need both Python3 and Pip. After those are installed, run the following command to install required modules in the project directory:

```sh
pip install -r ./requirements.txt
```

Next, you'll want to download the raw dataset to begin working with [from kaggle](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/data).

## Getting Started

After downloading the raw data and installing requirements, you can run the following to get a usable (reduced) dataset for the project:

```sh
python preprocess_lyric_newlines.py lyrics.csv
```

This will execute a python script that processes the raw file, stripping unnecessary characters (e.g. punctuation, newlines etc.) in the lyrics themselves, and concatenates the lyrics for a given datum with its label. The output file will be named lyricscleaned.csv, and will conform to the following format for each line:

**genre:lyrics**

So, you can read in a single data point by reading in a line, splitting on ':' and using the first returned value as a label and the second value as a set of features.

## Helper Loading Functions

Included is a small module with some helper loading functions that will do the previously mentioned splitting and some other useful things in one unified location, so we won't make mistakes later on in loading the data. The first three helper loading functions are the following:

* **load(filename)** - loads a preprocessed file, returns a tuple with (dat,genres,words)
  * dat - A numpy array with shape (n,2) including the labels on the first column
  * genres - A list of all unique genres in the file
  * words - A list of all unique words in the file
* **load_one_hot_lyrics(filename)** - loads a preprocessed file, encodes the genre labels using a one-hot encoding. Returns a tuple with (x,y,genres,words)
  * x - a numpy array where each element contains the lyrics for a given song represented as a single space-delimited string
  * y - a numpy array where each element contains a one-hot coded representation of the genre for a given song. The ordering of the genres in the one-hot encoding scheme follows the same ordering as in the returned *genres* list
  * genres - same as in load(fname)
  * words - same as in load(fname)
* **load_bag_of_words(filename)** - loads a preprocessed file, calculates word counts for 1-gram in a bag of words representation as well as representing the genre in a one-hot encoding scheme. Returns a tuple with (x,y,genres,words)
  * x - a numpy array with shape (n,w) where n is the number of songs in the file, w is the number of unique words in the file. Each row is array-like with shape (1,w) and each position represents the number of occurences of a given word in the same ordering as the returned *words* list
  * y - same as in load_one_hot_lyrics(fname)
  * genres - same as in load_one_hot_lyrics(fname)
  * words - same as in load_one_hot_lyrics(fname)

## Demo Program

A simple demo classifier is build in demo_classifier.py that demonstrates use of the helper functions to load the data and successfully build a classifier. To run it simply run the following:

```sh
python demo_classifier.py
```

## To Do

Loading the whole dataset using the helper functions takes very long and can be killed by the host OS. Since the dimensionality of the dataset is large and sparse, and since we have many samples in the dataset, running something like PCA on the whole dataset probably wouldn't work. However, we could consider dimensionality reduction using a large subset of the data, possibly with ~50,000 samples instead of all 300,000+ samples.

For Markov models, since we need the whole sequence to predict, I'm unsure what dimensionality reduction can be accomplished, but that's something to look into.

The bag of words representation is just a starting point as well, TF-IDF vectors could be calculated after PCA or some other dimensionality reduction.

In any case, testing models should be done on a small subset for now.