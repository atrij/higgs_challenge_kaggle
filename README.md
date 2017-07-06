
Higgs-Boson Challenge
=============================

My solution to the [Higgs Boson Challenge at Kaggle](https://www.kaggle.com/c/higgs-boson)

Atrij Singhal

June-2017

### Introduction

Discovery of the Higgs boson was announced at CERN in July 2012.  High-energy physicists now are on a quest to measure its characteristics, such as the Higgs decay modes, and determine if it fits the current model of nature.

ATLAS is a particle physics experiment installed at the Large Hadron Collider at CERN that searches for new particles and processes using head-on collisions of protons of extraordinarily high energy. The ATLAS experiment has observed a signal of the Higgs boson decaying into two tau lepton particles, but this process is a small signal buried in "background" noise. 

The goal of the Higgs Boson Machine Learning Challenge is to explore the potential of machine learning methods to improve the discovery significance of the experiment.  Using simulated data with features characterizing events detected by ATLAS, the task is to classify events into "tau tau decay of a Higgs boson" versus "background."

A data set of 250,000 "training" events is given, including their weight *w* (a continuous quantity), and their character (signal or background).  In the simulated set, there is a sharp threshold in *w* separating signals (*w* < 0.05) and background (*w* > 0.05).  The challenge asks to classify 550,000 data points in a test set as signal or background, including a likelihood ranking.  The scoring algorithm takes into account the aggregate weights of both "true positives" (signals correctly identified) and "false positives" (background events mistaken for signals), but none of the "negatives" (data evaluated as background, rightly or wrongly).

### My Approach

I use the python Scikit-Learn library. My approach is very simple:

 I use an AdaBoostClassifier on ExtraTreesClassifier. The reason is that because since there are many examples, it benefits from subsampling. GradientBoostingClassifier is too slow and RandomForestClassifier does not take advantage of the subsampling. Subsampling is good when training data is abundant. With many examples, I grid search and cross-validate to use min_samples_split = 100 for min_samples_leaf = 100, which reduces variances a bit. Grid search for optimal parameters for the AdaBoostClassifier and ExtraTreesClassifier. I use inverse log features on some of the features which has no negatives. I expected the model to pick up the rules even without this though. This is actually very easy to get an overfit, if done wrongly. I used the weights to train the solution. This is effectively telling classifers: Hey, look at these, seriously, these are more important. Not all predictions are equal! I picked the 83th percentile to cut for the signals in my solution..

### Implementation Notes
1. Download data from site mentioned above
2. Follow python notebook exploration.ipynb for insight into data and choices maade to choose particular model
3. Run the code Higgs_challenge.py to generate prediction file on test data