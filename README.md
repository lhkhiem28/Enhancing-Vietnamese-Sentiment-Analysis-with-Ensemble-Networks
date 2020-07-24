# Enhancing Vietnamese Sentiment Analysis withEnsemble Networks
This is the Pytorch implementation for the paper "Enhancing Vietnamese Sentiment Analysis withEnsemble Networks", which is submitted at the conference SOFSEM 2021.

## Requirement
* python                    3.7.6     
* scikit-learn              0.23.1
* keras                     2.3.1
* torch                     1.5.1+cu101

## Data preparation

* In this work, we use the dataset:
AIVIVN: this is the publish dataset from AIVIVN 2019 Sentiment Challenge, including approximately 160K training reviews with the available labels and 11K testing reviews without the available labels. We manually did labelling for the testing dataset.
