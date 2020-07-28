# Enhance Vietnamese Sentiment Classification with Deep Learning and Ensemble Techniques

This is the Pytorch implementation for the paper "Enhance Vietnamese Sentiment Classification with Deep Learning and Ensemble Techniques", which is submitted at the conference SOFSEM 2021.

## Requirement

* python                    3.7.6     
* scikit-learn              0.23.1
* keras                     2.3.1
* torch                     1.5.1+cu101

## Data preparation

In this work, we use the dataset:
* AIVIVN: this is the publish dataset from AIVIVN 2019 Sentiment Challenge, including approximately 160K training reviews with the available labels and 11K testing reviews without the available labels. We manually did labelling for the testing dataset.
* The validation dataset is randomly selected from the training dataset, with 20%.
* The dataset is placed at the folders */dataset/aivivn/

## Contact

* Cuong V. Nguyen (cuong.vn08@gmail.com)
* Khiem H. Le (lhkhiem28@gmail.com)
* Binh T. Nguyen (ngtbinh@hcmus.edu.vn)
