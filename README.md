# Project (In Progress) for 6.867 - Machine Learning (MIT) 

# Seizure Forecasting and Feature Extraction Using EEG Data

### Jordan Harrod and Daphne Schlesinger 


## Milestone 1

Dataset:

We are using the “Epileptic Seizure Recognition Data Set” from the UC Irvine ML Repository. While the electroencephalogram (EEG) data provided has been divided into 1 second segments, a link to the original dataset is available on the page. This dataset is grouped into five classes of 100 patients, with one class containing seizure data and the other four containing non-seizure data.  This gives us a total of 500 patients and 11500 samples (23 seconds per patient x 500 patients). The classes represent the following patient categories:

Class 1 - Recording of Seizure Activity

Class 2 - Recording of Non-Seizure Activity from Area of the Brain Containing Tumor

Class 3 - Recording of Non-Seizure Activity from Area of the Brain Not Containing Tumor (where the location of the tumor is known) 

Class 4 - Recording of Non-Seizure Activity with Patient Eyes Closed 

Class 5 - Recording of Non-Seizure Activity with Patient Eyes Open

Plan of Action:

Search literature for typical features and windows used in analysis of EEG data

Filter and extract a set of features from the waveforms as per results from step 1

Select a classifier

Run experiments, with varied subsets of the features, windows for feature extraction, and hyperparameters

Examine results and iterate steps 2-4

Problems to Solve:

What length of EEG segment is optimal for feature extraction and classification?

Which waveform features are most important to classification? Can we draw any conclusions about the underlying mechanisms of the seizure activity from this information?

What are the hyperparameters for a given model that produce the best results? Is there any intuition for the selection of these parameters?



## Milestone 2

Machine Learning Algorithms:

Cost-Sensitive SVM - Based on the distribution of the dataset (heavily skewed towards non-epilepsy samples), we would implement a cost-sensitive SVM to penalize false negatives more severely than false positives. 

Recurrent Neural Net - Given that this data is temporal/sequence data, we would implement an RNN 

Random Forest - Previous literature has also shown Random Forest to be an effective classifier for epilepsy data. 

Evaluation: 
Cross validation to analyze/compare model accuracy

False positives/false negatives, because in the clinical context false negatives have a higher cost

Feature selection to refine features based on importance	

## Milestone 3

Software Infrastructure: 

All work will be completed in Python. Analysis of the signals to produce features can be accomplished with packages like numpy and scipy.

Cost-Sensitive SVM: scikit-learn package sklearn.svm.SVM, which allows the specification of the parameter class_weight for cost sensitivity

Recurrent Neural Network: RNN can be constructed in Keras

Random Forest: scikit-learn package sklearn.ensemble.RandomForestClassifier, which allows for fairly specific control of the hyper-parameters of the trees

Division of Labor:

Jordan:  Will perform pre-processing on data for spectral features. Will implement the Cost-Sensitive SVM, perform parameter optimization, and look at false positives on this model. Will co-implement RNN with Daphne in Keras, co-perform feature extraction, and co-analyze false positives. 

Daphne: Will perform pre-processing on data for temporal features. Will implement the Random Forest, perform parameter optimization, and look at false positives on this model.  Will co-implement RNN with Jordan in Keras, co-perform feature extraction, and co-analyze false positives. 

Open Questions:

What are the best choices of hyperparameters for each algorithm to be tested? 

Are there “intelligent” ways we can pick parameters, or a range of parameters, rather than sweeping through a large number of possibilities? I’m thinking in particularly about the parameters for Random Forest, like maximum tree depth.

Will spectral features be more informative than temporal, or vice versa? 

