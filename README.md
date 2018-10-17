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
