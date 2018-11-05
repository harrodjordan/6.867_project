# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import scipy.io as sio
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
import pywt
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Any results you write to the current directory are saved as output.

# Extracting feature vectors from data
# file structure - input/patient_#/ictal train or non-ictal train or test 

# next attempt = decompose signals down to 4th level, average wavelets across channels, do energy, variance, entropy, line length 

file_path1 = "/Users/jordanharrod/Desktop/data/patient_"

def getint(name):
 
    num = name.split('_')
    num = num[2].split('.')
    
    try:
        return int(num[0])
    except ValueError:
        return -1
        
def getint3(name):
 
    num = name.split('_')


    try: 
        num = num[3].split('.')
    except IndexError:
        return -1
    
    try:
        return int(num[0])
    except ValueError:
        return -1

def decomposition(data):

    coeffs = pywt.wavedec(data, 'db4', level=4, axis=0)
    
    level_one = coeffs[-1]
    level_two = coeffs[-2]
    level_three = coeffs[-3]
    level_four = coeffs[-4]
    approx = coeffs[-5]
    
    sum_one = np.sum(level_one, axis = 1)
    sum_two = np.sum(level_two, axis = 1)
    sum_three = np.sum(level_three, axis = 1)
    sum_four = np.sum(level_four, axis = 1)
    approx_sum = np.sum(approx, axis = 1)



    return level_one, level_two, level_three, level_four, approx

def featureExtraction(data):

    length = len(data)

    variance = np.var(data)

    energy = np.sum(np.square(temp))

    L = 0
    
    for j in range(length-1):
        L = L + (temp[j+1,:] - temp[j,:]);

    line_length = np.sum(L/length);


        
    entropy = np.sum(np.square(data)*np.log10(np.square(data)));


    return variance, energy, line_length, entropy 


#def featureExtraction(file_path1=file_path1):
    

test_names = []


ictal_training_dwt_three_variance = []
nonictal_training_dwt_three_variance = []
test_dwt_three_variance = []

ictal_training_dwt_four_variance = []
nonictal_training_dwt_four_variance = []
test_dwt_four_variance = []

ictal_training_dwt_approx_variance = []
nonictal_training_dwt_approx_variance = []
test_dwt_approx_variance = []


ictal_training_dwt_three_energy = []
nonictal_training_dwt_three_energy = []
test_dwt_three_energy = []

ictal_training_dwt_four_energy = []
nonictal_training_dwt_four_energy = []
test_dwt_four_energy = []

ictal_training_dwt_approx_energy = []
nonictal_training_dwt_approx_energy = []
test_dwt_approx_energy = []


ictal_training_dwt_three_line = []
nonictal_training_dwt_three_line = []
test_dwt_three_line = []

ictal_training_dwt_four_line = []
nonictal_training_dwt_four_line = []
test_dwt_four_line= []

ictal_training_dwt_approx_line = []
nonictal_training_dwt_approx_line = []
test_dwt_approx_line = []


ictal_training_dwt_three_entropy= []
nonictal_training_dwt_three_entropy = []
test_dwt_three_entropy = []

ictal_training_dwt_four_entropy = []
nonictal_training_dwt_four_entropy = []
test_dwt_four_entropy = []

ictal_training_dwt_approx_entropy = []
nonictal_training_dwt_approx_entropy = []
test_dwt_approx_entropy = []
    


for x in range(7):
    
    nonictal_files = os.listdir((str(file_path1) + str((x+1)) + "/non-ictal train/"))
    nonictal_files = nonictal_files[0:-2]
    nonictal_files.sort(key=getint) 
    
    
    ictal_files = os.listdir((str(file_path1) + str((x+1)) + "/ictal train/"))
    ictal_files = ictal_files[0:-2]
    ictal_files.sort(key=getint)
    
    test_files = os.listdir((str(file_path1) + str((x+1)) + "/test/"))
    test_files = test_files[0:-2]
    test_files.sort(key=getint3)
    
    test_names = test_names + test_files 



    print("Loading files from patient " + str(x+1))
    
    count = 0
    
    for i in range(len(nonictal_files)):
        
        
        
        file_name = str(nonictal_files[i])
        first = file_name.split("_")
        

        if first[0] != 'patient':
            
            count = count+1
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/non-ictal train/" + str(nonictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])



        [sum_one, sum_two, sum_three, sum_four, approx] = decomposition(temp)


        [variance_three, energy_three, line_length_three, entropy_three] = featureExtraction(sum_three)

        [variance_four, energy_four, line_length_four, entropy_four] = featureExtraction(sum_four)

        [variance_approx, energy_approx, line_length_approx, entropy_approx] = featureExtraction(approx)
        
        


        nonictal_training_dwt_three_variance = np.append(nonictal_training_dwt_three_variance, variance_three)

        nonictal_training_dwt_four_variance = np.append(nonictal_training_dwt_four_variance, variance_four)

        nonictal_training_dwt_approx_variance = np.append(nonictal_training_dwt_approx_variance, variance_approx)
        


        nonictal_training_dwt_three_energy = np.append(nonictal_training_dwt_three_energy, energy_three)

        nonictal_training_dwt_four_energy = np.append(nonictal_training_dwt_four_energy, energy_four)

        nonictal_training_dwt_approx_energy = np.append(nonictal_training_dwt_approx_energy, energy_approx)




        nonictal_training_dwt_three_line = np.append(nonictal_training_dwt_three_line, line_length_three)

        nonictal_training_dwt_four_line = np.append(nonictal_training_dwt_four_line, line_length_four)

        nonictal_training_dwt_approx_line = np.append(nonictal_training_dwt_approx_line, line_length_approx)


        nonictal_training_dwt_three_entropy = np.append(nonictal_training_dwt_three_entropy, entropy_three)

        nonictal_training_dwt_four_entropy = np.append(nonictal_training_dwt_four_entropy, entropy_four)

        nonictal_training_dwt_approx_entropy = np.append(nonictal_training_dwt_approx_entropy, entropy_approx)

    print("Nonictal Data - Done")
    
    count = 0
    
    for i in range(len(ictal_files) ):
        
        
        file_name = str(ictal_files[i])
        first = file_name.split("_")
        
        
        
        if first[0] != "patient":
            count = count + 1
            continue 
    
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/ictal train/" + str(ictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        

        
        [sum_one, sum_two, sum_three, sum_four, approx] = decomposition(temp)

        [variance_three, energy_three, line_length_three, entropy_three] = featureExtraction(sum_three)

        [variance_four, energy_four, line_length_four, entropy_four] = featureExtraction(sum_four)

        [variance_approx, energy_approx, line_length_approx, entropy_approx] = featureExtraction(approx)
        
  

        ictal_training_dwt_three_variance = np.append(ictal_training_dwt_three_variance, variance_three)

        ictal_training_dwt_four_variance = np.append(ictal_training_dwt_four_variance, variance_four)

        ictal_training_dwt_approx_variance = np.append(ictal_training_dwt_approx_variance, variance_approx)

        
   

        ictal_training_dwt_three_energy = np.append(ictal_training_dwt_three_energy, energy_three)

        ictal_training_dwt_four_energy = np.append(ictal_training_dwt_four_energy, energy_four)

        ictal_training_dwt_approx_energy = np.append(ictal_training_dwt_approx_energy, energy_approx)




        ictal_training_dwt_three_line = np.append(ictal_training_dwt_three_line, line_length_three)

        ictal_training_dwt_four_line = np.append(ictal_training_dwt_four_line, line_length_four)

        ictal_training_dwt_approx_line = np.append(ictal_training_dwt_approx_line, line_length_approx)



        ictal_training_dwt_three_entropy = np.append(ictal_training_dwt_three_entropy, entropy_three)

        ictal_training_dwt_four_entropy = np.append(ictal_training_dwt_four_entropy, entropy_four)

        ictal_training_dwt_approx_entropy = np.append(ictal_training_dwt_approx_entropy, entropy_approx)
        
    print("Ictal Data - Done")


    count = 0
    
    for i in range(len(test_files)):
        
        file_name = str(test_files[i])
        first = file_name.split("_")
        
        if first[0] != "patient":
            
            count = count + 1
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/test/" + str(test_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])

        
        [sum_one, sum_two, sum_three, sum_four, approx] = decomposition(temp)



        [variance_three, energy_three, line_length_three, entropy_three] = featureExtraction(sum_three)

        [variance_four, energy_four, line_length_four, entropy_four] = featureExtraction(sum_four)

        [variance_approx, energy_approx, line_length_approx, entropy_approx] = featureExtraction(approx)



        test_dwt_three_variance = np.append(test_dwt_three_variance, variance_three)


        test_dwt_four_variance = np.append(test_dwt_four_variance, variance_four)

        test_dwt_approx_variance = np.append(test_dwt_approx_variance, variance_approx)


        test_dwt_three_energy = np.append(test_dwt_three_energy, energy_three)

        test_dwt_four_energy = np.append(test_dwt_four_energy, energy_four)

        test_dwt_approx_energy = np.append(test_dwt_approx_energy, energy_approx)



        test_dwt_three_line = np.append(test_dwt_three_line, line_length_three)

        test_dwt_four_line = np.append(test_dwt_four_line, line_length_four)

        test_dwt_approx_line = np.append(test_dwt_approx_line, line_length_approx)



        test_dwt_three_entropy = np.append(test_dwt_three_entropy, entropy_three)

        test_dwt_four_entropy = np.append(test_dwt_four_entropy, entropy_four)

        test_dwt_approx_entropy = np.append(test_dwt_approx_entropy, entropy_approx)
    
    print("Test Data - Done")




print(test_dwt_three_entropy)


energy_training_three = np.append(np.repeat(ictal_training_dwt_three_energy, 5), nonictal_training_dwt_three_energy)

variance_training_three = (np.append(np.repeat(ictal_training_dwt_three_variance ,5), nonictal_training_dwt_three_variance))

line_training_three = (np.append(np.repeat(ictal_training_dwt_three_line, 5), nonictal_training_dwt_three_line))

entropy_training_three = (np.append(np.repeat(ictal_training_dwt_three_entropy, 5), nonictal_training_dwt_three_entropy))


# energy_training_three = np.append(ictal_training_dwt_three_energy, nonictal_training_dwt_three_energy)

# variance_training_three = (np.append(ictal_training_dwt_three_variance, nonictal_training_dwt_three_variance))

# line_training_three = (np.append(ictal_training_dwt_three_line, nonictal_training_dwt_three_line))

# entropy_training_three = (np.append(ictal_training_dwt_three_entropy, nonictal_training_dwt_three_entropy))

print(line_training_three.shape)
print(variance_training_three.shape)
print(energy_training_three.shape)
print(entropy_training_three.shape)



energy_training_four = np.append(np.repeat(ictal_training_dwt_four_energy, 5), nonictal_training_dwt_four_energy)

variance_training_four = (np.append(np.repeat(ictal_training_dwt_four_variance, 5), nonictal_training_dwt_four_variance))

line_training_four = (np.append(np.repeat(ictal_training_dwt_four_line, 5), nonictal_training_dwt_four_line))

entropy_training_four = (np.append(np.repeat(ictal_training_dwt_four_entropy,5), nonictal_training_dwt_four_entropy))



energy_training_approx = np.append(np.repeat(ictal_training_dwt_approx_energy,5), nonictal_training_dwt_approx_energy)

variance_training_approx = (np.append(np.repeat(ictal_training_dwt_approx_variance, 5), nonictal_training_dwt_approx_variance))

line_training_approx = (np.append(np.repeat(ictal_training_dwt_approx_line,5), nonictal_training_dwt_approx_line))

entropy_training_approx = (np.append(np.repeat(ictal_training_dwt_approx_entropy,5), nonictal_training_dwt_approx_entropy))

# energy_training_four = np.append(ictal_training_dwt_four_energy, nonictal_training_dwt_four_energy)

# variance_training_four = (np.append(ictal_training_dwt_four_variance, nonictal_training_dwt_four_variance))

# line_training_four = (np.append(ictal_training_dwt_four_line, nonictal_training_dwt_four_line))

# entropy_training_four = (np.append(ictal_training_dwt_four_entropy, nonictal_training_dwt_four_entropy))



# energy_training_approx = np.append(ictal_training_dwt_approx_energy, nonictal_training_dwt_approx_energy)

# variance_training_approx = (np.append(ictal_training_dwt_approx_variance, nonictal_training_dwt_approx_variance))

# line_training_approx = (np.append(ictal_training_dwt_approx_line, nonictal_training_dwt_approx_line))

# entropy_training_approx = (np.append(ictal_training_dwt_approx_entropy, nonictal_training_dwt_approx_entropy))



ictal_labels = np.ones(len(np.repeat(ictal_training_dwt_approx_energy, 5)))
print(ictal_labels.shape)
nonictal_labels = np.zeros(len(nonictal_training_dwt_approx_energy))
print(nonictal_labels.shape)

y = np.append(ictal_labels, nonictal_labels)


print(y.shape)

X_training = np.column_stack((variance_training_three, variance_training_four, variance_training_approx, energy_training_three, energy_training_four,energy_training_approx,  line_training_three, line_training_four,line_training_approx,   entropy_training_three, entropy_training_four, entropy_training_approx ))

X_training = np.asarray(X_training)

where_are_nans = np.isnan(X_training)
X_training[where_are_nans] = 0
where_are_inf = np.isinf(X_training)
X_training[where_are_inf] = 100

parameters = [0.000001, 0.0001, 0.1, 1, 100, 10000]

#clf = svm.LinearSVC(C = 0.001)

# clf = tree.DecisionTreeClassifier()

clf = RandomForestClassifier(n_estimators = 1000)






print("Fitting the SVM")


for i in range(5):
    X_train, X_valid, y_train, y_valid = train_test_split(X_training, y, test_size=0.4)

    clf.fit(X_train, y_train)  

    print("Model Complete. Predicting from Valid Data")

    #print(clf.predict(X_valid))
    print(clf.score(X_valid, y_valid))

 

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

test = np.column_stack((test_dwt_three_variance, test_dwt_four_variance, test_dwt_approx_variance,  test_dwt_three_energy, test_dwt_four_energy, test_dwt_approx_energy,  test_dwt_three_line, test_dwt_four_line, test_dwt_approx_line, test_dwt_three_entropy, test_dwt_four_entropy, test_dwt_approx_entropy))

where_are_nans = np.isnan(test)
test[where_are_nans] = 0
where_are_inf = np.isinf(test)
test[where_are_inf] = 100

test = np.asarray(test)

print("Predicting from Test Data")



test_results = clf.predict(test)
print(np.mean(test_results))

data_to_submit = pd.DataFrame({
    #'id':test_names,
    'prediction':test_results
})

data_to_submit.to_csv('csv_to_submit_unbalanced.csv', index = False)


w_norm = np.linalg.norm(clf.dual_coef_)
dist = clf.decision_function(X_train) / w_norm

closest_50 = []

count = 0

for i in range(len(dist)):

    current = X_train[i,:]

    if count > len(ictal_training_dwt_four_energy)-1:

        continue 

    if y[i] == 0:

        if count == 0:

            closest_50 = current

        else:

            closest_50 = np.append(closest_50, current)

        count = count + 1


positive_examples = np.transpose(np.vstack((ictal_training_dwt_three_variance, ictal_training_dwt_four_variance, ictal_training_dwt_approx_variance, ictal_training_dwt_three_energy, ictal_training_dwt_four_energy,ictal_training_dwt_approx_energy, ictal_training_dwt_three_line, ictal_training_dwt_four_line, ictal_training_dwt_approx_line,   ictal_training_dwt_three_entropy, ictal_training_dwt_four_entropy, ictal_training_dwt_approx_entropy )))

closest_50 = np.transpose(np.reshape(closest_50, (12, len(ictal_training_dwt_four_energy))))

print(positive_examples.shape)
print(closest_50.shape)

balanced_train = np.append(positive_examples, closest_50, axis=0)

where_are_nans = np.isnan(balanced_train)
balanced_train[where_are_nans] = 0
where_are_inf = np.isinf(balanced_train)
balanced_train[where_are_inf] = 100

y_bal = np.append(np.ones(len(positive_examples)), np.zeros(len(closest_50)))

print('Fitting balanced data')

clf.fit(balanced_train, y_bal)

print("Score on Validation Data")

print(clf.score(X_valid, y_valid))

print("Average Prediction on Test Data")

test_results = clf.predict(test)

print(np.mean(test_results))





print("Compiling Test Results")

print(len(test_names))

kept_names = []

# for k in range(len(test_names)-2):
#     name = test_names[k+2]
#     print(name)

#     first = name.split("_")


        
#     if first[0] != "patient":
#         print(first[0])
#         temp = test_names.pop()[k]
#         k = k - 1

#     try:
#         nomat = first[3].split(".")
#     except IndexError:
#         continue
    
#     test_names[k] = str(first[0]) + "_" + str(first[1]) + "_" + str(nomat[0])
        


data_to_submit = pd.DataFrame({
    #'id':test_names,
    'prediction':test_results
})

data_to_submit.to_csv('csv_to_submit.csv', index = False)
    
    
    