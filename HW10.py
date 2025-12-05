import os
import scipy.io as sio
from self_py_fun.HW10Fun import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# In HW7, you have the chance to visualize a truncated EEG dataset stratified by
# target and non-target stimulus type.
#
# The fundamental problem of P300 ERP-BCI speller system is to perform a binary classification.
#
# In HW10, you are asked to implement the binary classification using various methods,
# and evaluate the model performance with a testing dataset.
#
# You will use K114_001_BCI_TRN_Truncated_Data_0.5_6.mat as a training set, and
# K114_001_BCI_FRT_Truncated_Data_0.5_6.mat as a testing set.
#
# Notice that here, we do not split training/testing within K114_001_BCI_TRN_Truncated_Data_0.5_6.mat
# because each row is not entirely independent of each other due to the special structure of the dataset.

# Global constants:
np.random.seed(100)
bp_low = 0.5
bp_upp = 6
electrode_num = 16
# Change the following directory to your own one.
parent_dir = '/Users/Tej/Documents/GitHub/BIOS-584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25)
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
subject_name = 'K114'
# create a new folder called K114
subject_dir = '{}/{}'.format(parent_dir, subject_name)
if not os.path.exists(subject_dir):
    os.mkdir(subject_dir)

char_trn = 'THE0QUICK0BROWN0FOX'
char_trn_size = len(char_trn)

# Step 1: Import dataset
# Step 1.1: TRN dataset
trn_data_name = '{}_001_BCI_TRN_Truncated_Data_{}_{}'.format(subject_name, bp_low, bp_upp)
trn_data_dir = '{}/{}.mat'.format(parent_data_dir, trn_data_name)
eeg_trn_obj = sio.loadmat(trn_data_dir)

print(eeg_trn_obj.keys())
eeg_trn_signal = eeg_trn_obj['Signal']
print(eeg_trn_signal.shape)
eeg_trn_type = eeg_trn_obj['Type']
eeg_trn_type = np.squeeze(eeg_trn_type)
print(eeg_trn_type.shape)

# Step 1.2: FRT dataset
# The following code should be completed by students themselves.
# you should be able to obtain relevant data files named
# eeg_frt_signal and eeg_frt_type
# Write your own code below:

frt_data_name = '{}_001_BCI_FRT_Truncated_Data_{}_{}'.format(subject_name, bp_low, bp_upp)
frt_data_dir = '{}/{}.mat'.format(parent_data_dir, frt_data_name)

eeg_frt_obj = sio.loadmat(frt_data_dir)

print(eeg_frt_obj.keys())
eeg_frt_signal = eeg_frt_obj['Signal']
print(eeg_frt_signal.shape)

eeg_frt_type = eeg_frt_obj['Type']
eeg_frt_type = np.squeeze(eeg_frt_type)
print(eeg_frt_type.shape)

# You have completed the exploratory data analysis in HW7 and HW8.
# The dataset has been carefully reviewed by Dr. Jane E. Huggins,
# so we do not need to worry about missing, outliers, errors of the dataset.

# Step 2: Fit classification models
# You will try the following methods:
# Logistic Regression,
# Linear Discriminant Analysis,
# Support Vector Machine (sometimes called support vector classification)
# You do not need to modify the parameters of each classifier
# except for LogisticRegression: set max_iter=1000
# Write your own code below:

X_trn = eeg_trn_signal
y_trn = eeg_trn_type

#Logistic Regression
log_reg_clf = LR(max_iter=5000)
log_reg_clf.fit(X_trn, y_trn)
#Linear Discriminant Analysis
lda_clf = LDA()
lda_clf.fit(X_trn, y_trn)

svm_clf = SVC(probability=True)
svm_clf.fit(X_trn, y_trn)


# Step 3: Evaluate model performance on both TRN and FRT files
# Step 3.1: Prediction accuracy on TRN files
# We will compute the probability of each stimulus with .predict_proba()
# before we convert the stimulus-level probability to character-level probability.
# You are asked to generate stimulus-level probability for each method on TRN files,
# denoted as logistic_y_trn, lda_y_trn, and svm_y_trn.
# Write your own code below:

X_trn = eeg_trn_signal

logistic_y_trn = log_reg_clf.predict_proba(X_trn)[:, 1:2]
lda_y_trn      = lda_clf.predict_proba(X_trn)[:, 1:2]
svm_y_trn      = svm_clf.predict_proba(X_trn)[:, 1:2]

print(logistic_y_trn.shape, lda_y_trn.shape, svm_y_trn.shape)

# Step 3.2: Prediction accuracy on FRT files
# Similarly, you are asked to generate stimulus-level probability for each method on FRT files,
# denoted as logistic_y_frt, lda_y_frt, and svm_y_frt.
# Write your own code below:

X_frt = eeg_frt_signal

logistic_y_frt = log_reg_clf.predict_proba(X_frt)[:, 1:2]
lda_y_frt      = lda_clf.predict_proba(X_frt)[:, 1:2]
svm_y_frt      = svm_clf.predict_proba(X_frt)[:, 1:2]

print(logistic_y_frt.shape, lda_y_frt.shape, svm_y_frt.shape)


# Step 4: Convert binary classification probability to character-level accuracy
# This involves advanced data manipulation, so you do not need to write any new code.
# Please run the following code to view the final results.
# Make Type arrays 2D again for streamline_predict
eeg_trn_code = eeg_trn_obj['Code']
eeg_frt_code = eeg_frt_obj['Code']
char_frt = convert_raw_char_to_alphanumeric_stype(eeg_frt_obj['Text'])
# raw format is different from the current 6x6 layout characters.
char_frt_size = len(char_frt)
frt_seq_size = int(eeg_frt_signal.shape[0]/char_frt_size/12)

# Logistic regression
print('Logistic Regression on TRN:')
logistic_letter_mat_trn, logistic_letter_prob_mat_trn = streamline_predict(
    logistic_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(logistic_letter_mat_trn)
print(list(char_trn)) # This is the true spelling characters for training set!
logistic_trn_accuracy = np.mean(logistic_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Logistic Regression on FRT:')
logistic_letter_mat_frt, logistic_letter_prob_mat_frt = streamline_predict(
    logistic_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(logistic_letter_mat_frt)
print(list(char_frt)) # This is the true spelling characters for testing set!
logistic_frt_accuracy = np.mean(logistic_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

# LDA:
print('LDA on TRN:')
lda_letter_mat_trn, lda_letter_prob_mat_trn = streamline_predict(
    lda_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(lda_letter_mat_trn)
print(list(char_trn)) # This is the true spelling characters for training set!
lda_trn_accuracy = np.mean(lda_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('LDA on FRT:')
lda_letter_mat_frt, lda_letter_prob_mat_frt = streamline_predict(
    lda_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(lda_letter_mat_frt)
print(list(char_frt)) # This is the true spelling characters for testing set!
lda_frt_accuracy = np.mean(lda_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

# SVM:
print('Support Vector Machine on TRN:')
svm_letter_mat_trn, svm_letter_prob_mat_trn = streamline_predict(
    svm_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(svm_letter_mat_trn)
print(list(char_trn)) # This is the true spelling characters for training set!
svm_trn_accuracy = np.mean(svm_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Support Vector Machine on FRT:')
svm_letter_mat_frt, svm_letter_prob_mat_frt = streamline_predict(
    svm_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(svm_letter_mat_frt)
print(list(char_frt)) # This is the true spelling characters for training set!
svm_frt_accuracy = np.mean(svm_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)


print(logistic_trn_accuracy)
print(lda_trn_accuracy)
print(svm_trn_accuracy)

print(logistic_frt_accuracy)
print(lda_frt_accuracy)
print(svm_frt_accuracy)


# Remember to answer two questions below:

# What do rows 122, 131, 141, 150, 160, and 169 do? Briefly answer the question below:
# In case that your row IDs are messed up when you start to fill in the blank,
# I attach the lines of code for your reference.
# logistic_trn_accuracy = np.mean(logistic_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# logistic_frt_accuracy = np.mean(logistic_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
# lda_trn_accuracy = np.mean(lda_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# lda_frt_accuracy = np.mean(lda_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
# svm_trn_accuracy = np.mean(svm_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# svm_frt_accuracy = np.mean(svm_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

''' 
The aforementioned lines of code checks how the model is working by comparing whether the prediction is accurate to
the real value. In our case it is the letter values. We get the percentage of correct predictions and it repeates
for 1, 2, 3, and 4 repetitions, so we can see how accuracy improves as we use more EEG data
'''


# Step 5: Summary
# Which method performs the best? Why?
'''
The Support Vector Machine (SVM) method performed better compared to LDN and Logisitic Regression. Out of the 4 repetitions,
although all three methods were getting increasing percent of letter correct, SVM was the only method to 
reach a percentage of 100% at the 4th repetition. Thus, it reaches the best accuracy the fastest and is better than the
other models. SVM is better at identifying the difference between target and non-target when signals are complicated. 
'''