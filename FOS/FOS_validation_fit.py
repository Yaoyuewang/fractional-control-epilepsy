import numpy as np 
import os 
import scipy.io
from sklearn.metrics import r2_score 
from functions import fracOrdUU, reconstruct_FOS
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/'

# Checking abnormal eigen/alpha values for a few patients, making sure FOS fits corrects 
patient = 'HUP78'
seizure = 1
phase = 'ictal'
parameters = scipy.io.loadmat(os.path.join(main_pathname, 'data_v2', patient, f'{phase}-block-{seizure}_parameters_3sec_1iter.mat'))
raw_signal = scipy.io.loadmat(os.path.join(main_pathname, 'data', patient, f'{patient}-{phase}-block-{seizure}.mat'))
alpha = parameters['alpha']
A_matrix = parameters['A']
xPred = parameters['xPred']
print(alpha.shape)
print(A_matrix.shape)

