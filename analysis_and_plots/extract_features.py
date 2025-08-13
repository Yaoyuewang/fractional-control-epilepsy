import os
import scipy.io
import numpy as np

patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
# patients = ['HUP64']
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/data_v2'
window_size = 3

data_length = 20 - window_size + 1
ictal_start = data_length
def load_fos_params(patient, seizure, condition, window_size):
    path = f'{main_pathname}/{patient}/{condition}-block-{seizure}_parameters_{window_size}sec_1iter.mat'
    return scipy.io.loadmat(path)

for patient in patients: 
    for seizure in range (1, 9):
        file_path = os.path.join(main_pathname, patient, f"ictal-block-{seizure}_parameters_{window_size}sec_1iter.mat")
        if os.path.exists(file_path):
            ictal_data = load_fos_params(patient, seizure, 'ictal', window_size)
            interictal_data = load_fos_params(patient, seizure, 'interictal', window_size)
            num_windows = ictal_data['alpha'].shape[1]
            ictal_end = min(num_windows-20, ictal_start + data_length)
            
            ictal_alpha_all = ictal_data['alpha']
            ictal_eigen_all = ictal_data['eigenvalues']

            interictal_alpha = interictal_data['alpha'][:, 0:data_length]
            interictal_eigen = interictal_data['eigenvalues'][:, 0:data_length]

            preictal_alpha = ictal_alpha_all[:, 0:ictal_start]
            ictal_alpha = ictal_alpha_all[:, ictal_start:ictal_end]
            postictal_alpha = ictal_alpha_all[:, num_windows - 20:num_windows -2]

            preictal_eigen = ictal_eigen_all[:, 0:ictal_start]
            ictal_eigen = ictal_eigen_all[:, ictal_start:ictal_end]
            postictal_eigen = ictal_eigen_all[:, num_windows-20:num_windows - 2]

            save_path = os.path.join(main_pathname, patient, f"fos_features_block_{seizure}.npz")
            np.savez(save_path,
                preictal_alpha = preictal_alpha,
                ictal_alpha = ictal_alpha,
                postictal_alpha = postictal_alpha,
                interictal_alpha = interictal_alpha,
                preictal_eigen = preictal_eigen,
                ictal_eigen = ictal_eigen,
                postictal_eigen = postictal_eigen,
                interictal_eigen = interictal_eigen,
                timepoints = list(range(num_windows-20,num_windows - 2)))
            
            

            



                