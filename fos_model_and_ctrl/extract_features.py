import os
import scipy.io
import numpy as np

patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
# patients = ['HUP70']
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/data_v2'
raw_data_path = 'c:/Users/yaoyu/Documents/Epilepsy_research/data'
window_size = 3

data_length = 20 - window_size + 1
ictal_start = 20
def load_fos_params(patient, seizure, condition, window_size):
    path = f'{main_pathname}/{patient}/{condition}_block_{seizure}_parameters_{window_size}sec_1iter.mat'
    return scipy.io.loadmat(path)

for patient in patients: 
    for seizure in range (1, 9):
        file_path = os.path.join(main_pathname, patient, f"ictal_block_{seizure}_parameters_{window_size}sec_1iter.mat")
        if os.path.exists(file_path):
            ictal_parameters = load_fos_params(patient, seizure, 'ictal', window_size)
            interictal_parameters = load_fos_params(patient, seizure, 'interictal', window_size)
            ictal_iEEG_path= os.path.join(raw_data_path, patient, f"{patient}-ictal-block-{seizure}.mat")
            interictal_iEEG_path = os.path.join(raw_data_path, patient, f"{patient}-interictal-block-{seizure}.mat")
            ictal_fos_fit_path = os.path.join(main_pathname, patient, f"ictal_block_{seizure}_r2_all.npz")
            interictal_fos_fit_path = os.path.join(main_pathname, patient, f"interictal_block_{seizure}_r2_all.npz")

            num_windows = ictal_parameters['alpha'].shape[1]
            sampling_rate = int(np.ceil(scipy.io.loadmat(ictal_iEEG_path)['Fs'][0][0]))
            ictal_end = min(num_windows-21, ictal_start + data_length)
            
            ictal_iEEG_all = scipy.io.loadmat(ictal_iEEG_path)['evData']
            interictal_iEEG_all = scipy.io.loadmat(interictal_iEEG_path)['evData']
            ictal_alpha_all = ictal_parameters['alpha']
            interictal_alpha_all = interictal_parameters['alpha']
            ictal_eigen_all = ictal_parameters['eigenvalues']
            interictal_eigen_all = interictal_parameters['eigenvalues']
            ictal_fos_fit = np.load(ictal_fos_fit_path)['r2_fit']
            interictal_fos_fit = np.load(interictal_fos_fit_path)['r2_fit']
            # Alpha
            interictal_alpha = interictal_alpha_all[:, 0:data_length]
            preictal_alpha = ictal_alpha_all[:, 0:data_length]
            ictal_alpha = ictal_alpha_all[:, ictal_start:ictal_end]
            postictal_alpha = ictal_alpha_all[:, num_windows - data_length:num_windows]
            # Eigenvalues
            interictal_eigen = interictal_eigen_all[:, 0:data_length]
            preictal_eigen = ictal_eigen_all[:, 0:data_length]
            ictal_eigen = ictal_eigen_all[:, ictal_start:ictal_end]
            postictal_eigen = ictal_eigen_all[:, num_windows-data_length:num_windows]
            # iEEG Data
            interictal_iEEG = interictal_iEEG_all[:, 0:data_length*sampling_rate]
            preictal_iEEG = ictal_iEEG_all[:, 0:data_length*sampling_rate]
            ictal_iEEG = ictal_iEEG_all[:, ictal_start*sampling_rate:ictal_end*sampling_rate]
            postictal_iEEG = ictal_iEEG_all[:, (num_windows-data_length)*sampling_rate:num_windows*sampling_rate]
            # FOS fit 
            interictal_fit = interictal_fos_fit[:, 0:data_length]
            preictal_fit = ictal_fos_fit[:, 0:data_length]
            ictal_fit = ictal_fos_fit[:, ictal_start:ictal_end]
            postictal_fit = ictal_fos_fit[:, num_windows-data_length:num_windows]

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
                preictal_iEEG = preictal_iEEG,
                ictal_iEEG = ictal_iEEG,
                postictal_iEEG = postictal_iEEG,
                interictal_iEEG = interictal_iEEG,
                preictal_fit = preictal_fit,
                ictal_fit = ictal_fit,
                postictal_fit = postictal_fit,
                interictal_fit = interictal_fit,
                timepoints = list(range(num_windows-data_length,num_windows)))
            

            



                