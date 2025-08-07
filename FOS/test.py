from functions import reconstruct_FOS
import numpy as np
import os 
import scipy.io 
from multiprocessing import Pool

patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/'
window_length = 3 
stride = 1

def process_block(patient, seizure, condition):
    filepath = os.path.join(main_pathname, 'data', patient, f"{patient}-{condition}-block-{seizure}.mat")
    
    FOS_params = scipy.io.loadmat(os.path.join(main_pathname, 'data_v2', patient, f"{condition}-block-{seizure}_parameters_3sec_1iter.mat"))
    raw_data = scipy.io.loadmat(filepath)

    sampling_rate = int(np.ceil(raw_data['Fs'][0][0]))
    A = FOS_params["A"]
    alpha = FOS_params["alpha"]
    num_chns = FOS_params["A"].shape[0]
    evData = raw_data["evData"]

    window_samples = sampling_rate * window_length
    stride_samples = stride * sampling_rate
    num_windows = alpha.shape[1]

    xPred = np.zeros((num_chns,evData.shape[1]))
    counts = np.zeros(evData.shape[1])

    for w in range(num_windows):
        start_idx = w * stride_samples
        X_window = evData[:, start_idx:start_idx + window_samples]
        X_window = X_window.T - np.mean(X_window, axis = 1)
        X_window = X_window.T
        x_window_pred = reconstruct_FOS(
            alpha[:, w],
            A[:, :, w],
            X_window,
            num_chns,
            sampling_rate,
            window_length
        )
        xPred[:, start_idx:start_idx + window_samples] += x_window_pred
        counts[start_idx:start_idx + window_samples] += 1
    counts[counts == 0] = 1
    xPred /= counts[np.newaxis, :]
    FOS_params['xPred'] = xPred
    param_path = os.path.join(main_pathname, 'data_v2', patient, f'{condition}-block-{seizure}_parameters_{window_length}sec_1iter.mat')
    scipy.io.savemat(param_path, FOS_params, do_compression=True)
    print(filepath)

if __name__ == "__main__":
    all_jobs = []
    for patient in patients:
        for seizure in range(9):
            for condition in ["ictal", "interictal"]:
                path = os.path.join(main_pathname, 'Data', patient, f'{patient}-{condition}-block-{seizure}.mat')
                if os.path.exists(path):
                    all_jobs.append((patient, seizure, condition))
    with Pool(processes=14) as pool:
        results = pool.starmap(process_block, all_jobs)