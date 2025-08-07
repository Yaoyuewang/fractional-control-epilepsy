import scipy
import numpy as np
import pandas as pd
from scipy.special import gamma
import scipy.io
import os
import math
import mat73
import csv
from tqdm import tqdm 
from functions import fracOrdUU, reconstruct_FOS
from multiprocessing import Pool

# patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
patients = ['MAYO010','MAYO011','MAYO016','MAYO020']

# patients = ['HUP70']
num_seizures = 35 
window_length = 3
stride = 1 
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/'

# compute the FOS parameters for both ictal data and interictal snapshots for every patient and every ictal snapshot
# you should only have to run this one time

def process_block(sampling_rate, evData, num_chns, window_length):
    window_samples = window_length* sampling_rate
    stride_samples = stride * sampling_rate
    num_windows = int((evData.shape[1]- window_samples) / stride_samples) + 1

    xPred = np.zeros((num_chns,evData.shape[1]))
    eigenvalues = np.zeros((num_chns, num_windows), dtype=np.complex128)
    eigenvectors = np.zeros((num_chns, num_chns, num_windows), dtype=np.complex128)
    alpha = np.zeros((num_chns, num_windows))
    A = np.zeros((num_chns, num_chns, num_windows))
    A_0 = np.zeros((num_chns, num_chns, num_windows))
    all_mse = []

    # print(f"Running block in PID {os.getpid()}")
    for window, start_idx in enumerate(range(0, evData.shape[1] - window_samples + 1, stride_samples)):
        X = evData[:, start_idx:start_idx + window_samples]
        X = X.T - np.mean(X, axis = 1)
        X = X.T
        fModel = fracOrdUU(verbose=-1)
        mseIter = fModel.fit(X)
        all_mse.append(mseIter)
        fModel._order[np.abs(fModel._order) < 0.01] = 0
        alpha[:, window] = fModel._order
        A[:, :, window] = np.squeeze(fModel._AMat[-1])

        xPred[:, start_idx:start_idx + sampling_rate * window_length] = reconstruct_FOS(
            alpha[:, window], A[:, :, window], X, num_chns, sampling_rate, window_length
        )

        v = np.where(fModel._order == 0, 1, gamma(1 - fModel._order) / gamma(-fModel._order))
        D = np.diag(v)
        A_0[:, :, window] = A[:, :, window] - D
        eigenvalues[:, window], eigenvectors[:, :, window] = np.linalg.eig(A_0[:, :, window])
    return alpha, A, A_0, eigenvalues, eigenvectors, xPred, all_mse

def process_data(seizure, patient, path, window_length, main_pathname, condition):
    
    results = os.path.join(main_pathname, 'data_v2', patient)
    try:
        os.makedirs(results)
    except FileExistsError: 
        pass 
    struc = scipy.io.loadmat(path)
    sampling_rate = int(np.ceil(struc['Fs'][0][0]))
    evData = struc['evData']
    num_chns = evData.shape[0]

    alpha, A, A_0, eigenvalues, eigenvectors, xPred, mse = process_block(
        sampling_rate, evData, num_chns, window_length
    )

    fos_data = {
        "A": A, "A_0": A_0, "alpha": alpha, 
        "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, 
        "xPred": xPred
    }
    save_path = os.path.join(main_pathname, 'data_v2', patient, f'{condition}-block-{seizure}_parameters_{window_length}sec_1iter.mat')
    if os.path.exists(save_path):
        os.remove(save_path)
    scipy.io.savemat(save_path, fos_data, do_compression=True)
    avg_mse = np.mean(np.array(mse), axis = 0)
    mse_rows = []
    for iter_idx, mse in enumerate(avg_mse):
        mse_rows.append([patient, seizure, "ictal", iter_idx, mse])
    
    return mse_rows
    
if __name__ == "__main__":

    all_mse_rows = []
    all_jobs = []
    for patient in patients:
        for seizure in range(9):
            for condition in ["ictal", "interictal"]:
                path = os.path.join(main_pathname, 'Data', patient, f'{patient}-{condition}-block-{seizure}.mat')
                if os.path.exists(path):
                    all_jobs.append((seizure, patient, path, window_length, main_pathname, condition))
    with Pool(processes=12) as pool:
        results = pool.starmap(process_data, all_jobs)

    # for mse_rows in ictal_results + interictal_results:
    #     all_mse_rows.extend(mse_rows)
    # csv_path = os.path.join(main_pathname, "results", "all_mse_summary_5iter_2sec_test.csv")
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Patient", "Seizure", "Phase", "Iteration", "Avg_MSE"])
    #     writer.writerows(all_mse_rows)