import scipy
import numpy as np
import pandas as pd
from scipy.special import gamma
import matplotlib.pyplot as plt
import scipy.io
import os
import math
import mat73
from tqdm import tqdm 
from FOS.functions import HaarWaveletTransform, fracOrdUU, adjust_fontsize, reconstruct_FOS


# patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
num_seizures = 35 
window_length = 1
patients = ['HUP64']
# pathname
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/'

labels = ['HUP64-ictal-1','HUP68-ictal-1','HUP68-ictal-2','HUP68-ictal-3','HUP68-ictal-4','HUP68-ictal-5','HUP70-ictal-1','HUP70-ictal-2','HUP70-ictal-3','HUP70-ictal-4','HUP70-ictal-5','HUP70-ictal-6','HUP70-ictal-7','HUP70-ictal-8', 'HUP72-ictal-1', 'HUP78-ictal-1', 'HUP78-ictal-2', 'HUP78-ictal-3', 'HUP78-ictal-4', 'HUP78-ictal-5', 'HUP86-ictal-1', 'HUP86-ictal-2', 'MAYO010-ictal-1', 'MAYO010-ictal-3','MAYO011-ictal-1', 'MAYO011-ictal-2', 'MAYO016-ictal-1', 'MAYO016-ictal-2', 'MAYO016-ictal-5', 'MAYO016-ictal-6', 'MAYO020-ictal-1', 'MAYO020-ictal-2', 'MAYO020-ictal-3', 'MAYO020-ictal-4', 'MAYO020-ictal-6'] 
EEC = [632657, 819104, 836557, 911307, 933636, 968205, 62929, 65364, 66661, 252106, 277320, 277367, 325865, 328368, 0, 267376, 292659, 305960, 314000, 342228, 763898, 798468, 0, 0, 0, 0, 41639.01446, 298062.0994, 452586.826922, 466913.56625, 0, 0, 0, 0, 0]
UEO = [632666.73, 819104, 836557, 911307, 933639.01, 968211.34, 62930.48, 65365.04, 66663.17, 252114.37,277323.1, 277367.55, 325868.68, 328370.11, 0, 267375.83, 292666.07, 305959.2, 314031.75, 342228.12, 763899.75, 798469.62, 0, 0, 0, 0, 41645.10177, 298066.10, 452592.826922, 466922.56625, 0, 0, 0, 0, 0]
diff = np.array(UEO)-np.array(EEC)
d = {'patient': labels, 'EEC': EEC, 'UEO': UEO, 'Difference': diff}
data = pd.DataFrame(data=d)

# extracting the SOZ channels
data_dict = mat73.loadmat(main_pathname+'data/clinical_metadata.mat')
data_sets = ['HUP64_phaseII', 'HUP68_phaseII', 'HUP70_phaseII', 'HUP72_phaseII', 'HUP78_phaseII', 'HUP86_phaseII', 'Study 010', 'Study 011', 'Study 016', 'Study 020']
data_dict['subject']['IEEG']
all_channels = data_dict['subject']['Channels']
all_patients = data_dict['subject']['ID']
data_sets = ['HUP64', 'HUP68', 'HUP70', 'HUP72', 'HUP78', 'HUP86', 'MAYO010', 'MAYO011', 'MAYO016', 'MAYO020']
patient_idx = []
for dset in data_sets:
    patient_idx.append(all_patients.index(dset))
print(patient_idx)
all_channels = data_dict['subject']['Channels']
patient_chns = []
for idx in patient_idx:
    patient_chns.append(all_channels[idx])
#print(patient_chns)
soz = data_dict['subject']['Channels_Sz']
patient_soz = []
for idx in patient_idx:
    patient_soz.append(soz[idx])
#print(patient_soz)
patient_soz_flatten = []
for patient_sozi in patient_soz:
    if len(patient_sozi)==1:
        #print([patient_sozj for patient_sozj in patient_sozi])
        patient_soz_flatten.append([patient_sozj for patient_sozj in patient_sozi])
    else:
        #print([patient_sozj[0] for patient_sozj in patient_sozi])
        patient_soz_flatten.append([patient_sozj[0] for patient_sozj in patient_sozi])
#print(patient_soz_flatten)
patient_chns_flatten = []
for patient_chni in patient_chns:
#     print([patient_sozj[0] for patient_sozj in patient_sozi])
    patient_chns_flatten.append([patient_chnj[0] for patient_chnj in patient_chni])

#patient_chns_flatten
#print(patient_soz_flatten)
#print(patient_chns_flatten)
print(data_sets)
soz_idx = []
for pati in range(len(patient_chns_flatten)):
    ls = []
    for chn in patient_soz_flatten[pati]:
        try:
            thing_index = patient_chns_flatten[pati].index(chn)
            ls.append(patient_chns_flatten[pati].index(chn))
        except ValueError:
            thing_index = -1
    soz_idx.append(ls)
print(soz_idx)

# Part 1
# compute the FOS parameters for both ictal data and interictal snapshots for every patient and every ictal snapshot
# you should only have to run this one time.

seiz_num = 0
patient_num = 0

for patient in patients:
    soz_idx_pat = np.unique(np.array(soz_idx[patient_num]))
    for seizure in range(1, 9):
        if os.path.exists(main_pathname+'Data/' + patient+'/'+patient+'-ictal-block-'+str(seizure)+'.mat'):
            #ictal data
            #create directories for results 
            results = main_pathname +'Results/'+patient+'/'+'ictal-block-'+str(seizure)
            try:
                os.makedirs(results)
            except FileExistsError:
                pass
            try:
                os.makedirs(results+'/Reconstructions/Ictal/')
                os.makedirs(results+'/Reconstructions/Interictal/')
            except FileExistsError:
                pass
            #load data into python 
            struc = scipy.io.loadmat(main_pathname+'Data/' + patient+'/'+patient+'-ictal-block-'+str(seizure)+'.mat')
            sampling_rate = struc['Fs'][0][0]
            sampling_rate = int(np.ceil(sampling_rate))
            evData = struc['evData']
            #print(struc['channels_soz'])
            num_chns = evData.shape[0]
            #For patient Mayo020 the soz_idx_pat is empty, handled accordingly below 
            if soz_idx_pat.size != 0: 
                non_soz_idx_pat = np.delete(np.arange(num_chns), soz_idx_pat)
            else:
                non_soz_idx_pat = np.arange(num_chns)

            # compute the FOS from data and extract the eigenvalues, eigenvectors
            start_idx = 0
            num_windows = int(evData.shape[1]/(window_length*sampling_rate))
            xPred = np.zeros((num_chns,num_windows*window_length*sampling_rate))
            eigenvalues = np.zeros((num_chns,num_windows), dtype=np.complex128)
            eigenvectors = np.zeros((num_chns,num_chns,num_windows), dtype=np.complex128)
            alpha = np.zeros((num_chns,num_windows))
            A = np.zeros((num_chns, num_chns, num_windows))
            A_0 = np.zeros((num_chns, num_chns, num_windows))
            for window in range(num_windows):
                #X is windowed data
                X = evData[:,start_idx:start_idx+sampling_rate*window_length]
                meanX = np.mean(X, axis=1)
                X = X.T - meanX
                X = X.T
                fModel = fracOrdUU(verbose=-1)
                fModel.fit(X)
                fModel._AMat # A matrix
                fModel._order # fractional-order exponents
                # force to zero any fractional-order exponents that are small
                for i in range(len(fModel._order)):
                    if np.absolute(fModel._order[i])<0.01:
                        fModel._order[i] = 0
                alpha[:,window] = fModel._order
                A[:,:,window] = fModel._AMat[-1]
                xPred[:,start_idx:start_idx+sampling_rate*window_length] = reconstruct_FOS(alpha[:,window], A[:,:,window], X, num_chns, sampling_rate,window_length)
                

                # Calculate the eigenvalues
                v=np.zeros((num_chns))
                for chn in range(num_chns):
                    if fModel._order[chn]==0:
                        v[chn] = 1
                    else:
                        v[chn]=gamma(1-fModel._order[chn])/gamma(-fModel._order[chn])
                D=np.diag(v)
                A_0[:,:,window] = A[:,:,window] - D
                eigenvalues[:,window], eigenvectors[:,:,window] = np.linalg.eig(A_0[:,:,window])
                start_idx = start_idx+sampling_rate*window_length


            eec = int(20/window_length)
            onset = math.floor((eec*window_length+math.floor(diff[seiz_num]))/window_length)
            offset = math.floor((evData.shape[1]-20*sampling_rate)/(sampling_rate*window_length))

            ictal_data = {"alpha": alpha, "A": A, "A_0":A_0, "eigenvalues": eigenvalues,"eigenvectors": eigenvectors,"onset": onset, "offset": offset, "eec": eec, "xPred":xPred}
            scipy.io.savemat(main_pathname+'Data/'+ patient+'/'+'ictal-block-'+str(seizure)+"_ictal_parameters_2sec_window_.mat", ictal_data)
            
            # interictal data
            struc = scipy.io.loadmat(main_pathname+'Data/'+patient+'/'+patient+'-interictal-block-'+str(seizure)+'.mat')
            sampling_rate = struc['Fs'][0][0]
            sampling_rate = int(np.ceil(sampling_rate))
            evData = struc['evData']
            num_chns = evData.shape[0]
            
            # compute the FOS and extract the eigenvalues for the interictal data
            start_idx = 0
            num_windows = int(evData.shape[1]/(sampling_rate*window_length))
            xPred = np.zeros((num_chns,num_windows*window_length*sampling_rate))
            eigenvalues = np.zeros((num_chns,num_windows), dtype=np.complex128)
            eigenvectors = np.zeros((num_chns,num_chns,num_windows), dtype=np.complex128)
            alpha = np.zeros((num_chns,num_windows))
            A = np.zeros((num_chns, num_chns, num_windows))
            A_0 = np.zeros((num_chns, num_chns, num_windows))
            for window in range(num_windows):
                #X is windowed data
                X = evData[:,start_idx:start_idx+sampling_rate*window_length]
                print("x shape", X.shape)
                meanX = np.mean(X, axis=1)
                X = X.T - meanX
                X = X.T
                fModel = fracOrdUU(verbose=-1)
                fModel.fit(X)
                print(X.shape)
                fModel._AMat # A matrix
                fModel._order # fractional-order exponents
                # force to zero any fractional-order exponents that are small
                for i in range(len(fModel._order)):
                    if np.absolute(fModel._order[i])<0.01:
                        fModel._order[i] = 0
                alpha[:,window] = fModel._order
                A[:,:,window] = fModel._AMat[-1]
                xPred[:,start_idx:start_idx+sampling_rate*window_length] = reconstruct_FOS(alpha[:,window], A[:,:,window], X, num_chns, sampling_rate, window_length)
                
                # Calculate the eigenvalues
                v=np.zeros((num_chns))
                for chn in range(num_chns):
                    if fModel._order[chn]==0:
                        v[chn] = 1
                    else:
                        v[chn]=gamma(1-fModel._order[chn])/gamma(-fModel._order[chn])
                D=np.diag(v)
                A_0[:,:,window] = A[:,:,window] - D
                eigenvalues[:,window], eigenvectors[:,:,window] = np.linalg.eig(A[:,:,window] - D)
                start_idx = start_idx+sampling_rate*window_length

            # save FOS interictal parameters     
            interictal_data ={'A': A, 'A_0':A_0,'alpha': alpha, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'xPred':xPred}
            scipy.io.savemat(main_pathname+'Data/'+ patient+'/interictal-block-'+str(seizure)+"_interictal_parameters_2sec_window.mat", interictal_data)   
            
            seiz_num = seiz_num + 1
    patient_num = patient_num + 1



