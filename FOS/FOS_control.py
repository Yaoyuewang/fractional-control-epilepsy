import os 
import cvxpy as cp
import numpy as np
import scipy.io
import math 

# patients = ['HUP64', 'HUP68','HUP70','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
patients = ['HUP64', 'HUP68','HUP72','HUP78','HUP86','MAYO010','MAYO011','MAYO016','MAYO020']
# patients = ['HUP86']
main_pathname = 'c:/Users/yaoyu/Documents/Epilepsy_research/'

window_length = 3
ictal_windows = 18
for patient in patients:
    for seizure in range(1, 9):  
        filepath = os.path.join(main_pathname, 'data', patient, f"{patient}-ictal-block-{seizure}.mat")
        if os.path.exists(filepath):
            print(f"{patient}_{seizure}")
            # results = os.path.join(main_pathname, 'Results', patient, f"ictal-block-{seizure}")
            FOS_params = scipy.io.loadmat(os.path.join(main_pathname, 'data_v2', patient, f"ictal-block-{seizure}_parameters_3sec_1iter.mat"))
            raw_data = scipy.io.loadmat(filepath)
            sampling_rate = int(np.ceil(raw_data['Fs'][0][0]))
            A = FOS_params["A"]
            A_0 = FOS_params["A_0"]
            alpha = FOS_params["alpha"]
            num_chns = FOS_params["A"].shape[0]
            evData = raw_data["evData"]
            
            ictal_start = 20
            # Changing A Problem 1
            A_0_inst = A_0[:,:,ictal_start]
            A_0_inst = np.array(A_0_inst)

            # Define variables
            P = cp.Variable((num_chns,num_chns),symmetric=True) 
            L = cp.Variable((num_chns,num_chns))

            # Construct matrix V
            P_T_A0 = P @ cp.transpose(A_0_inst)
            A0_P = A_0_inst @ P
            V = cp.bmat([[P, P_T_A0 + cp.transpose(L)],
                        [A0_P + L, P]])

            # Define objective
            objective = cp.Minimize(cp.norm(L, 1) + cp.norm(P, 1))

            # Define constraints
            constraints = [V - np.eye(2*num_chns) >> 0,
                        P - np.eye(num_chns) >> 0]

            # Set up and solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK)  # Try a different solver if needed

            # Extract results
            opt_L = np.array(L.value)
            opt_P = np.array(P.value)

            A_tilde = np.matmul(opt_L,np.linalg.inv(opt_P))
            X = evData[:,ictal_start:]
            num_chns = X.shape[0]
            eigs_before = np.zeros((num_chns, ictal_windows), dtype=np.complex128)
            eigs_after  = np.zeros_like(eigs_before)

            infit = 5  # the maximum number of states in the past should be considered          
            numStep = 1 # number of steps ahead
            p = 1 # number of past states
            TSteps = sampling_rate*ictal_windows
            #simulated data 
            xPred = np.zeros((num_chns, TSteps*numStep))
            xPred[:, 0:numStep] = X[:, 0:numStep]
            XTemp = np.zeros((num_chns, TSteps*numStep))
            for i in range(1,TSteps):
                XTemp[:, 0:i*numStep] = X[:, 0:i*numStep]
                for stepInd in range(numStep):
                    for chInd in range(num_chns):
                        alpha_inst = alpha[chInd,ictal_start+math.floor(i/sampling_rate)]
                        #check whether alpha is fractional
                        if math.ceil(alpha_inst) != alpha_inst:
                            trailLen = np.min(np.array([infit, i*numStep + stepInd - 1]))
                            j = np.arange(1, trailLen + 1)
                            preFact = scipy.special.gamma(-alpha_inst + j) / (scipy.special.gamma(-alpha_inst) * scipy.special.gamma(j + 1))
                            XTemp[chInd, i* numStep + stepInd] = XTemp[chInd, i*numStep + stepInd] - np.sum(XTemp[chInd, i*numStep + stepInd - j] * preFact)

                    XUse = np.zeros((num_chns,p))
                    for pInd in range(p):
                        if i*numStep + stepInd - pInd < 1:
                            break
                        XUse[:, pInd] = XTemp[:, i*numStep + stepInd - pInd]
                    A_no_ctrl = A_0[:, :, ictal_start + math.floor(i / sampling_rate)]
                    A_current = A_no_ctrl + A_tilde
                    XTemp[:, i*numStep + stepInd] = XTemp[:, i*numStep + stepInd] + np.dot(A[:,:,ictal_start+math.floor(i/sampling_rate)]+A_tilde, XUse[:, pInd]) 

                xPred[:, (i)*numStep:(i+1)*numStep] = XTemp[:, (i)*numStep:(i+1)*numStep]
 
                if (i - 1) % sampling_rate == 0:
                    col = (i - 1) // sampling_rate
                    eigs_before[:, col] = np.linalg.eigvals(A_no_ctrl)
                    eigs_after [:, col] = np.linalg.eigvals(A_current)
            
            savepath = os.path.join(main_pathname, 'data_v2', patient, f"controlled_data_block_{seizure}.mat")
            scipy.io.savemat(savepath, {"evData": xPred, "Fs": sampling_rate, "eigs_before": eigs_before, "eigs_after": eigs_after})

