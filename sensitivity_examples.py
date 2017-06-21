'''
Created on Dec 1, 2016

@author: leifh
'''
# matplotlib and settings
import matplotlib.pyplot as plt

# import modules
import numpy as np
from numpy import linalg as LA
import chaospy as cp
from sensitivity_examples_nonlinear import generate_distributions
from monte_carlo import generate_sample_matrices_mc
from monte_carlo import calculate_sensitivity_indices_mc
import pandas as pd
from _operator import index

cp.Sens_m

# start the linear model
def linear_model(w, z):
    return np.sum(w*z, axis=1)
# end the linear model


# calculate sens indices of non additive model
def mc_sensitivity_linear(Ns, jpdf, w, sample_method='R'):

    Nrv = len(jpdf)

    # 1. Generate sample matrices
    A, B, C = generate_sample_matrices_mc(Ns, Nrv, jpdf, sample_method)

    # 2. Evaluate the model
    Y_A, Y_B, Y_C = evaluate_linear_model(A, B, C, w)

    # 3. Approximate the sensitivity indices
    S, ST = calculate_sensitivity_indices_mc(Y_A, Y_B, Y_C)

    return A, B, C, Y_A, Y_B, Y_C, S, ST
# end calculate sens indices of non additive model


# model evaluation
def evaluate_linear_model(A, B, C, w):

    number_of_parameters = A.shape[1]
    number_of_sampless = A.shape[0]
    # 1. evaluate sample matrices A
    Y_A = linear_model(w, A)

    # 2. evaluate sample matrices B
    Y_B = linear_model(w, B)

    # 3. evaluate sample matrices C
    Y_C = np.empty((number_of_sampless, number_of_parameters))
    for i in range(number_of_parameters):
        z = C[i, :, :]
        Y_C[:, i] = linear_model(w, z)

    return Y_A, Y_B, Y_C
# end model evaluation


if __name__ == '__main__':

    # Set mean (column 0) and standard deviations (column 1) for each factor z. Nrv=nr. rows
    Nrv = 4  # number of random variables 
    zm = np.array([[0., i] for i in range(1, Nrv + 1)])

    # TODO: LR decide if the following lines should be kept
    # zm = np.zeros((Nrv, 2))
    # zm[0, 1] = 1
    # zm[1, 1] = 2
    # zm[2, 1] = 3
    # zm[3, 1] = 4

    # Set the weight
    c = 2
    w = np.ones(Nrv) * c

    # Generate distributions for each element in z and sample
    Ns = 500
    # jpdf = generate_distributions(zm)
    
    pdfs = []

    for i, z in enumerate(zm):
        pdfs.append(cp.Normal(z[0], z[1]))

    jpdf = cp.J(*pdfs)

    # generate Z
    Z = jpdf.sample(Ns)
    # evaluate the model
    Y = linear_model(w, Z.transpose())
    print(np.var(Y))

    # Scatter plots of data for visual inspection of sensitivity
    fig=plt.figure()
    for k in range(Nrv):
        plt.subplot(2, 2, k + 1)
        plt.plot(Z[k, :], Y[:], '.')
        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)
        
    fig.tight_layout()  # adjust subplot(s) to the figure area.
    
    # Theoretical sensitivity indices
    std_y = np.sqrt(np.sum((w * zm[:, 1])**2))
    s = w * zm[:,1]/std_y
    
    print("\nTheoretical sensitivity indices\n")
    row_labels= ['S_'+str(idx) for idx in range(1,Nrv+1)]
    print(pd.DataFrame(s**2, columns=['S anal'],index=row_labels).round(3))
          

    #  Expectation and variance from sampled values
    
    print("Expectation and std from sampled values\n")
    print('std(Y)={:2.3f} and relative error={:2.3f}'.format(np.std(Y, 0), (np.std(Y, 0) - std_y) / std_y))
    print('mean(Y)={:2.3f} and E(Y)={:2.3}'.format(np.mean(Y, 0), np.sum(zm[:,0]*w)))
    

    #  Standard Multivariate Regression
    import statsmodels.api as sm

    results = sm.OLS(Y, Z.T).fit()
    w_ols = results.params  # weights from ordinary least squares
    print(results.summary())
    relative_error = (w_ols - w) / w
    print("\n Regression coefficients\n")
    print('      w_ols |  rel.error \n')
    for k, (s_ref, s_sq) in enumerate(zip(w_ols, abs(relative_error))):
        print('S_{} : {:2.3f} |  {:2.3f}'.format(k + 1, s_ref, s_sq))
    # fig=plt.figure(figsize=(12,8))
    # fig = sm.graphics.plot_partregress_grid(results, fig=fig)
    # fig = plt.figure(figsize=(12, 8))
    # fig = sm.graphics.plot_ccpr_grid(results, fig=fig)


    # Scale the variables to obtain standardized regression coefficients (SRC)
    from scipy.stats.mstats import zscore

    res_standardize = sm.OLS(zscore(Y), zscore(Z.T)).fit()
    print(res_standardize.summary())
    beta = res_standardize.params
    relative_error = (beta ** 2 - s ** 2) / s ** 2
    print("\n Standardized parameters \n")
    print('        SRC | rel.error \n')
    for k, (s_ref, s_sq) in enumerate(zip(beta ** 2, abs(relative_error))):
        print('S_{} : {:2.3f} | {:2.3f}'.format(k + 1, s_ref, s_sq))

    # Monte Carlo
    # get joint distributions
    jpdf = generate_distributions(zm)

    Ns_mc = 1000000
    # calculate sensitivity indices
    A_s, B_s, C_s, f_A, f_B, f_C, S_mc, ST_mc = mc_sensitivity_linear(Ns_mc, jpdf, w)

    Sensitivities=np.column_stack((S_mc,s**2))
    row_labels= ['S_'+str(idx) for idx in range(1,Nrv+1)]
    print("First Order Indices")
    print(pd.DataFrame(Sensitivities,columns=['Smc','Sa'],index=row_labels).round(3))
    # end Monte Carlo

    # Polychaos computations
    Ns_pc = 80
    samples_pc = jpdf.sample(Ns_pc)
    polynomial_order = 4
    poly = cp.orth_ttr(polynomial_order, jpdf)
    Y_pc = linear_model(w, samples_pc.T)
    approx = cp.fit_regression(poly, samples_pc, Y_pc, rule="T")

    exp_pc = cp.E(approx, jpdf)
    std_pc = cp.Std(approx, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    
    
    S_pc = cp.Sens_m(approx, jpdf)

    Sensitivities=np.column_stack((S_mc,S_pc, s**2))
    print("\nFirst Order Indices")
    print(pd.DataFrame(Sensitivities,columns=['Smc','Spc','Sa'],index=row_labels).round(3))

#     print("\nRelative errors")
#     rel_errors=np.column_stack(((S_mc - s**2)/s**2,(S_pc - s**2)/s**2))
#     print(pd.DataFrame(rel_errors,columns=['Error Smc','Error Spc'],index=row_labels).round(3))

    # Polychaos convergence
    Npc_list = np.logspace(1, 3, 10).astype(int)
    error = []

    for i, Npc in enumerate(Npc_list):
        Zpc = jpdf.sample(Npc)
        Ypc = linear_model(w, Zpc.T)
        Npol = 4
        poly = cp.orth_chol(Npol, jpdf)
        approx = cp.fit_regression(poly, Zpc, Ypc, rule="T")
        s_pc = cp.Sens_m(approx, jpdf)
        error.append(LA.norm((s_pc - s**2)/s**2))

    plt.figure()
    plt.semilogy(Npc_list, error)
    _=plt.xlabel('Nr Z')
    _=plt.ylabel('L2-norm of error in Sobol indices')

    # # Scatter plots of data, z-slices, and linear model
    fig=plt.figure()

    Ndz = 10  # Number of slices of the Z-axes

    Zslice = np.zeros((Nrv, Ndz))  # array for mean-values in the slices
    ZBndry = np.zeros((Nrv, Ndz + 1))  # array for boundaries of the slices
    dz = np.zeros(Nrv)

    for k in range(Nrv):
        plt.subplot(2, 2, k + 1)

        zmin = np.min(Z[k, :])
        zmax = np.max(Z[k, :])  # each Z[k,:] may have different extremas
        dz[k] = (zmax - zmin) / Ndz

        ZBndry[k, :] = np.linspace(zmin, zmax, Ndz + 1) # slice Zk into Ndz slices
        Zslice[k, :] = np.linspace(zmin + dz[k] / 2., zmax - dz[k] / 2., Ndz) # Midpoint in the slice

        # Plot the the vertical slices with axvline
        for i in range(Ndz):
            plt.axvline(ZBndry[k, i], np.amin(Y), np.amax(Y), linestyle='--', color='.75')

        # Plot the data
        plt.plot(Z[k, :], Y[:], '.')
        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)
        plt.ylabel('Y')

        Ymodel = w[k] * Zslice[k, :]  # Produce the straight line

        plt.plot(Zslice[k, :], Ymodel)

        ymin = np.amin(Y); ymax = np.amax(Y)
        plt.ylim([ymin, ymax])
    
    fig.tight_layout()  # adjust subplot(s) to the figure area.    

    # # Scatter plots of averaged y-values per slice, with averaged data

    Zsorted = np.zeros_like(Z)
    Ysorted = np.zeros_like(Z)
    YsliceMean = np.zeros((Nrv, Ndz))

    fig=plt.figure()
    for k in range(Nrv):
        plt.subplot(2, 2, k + 1)

        # sort values for Zk, 
        sidx = np.argsort(Z[k, :]) #sidx holds the indexes for the sorted values of Zk
        Zsorted[k, :] = Z[k, sidx].copy()
        Ysorted[k, :] = Y[sidx].copy()  # Ysorted is Y for the sorted Zk

        for i in range(Ndz):
            plt.axvline(ZBndry[k, i], np.amin(Y), np.amax(Y), linestyle='--', color='.75')

            # find indexes of z-values in the current slice
            zidx_range = np.logical_and(Zsorted[k, :] >= ZBndry[k, i], Zsorted[k, :] < ZBndry[k, i + 1])

            if np.any(zidx_range):  # check if range has elements
                YsliceMean[k, i] = np.mean(Ysorted[k, zidx_range])
            else:  # set value to None if noe elements in z-slice
                YsliceMean[k, i] = None

        plt.plot(Zslice[k, :], YsliceMean[k, :], '.')
        
        

        # # Plot linear model
        Nmodel = 3
        zmin = np.min(Zslice[k, :])
        zmax = np.max(Zslice[k, :])

        zvals = np.linspace(zmin, zmax, Nmodel)
        #linear_model
        Ymodel = w[k] * zvals
        plt.plot(zvals, Ymodel)

        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)

        plt.ylim(ymin, ymax)
    
    fig.tight_layout()  # adjust subplot(s) to the figure area.
    
    SpoorMan=[np.nanvar(YsliceMean[k,:],axis=0)/np.var(Y) for k in range(4)]   
    print(SpoorMan)
    # end scatter plots y-values slice
    plt.show()
    plt.close()

    
    ## alternative
    #pdfs3 = [cp.Normal(mu, sig) for (mu, sig) in zm]

