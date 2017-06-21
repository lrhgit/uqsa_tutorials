'''
Created on Mai 1, 2017

@author: Vinzenz Eck and Leif Rune Hellevik
'''
# modules for plotting
import matplotlib.pyplot as plt

# import modules
import numpy as np
import chaospy as cp
from monte_carlo import generate_sample_matrices_mc
from monte_carlo import calculate_sensitivity_indices_mc
from xlwt.Utils import col_by_name
 


# start the linear model
def linear_model(w, z):
    return np.sum(w*z, axis=1)
# end the linear model


# definition of the distributions
def generate_distributions(zm, wm=None):
    # define marginal distributions
    if wm is not None:
        zm = np.append(zm, wm, axis=0)
    marginal_distributions = [cp.Normal(*mu_sig) for mu_sig in zm]
    # define joint distributions
    jpdf = cp.J(*marginal_distributions)

    return jpdf
# end definition of the distributions


# calculate sens indices of non additive model
def monte_carlo_sens_nonlin(Ns, jpdf, sample_method='R'):

    N_prms = len(jpdf)

    # 1. Generate sample matrices
    XA, XB, XC = generate_sample_matrices_mc(Ns, N_prms, jpdf, sample_method)

    # 2. Evaluate the model
    Y_A, Y_B, Y_C = evaluate_non_additive_linear_model(XA, XB, XC)

    # 3. Approximate the sensitivity indices
    S, ST = calculate_sensitivity_indices_mc(Y_A, Y_B, Y_C)

    return XA, XB, XC, Y_A, Y_B, Y_C, S, ST
# end calculate sens indices of non additive model


# model evaluation
def evaluate_non_additive_linear_model(X_A, X_B, X_C):

    N_prms = X_A.shape[1]
    Ns = X_A.shape[0]
    N_terms = int(N_prms / 2)
    # 1. evaluate sample matrices X_A
    Z_A = X_A[:, :N_terms]  # Split X in two vectors for X and W
    W_A = X_A[:, N_terms:]
    Y_A = linear_model(W_A, Z_A)

    # 2. evaluate sample matrices X_B
    Z_B = X_B[:, :N_terms]
    W_B = X_B[:, N_terms:]
    Y_B = linear_model(W_B, Z_B)

    # 3. evaluate sample matrices X_C
    Y_C = np.empty((Ns, N_prms))
    for i in range(N_prms):
        x = X_C[i, :, :]
        z = x[:, :N_terms]
        w = x[:, N_terms:]
        Y_C[:, i] = linear_model(w, z)

    return Y_A, Y_B, Y_C
# end model evaluation

# polynomial chaos sensitivity analysis
def polynomial_chaos_sens(Ns_pc, jpdf, polynomial_order, poly=None, return_reg=False):
    N_terms = int(len(jpdf) / 2)
    # 1. generate orthogonal polynomials
    poly = poly or cp.orth_ttr(polynomial_order, jpdf)
    # 2. generate samples with random sampling
    samples_pc = jpdf.sample(size=Ns_pc, rule='R')
    # 3. evaluate the model, to do so transpose samples and hash input data
    transposed_samples = samples_pc.transpose()
    samples_z = transposed_samples[:, :N_terms]
    samples_w = transposed_samples[:, N_terms:]
    model_evaluations = linear_model(samples_w, samples_z)
    # 4. calculate generalized polynomial chaos expression
    gpce_regression = cp.fit_regression(poly, samples_pc, model_evaluations)
    # 5. get sensitivity indices
    Spc = cp.Sens_m(gpce_regression, jpdf)
    Stpc = cp.Sens_t(gpce_regression, jpdf)
    
    if return_reg:
        return Spc,Stpc,gpce_regression    
    else:
        return Spc, Stpc
               
    
# end polynomial chaos sensitivity analysis

def generate_distributions_n_dim(N_terms=4):
    # Set mean (column 0) and standard deviations (column 1) for each factor z. r factors=nr. rows
    # number of factors

    # zm = np.zeros((N_terms, 2))
    # zm[0, 1] = 1
    # zm[1, 1] = 2
    # zm[2, 1] = 3
    # zm[3, 1] = 4
    # wm = np.zeros_like(zm)
    # c = 0.5
    # wm[:, 0] = [i * c for i in range(1, N_terms + 1)]
    # wm[:, 1] = zm[:, 1].copy()  # the standard deviation is the same as for  zmax

    c = 0.5
    zm = np.array([[0., i] for i in range(1, N_terms + 1)])
    wm = np.array([[i * c, i] for i in range(1, N_terms + 1)])

    Z_pdfs = []  # list to hold probability density functions (pdfs) for Z and w
    w_pdfs = []
    for i in range(N_terms):
        Z_pdfs.append(cp.Normal(zm[i, 0], zm[i, 1]))
        w_pdfs.append(cp.Normal(wm[i, 0], wm[i, 1]))

    pdfs_list = Z_pdfs + w_pdfs
    jpdf = cp.J(*pdfs_list)  # *-operator to unpack the arguments out of a list or tuple

    return jpdf


# func analytic sens coefficients
def analytic_sensitivity_coefficients(zm, wm):
    # calculate the analytic sensitivity coefficients

    VarY = np.sum(zm[:, 1]**2 * (wm[:, 0]**2 + wm[:, 1]**2), axis=0)
    Sz = wm[:, 0]**2 * zm[:, 1]**2/VarY # first order indices
    Sw = np.zeros_like(Sz)
    Szw= wm[:, 1]**2 * zm[:, 1]**2/VarY  # second order indices
    StZ = (wm[:, 0]**2 * zm[:, 1]**2 + wm[:, 1]**2 * zm[:, 1]**2)/VarY # total indices
    Stw = (wm[:, 1]**2 * zm[:, 1]**2)/VarY

    # join sensitivity arrays
    Sa = np.append(Sz, Sw)
    Sta = np.append(StZ, Stw)

    # end inside analytic sens
    return Sa,Szw,Sta
# end analytic sens coefficients

if __name__ == '__main__':

#     np.set_printoptions(precision=3)
#     np.set_printoptions(suppress=True)

    # definition of mu and sig for z and w
    N_terms = 4
    c = 0.5
    zm = np.array([[0., i] for i in range(1, N_terms+1)])
    wm = np.array([[i * c, i] for i in range(1, N_terms+1)])

    # to see the effect of changing the values in zm uncomment and change one of these lines and re-run

    # zm[0, 1] = 1
    # zm[1, 1] = 20
    # zm[2, 1] = 3
    # zm[3, 1] = 10

    # end definition of mu and sig

    # generate the joint distribution
    jpdf = generate_distributions(zm, wm)
    
    

    # Scatter plots of data for visual inspection of sensitivity
    N_plot=100
    N_prms = len(jpdf)
    N_terms = N_prms//2

    Xs=jpdf.sample(N_plot,sample_method='R').transpose()  
    Zs = Xs[:, :N_terms]  # Split X in two vectors for X and W
    Ws = Xs[:, N_terms:]
    Ys = linear_model(Ws, Zs)

    scatter = plt.figure('Scatter plots')
    for k in range(N_terms):
        plt.subplot(2, N_terms, k + 1)
        _=plt.plot(Zs[:, k], Ys[:], '.')
        plt.xlabel('Z{}'.format(k+1))
        plt.ylim([-150, 150])

        plt.subplot(2, N_terms, k + 1 + N_terms)
        _=plt.plot(Ws[:, k], Ys[:], '.')
        plt.xlabel('W{}'.format(k+1))
        plt.ylim([-150, 150])
    scatter.tight_layout()    
    # end scatter plots of data for visual inspection of sensitivity

    # sensitivity analytical values
    Sa, Szw, Sta = analytic_sensitivity_coefficients(zm, wm)
 

    # Monte Carlo
    #Ns_mc = 1000000 # Number of samples mc
    Ns_mc = 10000 # Number of samples mc
    # calculate sensitivity indices with mc
    A_s, B_s, C_s, f_A, f_B, f_C, Smc, Stmc = monte_carlo_sens_nonlin(Ns_mc, jpdf)

    # compute with Polynomial Chaos
    Ns_pc = 200
    polynomial_order = 3
    
    # calculate sensitivity indices with gpc
    Spc, Stpc, gpce_reg = polynomial_chaos_sens(Ns_pc, jpdf, polynomial_order,return_reg=True)

    # compare the computations
    import pandas as pd
    row_labels  = ['X_'+str(x) for x in range(1,N_terms*2+1)]
    S=np.column_stack((Sa,Spc,Smc,Sta,Stpc,Stmc))
    S_table = pd.DataFrame(S, columns=['Sa','Spc','Smc','Sta','Stpc','Stmc'], index=row_labels)  
    print(S_table.round(3))

    # Second order indices with gpc
    
    S2 = cp.Sens_m2(gpce_reg, jpdf) # second order indices with gpc
    
    # print all second order indices
    print(pd.DataFrame(S2,columns=row_labels,index=row_labels).round(3))
    
    # sum all second order indices 
    SumS2=np.sum(np.triu(S2))
    print('\nSum Sij = {:2.2f}'.format(SumS2))
    
    # sum all first and second order indices
    print('Sum Si + Sij = {:2.2f}\n'.format(np.sum(Spc)+SumS2))
    
    # compare nonzero second order indices with analytical indices 
    Szw_pc=[S2[i,i+N_terms] for i in range(N_terms) ]
    Szw_table=np.column_stack((Szw_pc,Szw,(Szw_pc-Szw)/Szw))
    print(pd.DataFrame(Szw_table,columns=['Szw','Szw pc','Error%']).round(3))
    
    # end second order
    convergence_analysis = False
    if convergence_analysis:
        # Convergence analysis
        # Convergence Monte Carlo with random sampling
        list_of_samples = np.array([10000, 50000, 100000, 500000, 1000000])
        s_mc_err = np.zeros((len(list_of_samples), N_prms))
        st_mc_err = np.zeros((len(list_of_samples), N_prms))
        # average over
        N_iter = 5
        print('MC convergence analysis:')
        for i, N_smpl in enumerate(list_of_samples):
            print('    N_smpl {}'.format(N_smpl))
            for j in range(N_iter):
                A_s, XB, XC, Y_A, Y_B, Y_C, S, ST = monte_carlo_sens_nonlin(N_smpl,
                                                                                jpdf,
                                                                                sample_method='R')
                s_mc_err[i] += np.abs(S - Sa)
                st_mc_err[i] += np.abs(ST - Sta)
                print('         finished with iteration {} of {}'.format(1 + j, N_iter))
            s_mc_err[i] /= float(N_iter)
            st_mc_err[i] /= float(N_iter)
        # Plot results for monte carlo
        fig_random = plt.figure('Random sampling - average of indices')
        fig_random.suptitle('Random sampling - average of indices')

        ax = plt.subplot(1, 2, 1)
        plt.title('First order sensitivity indices')
        _=plt.plot(list_of_samples / 1000, np.sum(s_mc_err, axis=1), '-')
        ax.set_yscale('log')
        _=plt.ylabel('abs error')
        _=plt.xlabel('number of samples [1e3]')

        ax1 = plt.subplot(1, 2, 2)
        plt.title('Total sensitivity indices')
        _=plt.plot(list_of_samples / 1000, np.sum(st_mc_err, axis=1), '-')
        ax1.set_yscale('log')
        _=plt.ylabel('abs error')
        _=plt.xlabel('number of samples [1e3]')

        # Plot results for monte carlo figure individual
        fig_random = plt.figure('Random sampling')
        fig_random.suptitle('Random sampling')
        for l, (s_e, st_e) in enumerate(zip(s_mc_err.T, st_mc_err.T)):
            ax = plt.subplot(1, 2, 1)
            plt.title('First order sensitivity indices')
            plt.plot(list_of_samples / 1000, s_e, '-', label='S_{}'.format(l))
            ax.set_yscale('log')
            _=plt.ylabel('abs error')
            _=plt.xlabel('number of samples [1e3]')
            _=plt.legend()

            ax1 = plt.subplot(1, 2, 2)
            plt.title('Total sensitivity indices')
            _=plt.plot(list_of_samples / 1000, st_e, '-', label='ST_{}'.format(l))
            ax1.set_yscale('log')
            _=plt.ylabel('abs error')
            _=plt.xlabel('number of samples [1e3]')
            plt.legend()

        # Convergence Polynomial Chaos
        list_of_samples = np.array([140, 160, 200, 220])
        s_pc_err = np.zeros((len(list_of_samples), N_prms))
        st_pc_err = np.zeros((len(list_of_samples), N_prms))
        polynomial_order = 3
        # average over
        N_iter = 4
        print('PC convergence analysis:')
        poly = cp.orth_ttr(polynomial_order, jpdf)
        for i, N_smpl in enumerate(list_of_samples):
            print('    N_smpl {}'.format(N_smpl))
            for j in range(N_iter):
                # calculate sensitivity indices
                Spc, Stpc = polynomial_chaos_sens(N_smpl, jpdf, polynomial_order, poly)
                s_pc_err[i] += np.abs(Spc - Sa)
                st_pc_err[i] += np.abs(Stpc - Sta)
                print('         finished with iteration {} of {}'.format(1 + j, N_iter))
            s_pc_err[i] /= float(N_iter)
            st_pc_err[i] /= float(N_iter)

        # Plot results for polynomial chaos
        fig_random = plt.figure('Polynomial Chaos - average of indices')
        fig_random.suptitle('Polynomial Chaos - average of indices')

        ax = plt.subplot(1, 2, 1)
        plt.title('First order sensitivity indices')
        _=plt.plot(list_of_samples, np.sum(s_pc_err, axis=1), '-')
        ax.set_yscale('log')
        _=plt.ylabel('abs error')
        _=plt.xlabel('number of samples [1e3]')

        ax1 = plt.subplot(1, 2, 2)
        plt.title('Total sensitivity indices')
        _=plt.plot(list_of_samples, np.sum(st_pc_err, axis=1), '-')
        ax1.set_yscale('log')
        _=plt.ylabel('abs error')
        _=plt.xlabel('number of samples [1e3]')

        # Plot results for polynomial chaos individual
        fig_random = plt.figure('Polynomial Chaos')
        fig_random.suptitle('Polynomial Chaos')
        for l, (s_e, st_e) in enumerate(zip(s_pc_err.T, st_pc_err.T)):
            ax = plt.subplot(1, 2, 1)
            plt.title('First order sensitivity indices')
            _=plt.plot(list_of_samples, s_e, '-', label='S_{}'.format(l))
            ax.set_yscale('log')
            plt.ylabel('abs error')
            plt.xlabel('number of samples [1e3]')
            plt.legend()

            ax1 = plt.subplot(1, 2, 2)
            plt.title('Total sensitivity indices')
            _=plt.plot(list_of_samples, st_e, '-', label='ST_{}'.format(l))
            ax1.set_yscale('log')
            plt.ylabel('abs error')
            plt.xlabel('number of samples [1e3]')
            plt.legend()

        # # Convergence Monte Carlo with sobol sampling
        # list_of_samples = np.array([10000, 50000, 100000, 500000, 1000000])
        # s_mc_err = np.zeros((len(list_of_samples), N_prms))
        # st_mc_err = np.zeros((len(list_of_samples), N_prms))
        # # average over
        # N_iter = 10
        # for i, N_smpl in enumerate(list_of_samples):
        #     for j in range(N_iter):
        #         A_s, XB, XC, Y_A, Y_B, Y_C, S, ST = monte_carlo_sens(N_smpl,
        #                                                                  jpdf,
        #                                                                  sample_method='S')
        #         s_mc_err[i] += np.abs(S - Sa)
        #         st_mc_err[i] += np.abs(ST - Sta)
        #
                # print('MC convergence analysis: N_smpl {} - finished with iteration {} of {}'.format(N_smpl, 1 + j, N_iter))
        #     s_mc_err[i] /= float(N_iter)
        #     st_mc_err[i] /= float(N_iter)
        #
        # fig_sobol = plt.figure('Sobol sampling')
        # fig_sobol.suptitle('Sobol sampling')
        # for l, (s_e, st_e) in enumerate(zip(s_mc_err.T, st_mc_err.T)):
        #     ax = plt.subplot(1, 2, 1)
        #     plt.title('First order sensitivity indices')
        #     plt.plot(list_of_samples/1000, s_e, '-', label='S_{}'.format(l))
        #     ax.set_yscale('log')
        #     plt.ylabel('abs error')
        #     plt.xlabel('number of samples [1e3]')
        #     plt.legend()
        #
        #     ax1 = plt.subplot(1, 2, 2)
        #     plt.title('Total sensitivity indices')
        #     plt.plot(list_of_samples/1000, st_e, '-', label='ST_{}'.format(l))
        #     ax1.set_yscale('log')
        #     plt.ylabel('abs error')
        #     plt.xlabel('number of samples [1e3]')
        #     plt.legend()
        #
        # fig_random = plt.figure('Sobol sampling - average of indices')
        # fig_random.suptitle('Sobol sampling - average of indices')
        #
        # ax = plt.subplot(1, 2, 1)
        # plt.title('First order sensitivity indices')
        # plt.plot(list_of_samples / 1000, np.sum(s_mc_err, axis=1), '-')
        # ax.set_yscale('log')
        # plt.ylabel('abs error')
        # plt.xlabel('number of samples [1e3]')
        #
        # ax1 = plt.subplot(1, 2, 2)
        # plt.title('Total sensitivity indices')
        # plt.plot(list_of_samples / 1000, np.sum(st_mc_err, axis=1), '-')
        # ax1.set_yscale('log')
        # plt.ylabel('abs error')
        # plt.xlabel('number of samples [1e3]')

    plt.show()
    plt.close()
