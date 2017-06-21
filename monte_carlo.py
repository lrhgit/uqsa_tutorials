"""
Module with Monte Carlo methods for uncertainty and  sensitivity analysis using the chaospy package for sampling
"""
import numpy as np

def uq_measures_dummy_func():
    if __name__ == '__main__':
        from sensitivity_examples_nonlinear import generate_distributions
        from sensitivity_examples_nonlinear import linear_model

    # start uq
    # generate the distributions for the problem
    Nrv = 4
    c = 0.5
    zm = np.array([[0., i] for i in range(1, Nrv + 1)])
    wm = np.array([[i * c, i] for i in range(1, Nrv + 1)])
    jpdf = generate_distributions(zm, wm)

    # 1. Generate a set of Xs
    Ns = 20000
    Xs = jpdf.sample(Ns, rule='R').T  # <- transform the sample matrix

    # 2. Evaluate the model
    Zs = Xs[:, :Nrv]
    Ws = Xs[:, Nrv:]
    Ys = linear_model(Ws, Zs)

    # 3. Calculate expectation and variance
    EY = np.mean(Ys)
    VY = np.var(Ys, ddof=1)  # NB: use ddof=1 for unbiased variance estimator, i.e /(Ns - 1)

    print('E(Y): {:2.5f} and  Var(Y): {:2.5f}'.format(EY, VY))
    # end uq


# sample matrices
def generate_sample_matrices_mc(Ns, number_of_parameters, jpdf, sample_method='R'):

    Xtot = jpdf.sample(2*Ns, sample_method).transpose()
    A = Xtot[0:Ns, :]
    B = Xtot[Ns:, :]

    C = np.empty((number_of_parameters, Ns, number_of_parameters))
    # create C sample matrices
    for i in range(number_of_parameters):
        C[i, :, :] = B.copy()
        C[i, :, i] = A[:, i].copy()

    return A, B, C
# end sample matrices


# mc algorithm for variance based sensitivity coefficients
def calculate_sensitivity_indices_mc(y_a, y_b, y_c):

    # single output value y_a for one set of samples
    if len(y_c.shape) == 2:
        Ns, n_parameters = y_c.shape

        # for the first order index
        f0sq_first = np.sum(y_a*y_b)/ Ns 
        y_var_first = np.sum(y_b**2.)/(Ns-1) - f0sq_first

        # for the total index
        f0sq_total = (sum(y_a)/Ns)**2
        y_var_total = np.sum(y_a**2.)/(Ns-1) - f0sq_total

        s = np.zeros(n_parameters)
        st = np.zeros(n_parameters)

        for i in range(n_parameters):
            # first order index
            cond_var_X = np.sum(y_a*y_c[:, i])/(Ns - 1) - f0sq_first
            s[i] = cond_var_X/y_var_first

            # total index
            cond_exp_not_X = np.sum(y_b*y_c[:, i])/(Ns - 1) - f0sq_total
            st[i] = 1 - cond_exp_not_X/y_var_total

    # vector output value y_a,.. for one set of samples
    elif len(y_c.shape) == 3:
        n_y, Ns, n_parameters = y_c.shape
        # for the first order index
        f0sq_first = np.sum(y_a, axis=1) / Ns * np.sum(y_b, axis=1) / Ns
        y_var_first = np.sum(y_b ** 2., axis=1) / (Ns - 1) - f0sq_first

        # for the total index
        f0sq_total = (np.sum(y_a, axis=1) / Ns) ** 2
        y_var_total = np.sum(y_a ** 2., axis=1) / (Ns - 1) - f0sq_total

        s = np.zeros((n_parameters, n_y))
        st = np.zeros((n_parameters, n_y))

        for i in range(n_parameters):
            # first order index
            cond_var_X = np.sum(y_a * y_c[:, :, i], axis=1) / (Ns - 1) - f0sq_first

            s[i, :] = cond_var_X / y_var_first

            # total index
            cond_exp_not_X = np.sum(y_b * y_c[:, :, i], axis=1) / (Ns - 1) - f0sq_total
            st[i, :] = 1 - cond_exp_not_X / y_var_total

    return s, st
# end mc algorithm for variance based sensitivity coefficients
