import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# === The useful help function ===
# show help for uniform distributions
cp.Uniform?

# show help for sample generation
cp.samplegen?
# end help

# === Distributions ===
# simple distributions
rv1 = cp.Uniform(0, 1)
rv2 = cp.Normal(0, 1)
rv3 = cp.Lognormal(0, 1, 0.2, 0.8)
print(rv1, rv2, rv3)
# end simple distributions

# joint distributions
joint_distribution = cp.J(rv1, rv2, rv3)
print(joint_distribution)
# end joint distributions

# creating iid variables
X = cp.Normal()
Y = cp.Iid(X, 4)
print(Y)
# end creating iid variables

# === Sampling ===
# sampling in chaospy
u = cp.Uniform(0,1)
u.sample?
# end sampling chaospy

# example sampling
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)
number_of_samples = 350
samples_random = joint_distribution.sample(size=number_of_samples, rule='R')
samples_hammersley = joint_distribution.sample(size=number_of_samples, rule='M')

fig1, ax1 = plt.subplots()
ax1.set_title('Random')
ax1.scatter(*samples_random)
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')

fig2, ax2 = plt.subplots()
ax2.set_title('Hammersley sampling')
ax2.scatter(*samples_hammersley)
ax2.set_xlabel("Uniform 1")
ax2.set_ylabel("Uniform 2")
ax2.axis('equal')
# end example sampling

# === quadrature ===
# quadrature in polychaos
cp.generate_quadrature?
# end quadrature in polychaos

# example quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 5

nodes_gaussian, weights_gaussian = cp.generate_quadrature(order=order, domain=joint_distribution, rule='G')
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C')

print('Number of nodes gaussian quadrature: {}'.format(len(nodes_gaussian[0])))
print('Number of nodes clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[1])))


fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_gaussian, marker='o', color='b')
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example quadrature

# example sparse grid quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 2
# sparse grid has exponential growth, thus a smaller order results in more points
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C', growth=True)
nodes_clenshaw_sparse, weights_clenshaw_sparse = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C', sparse=True, growth=True)

print('Number of nodes normal clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[0])))
print('Number of nodes clenshaw-curtis quadrature with sparse grid : {}'.format(len(nodes_clenshaw_sparse[0])))

fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.scatter(*nodes_clenshaw_sparse, marker= 'o', color='b')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example sparse grid quadrature

# example orthogonalization schemes
# a normal random variable
n = cp.Normal(0, 1)

x = np.linspace(0,1, 50)
# the polynomial order of the orthogonal polynomials
polynomial_order = 3

poly = cp.orth_bert(polynomial_order, n, normed=True)
print('Bertran recursion {}'.format(poly))
ax = plt.subplot(221)
ax.set_title('Bertran recursion')
_=plt.plot(x, poly(x).T)
_=plt.xticks([])

poly = cp.orth_chol(polynomial_order, n, normed=True)
print('Cholesky decomposition {}'.format(poly))
ax = plt.subplot(222)
ax.set_title('Cholesky decomposition')
_=plt.plot(x, poly(x).T)
_=plt.xticks([])

poly = cp.orth_ttr(polynomial_order, n, normed=True)
print('Discretized Stieltjes / Thre terms reccursion {}'.format(poly))
ax = plt.subplot(223)
ax.set_title('Discretized Stieltjes ')
_=plt.plot(x, poly(x).T)

poly = cp.orth_gs(polynomial_order, n, normed=True)
print('Modified Gram-Schmidt {}'.format(poly))
ax = plt.subplot(224)
ax.set_title('Modified Gram-Schmidt')
_=plt.plot(x, poly(x).T)
# end example orthogonalization schemes

# _Linear Regression_
# linear regression in chaospy
cp.fit_regression?
# end linear regression in chaospy


# example linear regression
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.orth_ttr(polynomial_order, joint_distribution)

# 3.1 generate samples
number_of_samples = 100
samples = joint_distribution.sample(size=number_of_samples, rule='R')

# 3.2 evaluate the simple model for all samples
model_evaluations = samples[0]+samples[1]*samples[0]

# 3.3 use regression to generate the polynomial chaos expansion
gpce_regression = cp.fit_regression(poly, samples, model_evaluations)
# end example linear regression


# _Spectral Projection_
# spectral projection in chaospy
cp.fit_quadrature?
# end spectral projection in chaospy


# example spectral projection
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.orth_ttr(polynomial_order, joint_distribution)

# 4.1 generate quadrature nodes and weights
order = 5
nodes, weights = cp.generate_quadrature(order=order, domain=joint_distribution, rule='G')

# 4.2 evaluate the simple model for all nodes
model_evaluations = nodes[0]+nodes[1]*nodes[0]

# 4.3 use quadrature to generate the polynomial chaos expansion
gpce_quadrature = cp.fit_quadrature(poly, nodes, weights, model_evaluations)
# end example spectral projection

# example uq
exp_reg = cp.E(gpce_regression, joint_distribution)
exp_ps =  cp.E(gpce_quadrature, joint_distribution)

std_reg = cp.Std(gpce_regression, joint_distribution)
str_ps = cp.Std(gpce_quadrature, joint_distribution)

prediction_interval_reg = cp.Perc(gpce_regression, [5, 95], joint_distribution)
prediction_interval_ps = cp.Perc(gpce_quadrature, [5, 95], joint_distribution)

print("Expected values   Standard deviation            90 % Prediction intervals\n")
print(' E_reg |  E_ps     std_reg |  std_ps                pred_reg |  pred_ps')
print('  {} | {}       {:>6.3f} | {:>6.3f}       {} | {}'.format(exp_reg,
                                                                  exp_ps,
                                                                  std_reg,
                                                                  str_ps,
                                                                  ["{:.3f}".format(p) for p in prediction_interval_reg],
                                                                  ["{:.3f}".format(p) for p in prediction_interval_ps]))
# end example uq

# example sens
sensFirst_reg = cp.Sens_m(gpce_regression, joint_distribution)
sensFirst_ps = cp.Sens_m(gpce_quadrature, joint_distribution)

sensT_reg = cp.Sens_t(gpce_regression, joint_distribution)
sensT_ps = cp.Sens_t(gpce_quadrature, joint_distribution)

print("First Order Indices           Total Sensitivity Indices\n")
print('       S_reg |  S_ps                 ST_reg |  ST_ps  \n')
for k, (s_reg, s_ps, st_reg, st_ps) in enumerate(zip(sensFirst_reg, sensFirst_ps, sensT_reg, sensT_ps)):
    print('S_{} : {:>6.3f} | {:>6.3f}         ST_{} : {:>6.3f} | {:>6.3f}'.format(k, s_reg, s_ps, k, st_reg, st_ps))
# end example sens

