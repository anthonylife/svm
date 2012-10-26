%RBFKERNEL achieves Radial Basis Function.

function rbf_val = rbfKernel(x1, x2)
global two_sigma_squared;
rbf_val = exp(-norm(x1-x2,'fro')/two_sigma_squared);
