%RBFKERNEL achieves Radial Basis Function.

function rbf_val = rbfKernel(x1, x2)
global two_sigma_squared;
two_sigma_squared = 2;

if size(x1,1) ~= size(x2,1),
    x2 = repmat(x2, size(x1,1), 1);
end

rbf_val = repmat(0.0, size(x1,1), 1);
for i=1:size(x1,1),
    rbf_val(i) = norm(x1(i,:)-x2(i,:), 2)/two_sigma_squared;
end
