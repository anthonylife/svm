%SVMTRAIN achieves SMO algorithm to train model parameters.
%
%  Input:
%    train_set: struct having 'fea' and 'tag';
%    C: parameter penalty for slack variable;
%    tolerance:...
%    eps:
%    two_sigma_squared:
%    kernel_func:
%
%  Output:
%    alpha: largarange paramter, determing decision boundary
%
%  @date: 10/25/2012
%

function alpha = svmTrain()
% defined in other place
global train_set;
global C;
global b;

% global definition
global tr_ins_num = length(train_set.fea);
global alpha = repmat(0.0, tr_ins_num, 1);
global error_cache = 0 - train_set.tag;

num_changed = 0;    % number of variable updating happend
examine_all = 1;    % scan all train instance

% Loop alpha with value > 0 && value < C first.
% Note: for the first time, as all alpha equa to
%   0, loop all alpha first.
while num_changed > 0 || examine_all,
    num_changed = 0;
    if examine_all,
        for i=1:tr_ins_num,
            num_changed = num_changed + examineSample(i);
        end
    else,
        for i=1:tr_ins_num,
            if alpha(i) > 0 && alpha(i) < C,
                num_changed = num_changed + examineSample(i);
            end
        end
    end

    % when examine_all = 0, if num_changed = 0, loop exit.
    if examine_all,
        examine_all = 0;
    % setting to scan all alpha variables
    elseif ~num_changed,
        examine_all = 1;
    end
end
