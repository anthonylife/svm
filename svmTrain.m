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
% Defined in other place
global train_set;
global C;
global b;

% Global definition
global tr_ins_num;
tr_ins_num = size(train_set.fea, 1);
global alpha;
alpha = repmat(0.0, tr_ins_num, 1);
global error_cache; % Function:ensure not to find same (a1,a2) 
                    %   continously
error_cache = repmat(0.0, tr_ins_num, 1);
maxIter = 25;

num_changed = 0;    % number of variable updating happend
examine_all = 1;    % scan all train instance

% Loop alpha with value > 0 && value < C first.
% Note: for the first time, as all alpha equa to
%   0, loop all alpha first.
iter_num = 0;
while num_changed > 0 | examine_all,
    num_changed = 0;
    if examine_all,
        for i=1:tr_ins_num,
            %disp('Non seperating point');
            num_changed = num_changed + examineSample(i);
        end
    else,
        for i=1:tr_ins_num,
            if alpha(i) > 0 & alpha(i) < C,
                %disp('seperating point');
                num_changed = num_changed + examineSample(i);
            end
        end
    end

    % when examine_all = 0, if num_changed = 0, loop exit.
    if examine_all == 1,
        examine_all = 0;
    % setting to scan all alpha variables
    elseif num_changed == 0,
        examine_all = 1;
    end

    iter_num = iter_num + 1;
    if iter_num == maxIter, break; end
end
