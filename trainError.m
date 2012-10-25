%TRAINERROR calculate the error rate in training data.

function err = trainError(alpha)
% defined in other place
global train_set;
global tr_ins_num;
global b;

n_total = 0;
n_error = 0;

for i=1:tr_ins_num,
    p_tag = (train_set.fea(i)*alpha + b) > 0;
    if ~p_tag,
        p_tag = -1;
    end
    if p_tag ~= train_set.tag(i),
        n_error += 1;
    end
end

n_total = tr_ins_num;
err = n_error/n_total;
