%TRAINERROR calculate the error rate in training data.

function err = trainError(alpha)
% defined in other place
global train_set;
global tr_ins_num;
global kernel_func;
global b;

n_total = 0;
n_error = 0;

p_tag = repmat(-1, tr_ins_num, 1);
p_tag(find(((alpha.*train_set.tag)'*kernel_func(train_set.fea, train_set.fea) - b) > 0)) = 1;

n_error = length(find(p_tag ~= train_set.tag));

n_total = tr_ins_num;
err = n_error/n_total;
