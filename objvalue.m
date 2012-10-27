function [objval_old, objval_new] = objvalue(alpha1_new, i1, alpha2_new, i2)

global alpha;
global train_set;

tep_alpha = alpha;
tep_alpha(i1) = alpha1_new;
tep_alpha(i2) = alpha2_new;

objval_old = 1/2*(train_set.tag.*alpha)'*train_set.fea*train_set.fea'*(train_set.tag.*alpha);
objval_new = 1/2*(train_set.tag.*tep_alpha)'*train_set.fea*train_set.fea'*(train_set.tag.*tep_alpha);

objval_old = objval_old - sum(alpha);
objval_new = objval_new - sum(tep_alpha);
