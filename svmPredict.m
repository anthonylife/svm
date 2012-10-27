%SVMPREDICT gives prediction in test data and doing evaluation.
%
%  Note: hockey '1'; baseball '2'.
%  @date: 10/26/2012

function [F1_score, precision1, recall1, F2_score, precision2, recall2] = svmPredict(alpha)
% defined in other place
global train_set;
global test_set;
global kernel_func;
global b;

te_ins_num = length(test_set.tag);

tp_hockey = 0;
tp_baseball = 0;
tp_fp_hockey = 0;
tp_fp_baseball = 0;
tp_fn_hockey = 0;
tp_fn_baseball = 0;

% prediction
pre_tag = (alpha.*train_set.tag)'*kernel_func(train_set.fea, test_set.fea) - b;
pre_tag(find(pre_tag>0)) = 1;
pre_tag(find(pre_tag~=1)) = -1;

tp_fp_hockey = length(find(pre_tag == 1));
tp_fp_baseball = length(find(pre_tag == -1));
tp_hockey = length(intersect(find(pre_tag == 1), find(test_set.tag == 1)));
tp_baseball = length(intersect(find(pre_tag == -1), find(test_set.tag == -1)));

tp_fn_hockey = length(find(test_set.tag == 1));
tp_fn_baseball = length(find(test_set.tag == -1));

precision1 = tp_hockey/tp_fp_hockey;
recall1 = tp_hockey/tp_fn_hockey;
precision2 = tp_baseball/tp_fp_baseball;
recall2 = tp_baseball/tp_fn_baseball;

F1_score = 2*precision1*recall1/(precision1+recall1);
F2_score = 2*precision2*recall2/(precision2+recall2);
