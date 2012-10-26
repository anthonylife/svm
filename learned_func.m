% Calculate the prediction error
function err = learned_func(i)
global train_set;
global alpha;
global kernel_func;
global b;

err = (alpha.*train_set.tag)'*(train_set.fea*train_set.fea(i,:)') - b;
