%SVM schedule all other related functions to do initialization,
%  loading source data, model training, prediction and evalution.
%
%  Main procedures of the program:
%       1. Global model parameter and related variable setting
%       2. Loading feature files (Default 5 file);
%       3. Choose 4 file as train data and train svm model paramter;
%       4. Evaluate on the left file and output eval result;
%       5. Repeat 3 and 4 procedures 5 times to 5-cross-validation.
%
%  Decision function: y = w'x - b;
%
% @date: 10/24/2012


% 1. Global model parameter and related variable setting
% ======================================================
global part_num;
part_num = 5;
global cross_eval_num;
cross_eval_num = 5;
global train_set;
train_set = struct('fea', [], 'tag', []);
global test_set;
test_set = struct('fea', [], 'tag', []);

% Model paramter
% --------------
global C;
C = 0.05;   %penalty parameter for slack variable
global tolerance;
tolerance = 0.001;  %tolerable error rate
global eps;
eps = 0.001; %minimum change rate
global two_sigma_squared;
two_sigma_squared = 2;    % RBF kernel parameter
global b;
b = 0;   %model threshold

% Source and target file path
% ---------------------------
cate_tag_f = '../features/ins_category_tag.txt';
ins_fea_f = '../features/feature.full.sparse.txt';
model_f = './svm.model';

% Kernel function setting
% -----------------------
global kernel_func;
if ~exist('RBF', 'var'),
    RBF = 0;
end

if RBF == 1,
    kernel_func = @rbfKernel;  %switch to rbf kernel
elseif RBF == 0,
    kernel_func = @linearKernel;   %default linear kernel
end

% 2. Loading feature files (Default 5 file)
% =========================================
ins_tag = load(cate_tag_f);
ins_feature = load(ins_fea_f);
ins_fea_mat = spconvert(ins_feature);   %convert sparse rep
clear ins_feature;

% 3. Choose 4 file as train data and train svm model paramter
% ===========================================================
ins_files = repmat(struct('fea', [], 'tag', []), part_num, 1);
ins_num = length(ins_tag);

rr = randperm(ins_num);
seg_num = floor(ins_num/part_num);

temp_idx = 1;
for i=1:part_num-1,
    ins_files(i).fea = ins_fea_mat(rr(temp_idx:temp_idx+seg_num),:);
    ins_files(i).tag = ins_tag(rr(temp_idx:temp_idx+seg_num));
    temp_idx = temp_idx + seg_num + 1;
end
ins_files(part_num).fea = ins_fea_mat(rr(temp_idx:end),:);
ins_files(part_num).tag = ins_tag(rr(temp_idx:end),:);

% 5-cross validation
% ==================
for i=1:cross_eval_num,
    fprintf('Cross validation:%d...\n', i);
    test_set = ins_files(i);
    for j=1:part_num,
        if j~=i,
            train_set.fea = [train_set.fea;ins_files(j).fea];
            train_set.tag = [train_set.tag;ins_files(j).tag];
        end
    end
    
    % Test Sample
    % ----------------------------------------------------
    %train_set.fea = [];                                %|
    %train_set.tag = [];                                %|
    % Make test data                                    %|
    %train_set.fea = [2,1;0.5,0.5;1,1;-1,1;1,-1;-1,-1]; %|
    %train_set.tag = [1;-1;1;-1;-1;-1];                 %|
    % ----------------------------------------------------

    % Implict passing parameter, using global parameter instead.
    tic;
    alpha = svmTrain();
    toc;
    %alpha
    err = trainError(alpha);
    fprintf('Train error: %f...\n', err);
    
    [F1_score, pre1, recall1, F2_score, pre2, recall2] = svmPredict(alpha);
    fprintf('Test result:\nHockey-->> Precision:%f; Recall:%f; F:%f!\n', pre1, recall1, F1_score); 
    fprintf('Baseball-->> Precision:%f; Recall:%f; F:%f!\n', pre2, recall2, F2_score);

    train_set.fea = [];
    train_set.tag = [];
end
