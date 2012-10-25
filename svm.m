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
% @date: 10/24/2012


% 1. Global model parameter and related variable setting
% ======================================================
global part_num = 5;
global corss_eval_num = 5;
global train_set = struct('fea', [], 'tag', []);
global test_set = struct('fea', [], 'tag', []);

% Model paramter
% --------------
global C = 0.05;   %penalty parameter for slack variable
global tolerance = 0.001;  %tolerable error rate
global eps = 0.001 %minimum change rate
global two_sigma_squared=2;    % RBF kernel parameter
global b = 0;   %model threshold

% Source and target file path
% ---------------------------
cate_tag_f = '../Features/ins_category_tag.txt';
ins_fea_f = '../Features/feature_all_sparse.txt'
model_f = './svm.model'

% Kernel function setting
% -----------------------
global kernel_func = @linearKernel;   %default linear kernel
if RBF == 1,
    kernel_func = @rbfKernel;  %switch to rbf kernel
end

% 2. Loading feature files (Default 5 file)
% =========================================
ins_tag = load(cate_tag_f);
ins_feature = load(ins_fea_f);
ins_fea_mat = spconvert(ins_feature);
clear ins_feature;

% 3. Choose 4 file as train data and train svm model paramter
% ===========================================================
ins_files = repmat(struct('fea', [], 'tag', []), part_num, 1);
ins_num = length(ins_tag);

rr = randperm(ins_tag);
seg_num = floor(ins_num/part_num);

temp_idx = 1;
for i=1:part_num-1,
    ins_files(i).fea = ins_fea_mat(rr(temp_idx:temp_idx+seg_num),:);
    ins_files(i).tag = ins_tag(rr(temp_idx:temp_idx+seg_num));
    temp_idx = temp_idx + seg_num + 1;
end
ins_files(part_num).fea = ins_fea_mat(rr(temp_idx:),:);
ins_files(part_num).tag = ins_tag(rr(temp_idx:),:);

% 5-cross validation
% ==================
for i=1:cross_eval_num,
    test_set = ins_files(i);
    for j=1:part_num,
        if j~=i,
            train_set.fea = [train_set.fea;ins_files(j).fea];
            train_set.tag = [train_set.tag;ins_files(j).tag];
        end
    end

    % Implict passing parameter, using global parameter instead.
    alpha = svmTrain();

    err = trainError(alpha);
    fprintf('Train error: %f...\n', err);
    
    [F1_score, F2_score] = svmPredict(alpha);
    fprintf('Test result:\nF score for hockey: %f, F score for baseball: %f...', F1_score, F2_score);
end
