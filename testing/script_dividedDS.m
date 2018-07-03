clc;
clear all;

load('DatafMRI15T_DU.mat');
Labels=lab;

[h,p,a,stat]=ttest2(X(1:30,:), X(31:60,:));
index=find(h==1);
X=X(:,index);


data.X=X';
data.Y=Labels';

[w] = lda_kernelized(data);

[sorted_value, sorted_index]=sort(w,'descend');

% new_index_w = sorted_index(1:285,:); % 10%
% new_index_w = sorted_index(1:570,:); % 20%
% new_index_w = sorted_index(1:855,:); % 30%
new_index_w = sorted_index(1:1140,:); % 40%
% new_index_w = sorted_index(1:1600,:); % 

new_data_X = X(:,new_index_w);

rng(5);

% X = cat(2, lab, new_data_X);
% 
% [aa, bb] = size(X); 
% 
% train_data = zeros(40,1141);
% test_data = zeros(20,1141);
% index1=1;
% index2=1;
% for i=1:60
%     if i < 21 || (i > 30 && i < 51)  
%         train_data(index1,:) = X(i,:);
%         index1=index1+1;
%     else
%         test_data(index2,:) = X(i,:);
%         index2=index2+1;
%     end
% end

size = size(X,1);
split = round(size * 0.6);

for i=1:20
    seq = randperm(size);
    train_data_80 = X(seq(1:split),:);
    train_lab_80 = lab(seq(1:split));
    test_data_20 = X(seq(split+1:end),:);
    test_lab_20 = lab(seq(split+1:end));
    train_data = cat(2, train_lab_80, train_data_80);
    test_data = cat(2, test_lab_20, test_data_20);


    for j=1:100
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, 1000, 'sig');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(train_data, test_data, 1, 200, 'lin_kernel', 1);
    end
    i
    Accuracy(i) = mean(TestingAccuracy);
end
ans = mean(Accuracy);
