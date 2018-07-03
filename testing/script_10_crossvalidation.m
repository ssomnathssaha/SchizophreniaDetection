clc;
clear all;

load('DatafMRI15T_DU.mat');
Labels=lab;
% X( :, ~any(X,1) ) = [];

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

new_data_X=X(:,new_index_w);

rng(5);


[indCV]=crossvalind('Kfold',60,10);
indCV

X=new_data_X;
for i=1:10
    
    test = (indCV == i);
    train = ~test;
    
    X_train=X(train,:);  %fulldata Matrix instance used in prev ttest particular iteration
    % X_train=X_train(:,index);
    [r c]=size(X_train);
    
    lab_train=Labels(train,:);
    train_data=zeros(54,c+1);
    train_data(1:54,1)=lab_train;
    train_data(1:54,2:c+1)=X_train;
    
    X_test=X(test,:);
    lab_test=Labels(test,1);
    test_data=zeros(6,c+1);
    test_data(1:6,1)=lab_test;
    test_data(1:6,2:c+1)=X_test;
    
   for j=1:100
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, 1000, 'sig');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(train_data, test_data, 1, 100, 'lin_kernel', 10);
    end
    Accuracy(i)=mean(TestingAccuracy);
    i
end
ans=mean(Accuracy);
