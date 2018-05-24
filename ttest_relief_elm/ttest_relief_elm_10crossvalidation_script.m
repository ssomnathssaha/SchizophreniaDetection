%   Description: This script takes the data and applies t-test to it. 
%   Then after extracting relevant data from t-test, it applies Relief 
%   feature selection to select relevant features. We divide the whole
%   dataset into 10 parts. One of them is used for testing and rest 9 parts
%   are used for training. We have done ELM 200 times and repeated this 10
%   times.

%   Input: It takes the dataset as input.

%   Output: The accuracy of the trained machine.

clc;    %   Clear Command window
clear all;  %   Clear Workspace

load('DatafMRI15T_DU.mat');     %   load dataset
Labels=lab;     %   rename lab to Labels
clear('lab');       %   remove lab to save memory

[h,p,a,stat]=ttest2(X(1:30,:), X(31:60,:));     %   applied ttest
index=find(h==1);       %   save all index whose hypothesis is true
X=X(:,index);   %   extract data of 'index' 

[w bestidx] = RELIEF(X, Labels);    %   Relief applied

new_index_w = bestidx(1:285,:); % index of top 10% of most relevent data
% new_index_w = bestidx(1:570,:); % index of top 20% of most relevent data
% new_index_w = bestidx(1:855,:); % index of top 30% of most relevent data
% new_index_w = bestidx(1:1140,:); % index of top 40% of most relevent data
% new_index_w = bestidx(1:1600,:); % index of top 50% of most relevent data

new_data_X = X(:,new_index_w);  %   data of those index extracted

rng(5);     %   seeding

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