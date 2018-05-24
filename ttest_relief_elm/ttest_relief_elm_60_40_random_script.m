%   Description: This script takes the data and applies t-test to it. 
%   Then after extracting relevant data from t-test, it applies Relief 
%   feature selection to select relevant features. Then it divides the 
%   dataset into 6:4 ratio to get the training data and testing data. 
%   And applies ELM on it. In this process, we have run ELM 200 times 
%   and the whole process after feature selection 20 times and mean is 
%   taken to get the final accuracy. We take number of hidden neurons as
%   random manner within a range and repeat the process.

%   Input: It takes the dataset as input.

%   Output: The accuracy of the trained machine.

clc;    %   clear command window
clear all;  %   clear workspace

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

rng(16);    %   seeding

size = size(X,1);   % compute number of subjects in size
split = round(size * 0.6);  %   split point

hidden_start = 100;
hidden_end = 1000;
hidden = hidden_start + round((hidden_end - hidden_start)*rand(100,1));


for ii=1:100
current_hidden_units = hidden(ii)
ii
for i=1:20
    seq = randperm(size);   %   Permute randomly
    train_data_60 = X(seq(1:split),:);  %   Take first 60% data for training
    train_lab_60 = Labels(seq(1:split));    %   Take first 60% labels for training
    test_data_40 = X(seq(split+1:end),:);   %   Take last 40% data for training
    test_lab_40 = Labels(seq(split+1:end)); %   Take last 40% labels for training
    train_data = cat(2, train_lab_60, train_data_60);   %   Concatenate Training Label and Training Data
    test_data = cat(2, test_lab_40, test_data_40);  %   Concatenate Testing Label and Testing Data

    %   Apply ELM 200 times
    for j=1:10
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, hidden(ii), 'sig');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(train_data, test_data, 1, 200, 'lin_kernel', 1);
    end
    Accuracy(i) = mean(TestingAccuracy);    %   take the mean of 200 ELM results
end
temp_ans(ii)=mean(Accuracy);
end
final_ans=mean(temp_ans);