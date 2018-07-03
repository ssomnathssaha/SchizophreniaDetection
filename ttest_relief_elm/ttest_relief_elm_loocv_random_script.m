%   Description: This script takes the data and applies t-test to it. 
%   Then after extracting relevant data from t-test, it applies Relief 
%   feature selection to select relevant features. We take each subject as
%   testing and rest subjects for training purpose. In every step we have
%   done 200 times ELM. At last we have taken mean to get the final
%   accuracy. We take number of hidden neurons as
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

% new_index_w = bestidx(1:285,:); % index of top 10% of most relevent data
% new_index_w = bestidx(1:570,:); % index of top 20% of most relevent data
% new_index_w = bestidx(1:855,:); % index of top 30% of most relevent data
% new_index_w = bestidx(1:1140,:); % index of top 40% of most relevent data
new_index_w = bestidx(1:1600,:); % index of top 50% of most relevent data

new_data_X = X(:,new_index_w);  %   data of those index extracted

rng(5); %   seeding

[indCV]=crossvalind('Kfold',60,60); %   store randomly 1-60

X=new_data_X;

hidden_start = 100;
hidden_end = 1500;
hidden = hidden_start + round((hidden_end - hidden_start)*rand(100,1));


for ii=1:100
current_hidden_units = hidden(ii)
ii
for i=1:60    
    X_train=X(setdiff([1:60],indCV(i)),:);  %fulldata Matrix instance used in prev ttest particular iteration
    [r c]=size(X_train);
    
    %   make the train data
    lab_train=Labels(setdiff([1:60],indCV(i)),:);
    train_data=zeros(59,c+1);
    train_data(1:59,1)=lab_train;
    train_data(1:59,2:c+1)=X_train;
    
    %   make the test data
    X_test=X(indCV(i),:);
    lab_test=Labels(indCV(i),1);
    test_data=zeros(1,c+1);
    test_data(1,1)=lab_test;
    test_data(1,2:c+1)=X_test;
    
   %    apply ELM 200 times 
   for j=1:10
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, hidden(ii), 'sig');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(train_data, test_data, 1, 500, 'lin_kernel', 10);
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = OSELM(train_data, test_data, 1, 500, 'sig', 20, 20);
   end
    Accuracy(i) = mean(TestingAccuracy);    %   take the mean of 200 ELM results
end
temp_ans(ii)=mean(Accuracy);
end
final_ans=mean(temp_ans);
error = temp_ans - final_ans;
errorbar(temp_ans,error)