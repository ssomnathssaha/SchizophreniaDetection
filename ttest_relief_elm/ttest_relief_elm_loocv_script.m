%   Description: This script takes the data and applies t-test to it. 
%   Then after extracting relevant data from t-test, it applies Relief 
%   feature selection to select relevant features. We take each subject as
%   testing and rest subjects for training purpose. In every step we have
%   done 200 times ELM. At last we have taken mean to get the final
%   accuracy.

%   Input: It takes the dataset as input.

%   Output: The accuracy of the trained machine.

clc;    %   clear command window
clear all;  %   clear workspace

load('DatafMRI15T_DU.mat');     %   load dataset
Labels=lab;     %   rename lab to Labels
clear('lab');       %   remove lab to save memory

[h,p,a,stat]=ttest2(X(1:30,:), X(31:60,:));     %   applied ttest
index=find(h==1);       %   save all index whose hypothesis is rejected
X=X(:,index);   %   extract data of 'index' 

[w bestidx] = RELIEF(X, Labels);    %   Relief applied

new_index_w = bestidx(1:285,:); % index of top 10% of most relevent data
% new_index_w = bestidx(1:570,:); % index of top 20% of most relevent data
% new_index_w = bestidx(1:855,:); % index of top 30% of most relevent data
% new_index_w = bestidx(1:1140,:); % index of top 40% of most relevent data
% new_index_w = bestidx(1:1600,:); % index of top 50% of most relevent data

new_data_X = X(:,new_index_w);  %   data of those index extracted

rng(5); %   seeding

[indCV]=crossvalind('Kfold',60,60); %   store randomly 1-60

X=new_data_X; 
for i=1:60    
    %   make the train data
    X_train=X(setdiff([1:60],indCV(i)),:);  %fulldata Matrix instance used in prev ttest particular iteration
    lab_train=Labels(setdiff([1:60],indCV(i)),:);
    train_data=cat(2,lab_train,X_train);
    
    %   make the test data
    X_test=X(indCV(i),:);
    lab_test=Labels(indCV(i),1);
    test_data=cat(2,lab_test,X_test);
    
   %    apply ELM 200 times 
   for j=1:10
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, 4122, 'sig');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(double(train_data), double(test_data), 1, 500, 'lin_kernel', 100);
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = OSELM(train_data, test_data, 1, 1500, 'sig', 50, 20);
   end
    Accuracy(i)=mean(TestingAccuracy);  %   take the mean of 200 ELM results
    i
end
ans=mean(Accuracy); %   Final accuracy is stored in 'ans'