clear all;
clc;
load('DatafMRI15T_DU.mat');
Labels = lab;
clear('lab');

R = PCC(X,Labels);
[final_data,train_data,test_data] = FeatureSelection(R,X,Labels);

rng(5);

% for i=1:500
%      [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, i, 'sig');
%      j=j+1;
% end
%[TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = OSELM(train_data, test_data, 1, 30, 'hardlim', 30, 5);
%[TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = B_ELM(train_data, test_data, 30, 'hardlim', 30);
%ELM_regularized_LXL(train_data, test_data, 0, 20, 'hardlim',100);
%plot(TestingAccuracy);
%plot(TrainingAccuracy);
%X( :, ~any(X,1) ) = [];
for j=1:200
    [TrainingTime, TestingTime, TrainAccuracy(j), TestAccuracy(j)] = elm(train_data, test_data, 1, 600, 'sin');
end
    
TestingAccuracy = mean(TestAccuracy);
TestingSD = std(TestAccuracy);