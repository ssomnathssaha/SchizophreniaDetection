clear all;
load('ContrastAll_15T_60jnu.mat');
rng(5);
X=double(X);


[indCV]=crossvalind('Kfold',60,60);

X=new_data_X;
for i=1:60
    %     if(indCV(i)<=30)
    %             [h,p,a,stat]=ttest2(X(setdiff([1:30],indCV(i)),:), X(31:60,:));
    %         else
    %             [h,p,a,stat]=ttest2(X(1:30,:), X(setdiff([31:60],indCV(i)),:));
    %     end
    %                                   % hypothesis matrix
    %
    %     index=find(h==1); %selecting best features
    
    
    X_train=X(setdiff([1:60],indCV(i)),:);  %fulldata Matrix instance used in prev ttest particular iteration
    % X_train=X_train(:,index);
    [r c]=size(X_train);
    
    lab_train=Labels(setdiff([1:60],indCV(i)),:);
    train_data=zeros(59,c+1);
    train_data(1:59,1)=lab_train;
    train_data(1:59,2:c+1)=X_train;
    
    % X_test=X(indCV(i),index);
    X_test=X(indCV(i),:);
    lab_test=Labels(indCV(i),1);
    test_data=zeros(1,c+1);
    test_data(1,1)=lab_test;
    test_data(1,2:c+1)=X_test;
    
   for j=1:100
        [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm(train_data, test_data, 1, 400, 'hardlim');
        % [TrainingTime, TestingTime, TrainingAccuracy(j), TestingAccuracy(j)] = elm_kernel(train_data, test_data, 1, 140, 'lin_kernel', 0.01);
    end
    Accuracy(i)=mean(TestingAccuracy);
    i
end
ans=mean(Accuracy);