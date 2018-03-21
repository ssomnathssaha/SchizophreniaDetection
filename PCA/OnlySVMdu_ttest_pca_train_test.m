clc;
close all;
clear all;

load('DatafMRI15T_DU.mat');

[indCV]=crossvalind('Kfold',60,60);
rng(2);

for i=1:60
        % ttest for finding significance level b/w healthy and schz patient
        if(indCV(i)<=30)
            [h,p,a,stat]=ttest2(X(setdiff([1:30],indCV(i)),:), X(31:60,:));
        else
            [h,p,a,stat]=ttest2(X(1:30,:), X(setdiff([31:60],indCV(i)),:));
        end
        H=h;                              % hypothesis matrix
        
        FeatSel=find(H==1);    %finding the index of those voxels among 153594 vox whose null hypo has been rejected.
        %indx stores the rank wise the index of
        %selected t score no of voxel (not reak voxel index)
        %indxAll=indx;  %storing the ranked index from 1 to ~2800
        %tempFeat=FeatSel; %stores real voxel index which was ranked as indx
        %taking first 300 from ranked index
        %fetching real voxel index from 1-153594
        
        
        Xtrain=X(setdiff([1:60],indCV(i)),:);  %fulldata Matrix instance used in prev ttest particular iteration
        Xtrain=Xtrain(:,FeatSel);
        %features from chrome which has 1
        labtrain=lab(setdiff([1:60],indCV(i)),:);   %Label from training data used in particular iteration of ttest/
        [coeff_tr,score_tr,latent_tr,tsquared_tr,explained_tr,Ind,mu_tr,coeff_all] = mod_data_red_pca(Xtrain,97);
        
        Xtest=X(indCV(i),FeatSel);
        labtest=lab(indCV(i),:);
        
        [r c]=size(Xtest);
        Xtest=Xtest-repmat(mu_tr,r,1); % subtracting mean from data
        ts_score=coeff_tr'*Xtest';
        ts_score=ts_score';
        % linear svm
%          model = svmtrain(double(labtrain),double(score_tr),'-t=0 -c=cost(j) -q');
%          predlab = svmpredict(double(labtest), double(ts_score),model,'-q');
%         
        %liblinear L1 norm
%             model = train(labtrain,sparse(score_tr),'-s=5 -c=100 -q');
%             predlab = predict(labtest, sparse(ts_score),model,'-q');
        
%         rbf svm
           % model = svmtrain(double(labtrain),double(score_tr),'-t 3 -q');
           % predlab = svmpredict(double(labtest), double(ts_score),model,'-q');
        
           
           
         [m,n] = size(score_tr);
         train_data(:,1) = labtrain;
         train_data(:,2:n+1) = score_tr;
         test_data(:,1) = labtest;
         test_data(:,2:n+1) = ts_score;
         [TrainingTime, TestingTime, TrainingAccuracy(i), TestingAccuracy(i)] = elm(train_data, test_data, 1, 200, 'sig');
         i
        
        %knn
%         predlab=knnclassify( ts_score, score_tr, labtrain);
        
%          if(labtest == predlab)
%              acc=100;
%          else
%              acc=0;
%          end
%          accIni(i,1)=acc;                      %storing accuracy
end
%      s(j,1)=mean(accIni);
%      clear accIni;
% end
mean(TestingAccuracy)