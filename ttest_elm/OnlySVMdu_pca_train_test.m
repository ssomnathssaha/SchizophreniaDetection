
X=double(X);
[indCV]=crossvalind('Kfold',60,60);

for i=1:60
    
    Xtrain=X(setdiff([1:60],indCV(i)),:);  %fulldata Matrix instance used in prev ttest particular iteration
    %features from chrome which has 1
    labtrain=lab(setdiff([1:60],indCV(i)),:);   %Label from training data used in particular iteration of ttest/
    [coeff_tr,score_tr,latent_tr,tsquared_tr,explained_tr,Ind,mu_tr,coeff_all] = mod_data_red_pca(Xtrain,95);

    Xtest=X(indCV(i),:);
    labtest=lab(indCV(i),:);
    
    [r c]=size(Xtest);
    Xtest=Xtest-repmat(mu_tr,r,1); % subtracting mean from data
    ts_score=coeff_tr'*Xtest';
    ts_score=ts_score';
    % linear svm
        model = svmtrain(double(labtrain),double(score_tr),'-t 0 -q -c 100');
        predlab = svmpredict(double(labtest), double(ts_score),model,'-q');
    
    
        %liblinear L1 norm
%             model = train(labtrain,sparse(score_tr),'-s=5 -c=100 -q');
%             predlab = predict(labtest, sparse(ts_score),model,'-q');
%         
        
    %rbf svm
%             model = svmtrain(double(labtrain),double(score_tr),'-t 3 -q');
%             predlab = svmpredict(double(labtest), double(ts_score),model,'-q');
        
        
        %knn
        predlab=knnclassify( ts_score, score_tr, labtrain);
    
    if(labtest == predlab)
        acc=100;
    else
        acc=0;
    end
    accIni(i,1)=acc;                      %storing accuracy
    %accIni(i,2)=s;
end
mean(accIni)