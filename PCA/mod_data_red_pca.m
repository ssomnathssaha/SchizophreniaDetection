function [reduced_coeff,reduced_data,latent,tsquared,explained,Ind,mu,coeff_all] = mod_data_red_pca(data,totalVarRequied)
if nargin<2
    totalVarRequied=95; % default 95% of the total variance
end
[coeff_all,score,latent,tsquared,explained,mu] = pca(data);
pricipal = totalVarRequied; 
pspace = explained(1); % take the first important feature
Ind = 1;
for i = 2:length(explained)
    if pspace < pricipal
        pspace = explained(i) + pspace;
        Ind = i;
    else
        break
    end
end
score = score(:,1:Ind);
reduced_data = score;
reduced_coeff=coeff_all(:,1:Ind);