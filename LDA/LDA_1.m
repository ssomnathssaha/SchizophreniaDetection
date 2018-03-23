%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML114
% Project Title: Implementation of Linear Discriminant Analysis in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [Y, W, lambda] = LDA_1(X, L)

    Classes=unique(L)';
    k=numel(Classes);

    n=zeros(k,1);
    C=cell(k,1);
    M=mean(X);
    S=cell(k,1);
    Sw=0;
    Sb=0;
    for j=1:k
        Xj=X(L==Classes(j),:);
        n(j)=size(Xj,1);
        C{j}=mean(Xj);
        S{j}=0;
        for i=1:n(j)
            S{j}=S{j}+(Xj(i,:)-C{j})'*(Xj(i,:)-C{j});
        end
        Sw=Sw+S{j};
        Sb=Sb+n(j)*(C{j}-M)'*(C{j}-M);
    end

    [W, LAMBDA]=eig(Sb,Sw);

    lambda=diag(LAMBDA);

    [lambda, SortOrder]=sort(lambda,'descend');

    W=W(:,SortOrder);

    Y=X*W;

end