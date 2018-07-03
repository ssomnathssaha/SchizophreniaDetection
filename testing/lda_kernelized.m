function[w]=lda_kernelized(data)
% This file was written by Bilwaj K. Gaonkar 
% This is intended to provide an LDA implementation for high 
% dimension low sample size data. n<<p
% n=> number of subjects
% p=> number of dimensions
% data.Y => data labels 
% data.X => data values (p x n)
 
n1=sum((data.Y==1));
n2=sum((data.Y==2));

e1=(data.Y==1);
e2=(data.Y==2);
e=e1+e2;
K=data.X'*data.X;%linear kernel replace with your own for non linear kernel functions

P1=(e1'/sum(e1)-e'/sum(e))*(e1'/sum(e1)-e'/sum(e))';
P2=(e2'/sum(e2)-e'/sum(e))*(e2'/sum(e2)-e'/sum(e))';
SB=K*(n1*P1+n2*P2)*K';

Q1=((eye(sum(e),sum(e))-repmat(e1'/sum(e1),1,sum(e)))*diag(e1))*((eye(sum(e),sum(e))-repmat(e1'/sum(e1),1,sum(e)))*diag(e1))';
Q2=((eye(sum(e),sum(e))-repmat(e2'/sum(e2),1,sum(e)))*diag(e2))*((eye(sum(e),sum(e))-repmat(e2'/sum(e2),1,sum(e)))*diag(e2))';

% Calculate the within class variance (SW)
SW=K*(Q1+Q2)*K;
%Regularization has to be smaller by two~three orders of magnitude than the
%smallest non zero eigenvalue
%Empirically setting regularizer using max eigenvalue
k=rank(SW);
evals=eig(SW);
evals=sort(evals,1,'descend');
minval=evals(k);
%Chosen 10^-8 because its the closest value to singularity below this matrix is often badly scaled 
%Change this appropriately in case of errors
v=inv(SW+(10^-8)*minval*eye(size(SW,1)))*SB;
[evec,eval]=eig(v);


%Pick  eigenvector corresponding to max eigenvalue
[r,s]=max(max(eval));
Alpha=evec(:,s);
w=data.X*Alpha;
w=w/norm(w,2);
% Sort eigen vectors according to eigen values (descending order) and
% neglect eigen vectors according to small eigen values
% v=evec(greater eigen value)
% or use all the eigen vectors

% project the data of the first and second class respectively
%y2=c2*v
%y1=c1*v
