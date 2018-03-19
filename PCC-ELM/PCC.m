function [r]=PCC(A,B)

disp('Calculating the Pearson Product-moment coefficient..');
disp('Returning (r)');
 
%dataCSV=csvread(fileName);
[N,C]=size(A);
r=zeros(1,C);
    for i = 1:C
        X=A(:,i);
        Y=B(:,1);
        XY=X.*Y;
        X2 = X.*X;
        Y2=Y.*Y;
        num = (C.*(sum(XY)))-(sum(X).*sum(Y));
        densq = (C.*sum(X2)-(sum(X).^2)).*(C.*sum(Y2)-(sum(Y).^2));
        den = sqrt(densq);
        correlation = num./den;
        r(i)=correlation;
    end
end