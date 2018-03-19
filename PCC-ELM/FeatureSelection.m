function [final_data,train_data,test_data] = FeatureSelection(r,X,Labels)
[R,C]=size(X);
count=0;
for i=1:C
    if r(i)> 0.6
        count=count+1;
    end
end
disp(count);
CC=count+1;
final_data = zeros(R,count+1);
final_data(:,1)= Labels(:,1);
count=0;
for i=1:C
    if r(i)> 0.5
        count=count+1;
        final_data(:,count+1)=X(:,i);
    end
end
train_data = zeros(40,CC);
test_data = zeros(20,CC);
index1=1;
index2=1;
for i=1:60
    if i < 21 || (i > 30 && i < 51)  
        train_data(index1,:) = final_data(i,:);
        index1=index1+1;
    else
        test_data(index2,:) = final_data(i,:);
        index2=index2+1;
    end
end