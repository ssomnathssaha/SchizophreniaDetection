function [w bestidx] = RELIEF(data, labels, T)
% function w = RELIEF(data, labels, T) 
% RELIEF - Kira & Rendell 1992
% T is number of patterns to use
% Defaults to all patterns if not specified.
%
% The license is in the LICENSE file.

if ~exist('T','var')
    T=size(data,1);
end

idx = randperm(length(labels));		%	Random Permutation from 1 to 60
idx = idx(1:T);

w = zeros(size(data,2),1);			%	153000 * 1 array of zeros
for t = 1:T							% 	T=60
    
    x = data(idx(t),:);				%	one row of X. (one instance)
    y = labels(idx(t));				%	one row of labels (label of that instance)
    
    %copy the x
    protos = repmat(x, length(labels), 1);							%	(size(x,1)*60) , (size(x,2)*1)  = (60,153000)
    %measure the distance from x to every other example
    distances = [sqrt(sum((data-protos).^2,2)) labels];				%	distance between protos and other examples
    %sort them according to distances (find nearest neighbours)
    [distances originalidx] = sortrows(distances,1);
   
    foundhit = false;  hitidx=0;
    foundmiss = false; missidx=0;
	
    i=2; 								%start from the second one because distance between one instance to itself is 0
    while (~foundhit || ~foundmiss)

        if distances(i,2) == y
            hitidx = originalidx(i);
            foundhit = true;
        end
        if distances(i,2) ~= y
            missidx = originalidx(i);
            foundmiss = true;
        end
        
        i=i+1;

    end
    
    alpha = 1/T;
    for f = 1:size(data,2)%each feature
        hitpenalty  = (x(f)-data(hitidx,f))  / (max(data(:,f))-min(data(:,f)));
        misspenalty = (x(f)-data(missidx,f)) / (max(data(:,f))-min(data(:,f)));
        
        w(f) = w(f) - alpha*hitpenalty^2 + alpha*misspenalty^2;
    end
    
end

[~,bestidx] = sort(w,'descend');


