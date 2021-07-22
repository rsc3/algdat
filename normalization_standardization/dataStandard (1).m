function [xStandard, indx] = dataStandard(X,decision)
% This code is for educational and research purposes of comparisons. This
% is code that provides the use of the z-score normalization 
% (standarization).
%
% Input:
%   X [l x n] - l is the number of exemplars/samples/observation
%             - n is the dimension of the data
%  decision [1 x 1] - determines what to do in the event a standard
%                     deviation is equal to zero
%           decision = 1 - removes the features whos std dev = 0
%           decision = 2 - if std dev = 0, then X/mean(X)
%           Default, decision = 1
%
% Output:
%   xStandard [l x n] - standardized data according to:
%                                 X - mean(X)         
%                           S = --------------                                  
%                                   std(X)                                       
%  indx [1 x m] - where m is the number of features which have a std dev
%                 equal to zero. If no feature contins a std dev of zero
%                 the indx is returned as an empty matrix.
%
% Each feature was separately standardized, by substracting its mean and 
% dividing by the standard deviation. To avoid dividing by zero, e.g. s = 0
% the feature is either removed if decision = 1, or the feature is devided
% by its mean if decision = 2. In the case of decision = 2 the mean is not
% substracted.
%
% Example:
%

if nargin < 2;decision = 1;end

[l, n] = size(X);

if decision == 1
    s = std(X);
    indx = find(s==0);
    if indx
        X(:,indx) = [];
        s = std(X);
        m = (mean(X));
        StndP = X - kron(ones(1,l),m')';
        xStandard = StndP./kron(ones(1,l),s')';
    else
        m = (mean(X));
        StndP = X - kron(ones(1,l),m')';
        xStandard = StndP./kron(ones(1,l),s')';
    end
elseif decision == 2    
    s = std(X);
    indx = find(s==0);
    m = (mean(X));
    if indx
        s(indx) = max(m(indx));
        m(indx) = 0;
    end
    StndP = X - kron(ones(1,l),m')';
    xStandard = StndP./kron(ones(1,l),s')';
end

% In Matlab
% y = (x-xmean)*(ystd/xstd) + ymean;
% ymean - Mean value for each row of Y (default is 0)
% ystd - Standard deviation for each row of Y (default is 1)