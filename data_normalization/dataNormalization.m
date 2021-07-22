function [xStandard, indx] = dataNormalization(X,normType,decision)
% This code is for educational and research purposes of comparisons. This
% is code that provides the use of 4 data normalization methods. 
%
% Input:
%   X [l x n] - l is the number of exemplars/samples/observation
%             - n is the dimension of the data
%   normType [1 x 1] - determines the type of normalization method used
%            normType = 1 - Global Normalization
%                              X                                                   
%                       N = --------                                               
%                            ||X||                                                 
%            normType = 2 - Local Normalization of the data.                   
%                           The data undergoes a line by line normalization
%                           with the Euclidean norm of the line. If it is 
%                           desired to center the data, subtract the 
%                           features mean before normalizing the features.
%                           By default the data should be centered.
%                            Xi - mean(Xi)                                       
%                       N = --------------                             
%                              ||Xi||                                        
%            
%            normType = 3 - Local Normalization of the data.
%                                                                   
%                                                    Xi                                          
%                                          N = --------------                                   
%                                                  ||Xi||
%
%            normType = 4 - Normalization of the data between [a, b].
%                           This method is the default and a = -1, b = 1
%                             X - min(X)                                         
%            (default)  N = --------------- * (b-a)+a                             
%                            max(X)-min(X)   
%
%   decision [1 x 1] - determines what to do in the event a standard
%                     deviation is equal to zero
%            decision = 1 - removes the features whos std dev = 0
%            decision = 2 - gives a fudge factor of meadian(s)
%            Default, decision = 1
%
% Output:
%   xNorm [l x n] - normalized data according to:                      
%   indx [1 x m] - where m is the number of features which have a std dev
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

if nargin < 2;normType = 4;decision = 1;end
if nargin < 3;decision = 1;end

[l, n] = size(X);
indx = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Global Normalization of the data.                              %
%          Normalize the columns of Training data                         %
%                     X                                                   %
%              N = --------                                               %
%                   ||X||                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if normType == 1
     NP = sqrt(diag(X'*X));
     xNorm = X./(ones(size(X,1),1)*NP');    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Local Normalization of the data.                               %
%                                                                         %
%                   Xi - mean(Xi)                                         %
%              N = --------------                                         %
%                     ||Xi||                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif normType == 2
    m = (mean(X)); 
    X = X - kron(ones(1,size(X,1)),m')';
    NP = sqrt(sum(X.^2,2));
    NP(find(NP == 0)) = 1;
    X = X./NP(:,ones(1,size(X,1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Local Normalization of the data.                               %
%                                                                         %
%                       Xi                                                %
%              N = --------------                                         %
%                     ||Xi||                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif normType == 3
    m = (mean(X)); 
    
    NP = sqrt(sum(X.^2,2));
    NP(find(NP == 0)) = 1;
    X = X./NP(:,ones(1,size(X,1)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Normalization of the data between [a, b].                      %
%                                                                         %
%                      X - min(X)                                         %
%               N = --------------- * (b-a) + a                           %
%                    max(X)-min(X)                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif normType == 4
    Pmin = min(X);
    Pmax = max(X);
    a=-1; b=1;
    minMax = Pmax - Pmin;
    indx = find(minMax==0);
    if indx% Ensures that a division by zero does not occur
        Pmin(indx) = [];
        Pmax(indx) = [];
        X(:,indx) = [];
    end
    X = ((X - kron(ones(1,l),Pmin')')./...
        (kron(ones(1,l),Pmax')' - kron(ones(1,l),Pmin')')).*(b-a)+ a;
end
xStandard = X;