function mahalanobisDistance = mahalanobisDist(data,Mean,Covariance)
% This code has been modified from its original content for educational
% and research purposes of comparisons. Full credit is given to the  
% original referenced developer. 
%
% This function computes the Mahalanobis distance.
%
% Input:
%   data [dim x observations]
%   Mean [dim x 1] 
%   Covariance  [dim x dim]
%
% Output:
%   mahalanobisDistance [1 x observations]
%  
% Reference:
%  
%  V. Franc, Optimization Algorithms for Kernal Methods, (2005)
%  V. Franc, Statistical Pattern Recognition Toolbox, (2007)
%  https://cmp.felk.cvut.cz/cmp/software/stprtool/

[dim, observations] = size( data );

dataC = data - repmat(Mean,1,observations);
mahalanobisDistance= sum((dataC'*inv( Covariance ).*dataC')',1);