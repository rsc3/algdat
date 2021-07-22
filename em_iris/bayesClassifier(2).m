function [y, yPosterior] = bayesClassifier( X, model )
% This code has been modified from its original content for educational
% and research purposes of comparisons. Full credit is given to the  
% original referenced developer. 
%
% This function implements the Bayesian classifier with reject option. The
% input observations X are classified into classes with the highest 
% a posterior probabilities computed from the Gaussian Mixture Model.
%   
% Input:
%  X [dim x numData]
%
%  model - is a struct with the estimated probabilistic values
%       .Pclass [1 x classes] cell with class conditional probabilities.
%       .Prior [1 x classes] a priory probabilities.
%       .eps [1x1] is an optional decision penalty for unknow classes 
%
% Output:
%    y [1 x numData] class labels in the range of [1  classes] and 0 for
%                    unknown class label
%    yPosterior [classes x numData] unnormalized a posterior 
%
% Reference:
%  
%  V. Franc, Optimization Algorithms for Kernal Methods, (2005)
%  V. Franc, Statistical Pattern Recognition Toolbox, (2007)
%  https://cmp.felk.cvut.cz/cmp/software/stprtool/

[dim,numData]=size(X);
classes = length( model.Pclass );

yPosterior=zeros(classes,numData);

for i=1:classes
    y = model.Pclass{i}.Prior(:)'*...
        multivariateGaussianDistribution(X,...
        model.Pclass{i}.Mean,model.Pclass{i}.Cov);
    yPosterior(i,:) = model.Prior(i)*y;
end

[tmp,y] = max(yPosterior);

if isfield(model, 'eps')
  perror = 1-tmp./sum(yPosterior,1);
  inx = find( perror > model.eps);
  y(inx) = 0;
end