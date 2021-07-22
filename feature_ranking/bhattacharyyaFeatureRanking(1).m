function model =  bhattacharyyaFeatureRanking(data)
% This file is for academic purposes. 
%
% bhattacharyyaFeatureRanking.m
% 
% Input:
%    data.X [dim x num_data] training vectors.
%    data.Y [1 x num_data] labels (class) of training data {-1, +1}
%
% Output:
%    model.featureIndex -- ranked feature indices
%    model.featureRankingMethod -- the feature ranking method used
%                                in this case 'bhattacharyyaFeatureRanking'
%    model.rankValue -- The value used in ranking the features
%
% Example:
%   X = iris_dataset;
%   iris_Data.X = X(:,1:100)';
%   iris_Data.Y = [ones(1,50) ones(1,50).*-1]';
%   model =  bhattacharyyaFeatureRanking(iris_Data)
%
% Reference : 
%     Fukunaga, K. (1990). Introduction to Statistical Pattern Recognition,
%     Academic Press, Inc., New Yourk, NY, pp. 267-281

X = data.X;
Y = data.Y;

m1 = mean(X(Y==-1,:));
m2 = mean(X(Y==1,:));

var1 = cov(X(Y==-1,:));
var2 = cov(X(Y==1,:));
B1 = (1/8).*(m1-m2).*diag((inv((var1+var2)./2)))'.*(m1-m2);
B2 = (1/2).*log((diag(((var1+var2)./2))')./sqrt(abs(diag(var1)').*abs(diag(var2)')));

B = B1 + B2;



[values rankIndx] = sort(abs(B),'descend');

model.featureIndex = rankIndx;
model.rankValue = values;
model.featureRankingMethod = 'bhattacharyyaFeatureRanking';