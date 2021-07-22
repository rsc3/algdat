function model =  fishersFeatureRanking(data)
% This file is for academic purposes. 
%
% Calculates a Fisher/Correlation score for each feature and ranks the 
% features based on the correlation score.
%
% This only works for the two class case
%
% Input:
%    data.X [dim x num_data] training vectors.
%    data.Y [1 x num_data] labels (class) of training data {-1, +1}
%
% Output:
%    model.featureIndex -- ranked feature indices
%    model.featureRankingMethod -- the feature ranking method used
%                                  in this case 'fishersFeatureRanking'
%    model.rankValue -- The value used in ranking the features
%
% Example:
%   X = iris_dataset;
%   iris_Data.X = X(:,1:100)';
%   iris_Data.Y = [ones(1,50) ones(1,50).*-1]';
%   model =  fishersFeatureRanking(iris_Data)
% Reference : C. Bishop, Neural Networks for Pattern Recognition (1995)

X = data.X; 
Y = data.Y;
if max(Y) == 2
    Y(Y==2) = -1;
end
[numData,dim] = size(X);
rank = [];

corr = zeros(dim,1);
corr = (mean(X(Y==1,:)) - mean(X(Y==-1,:))).^2;
s   = (std(X(Y==1,:)).^2) + (std(X(Y==-1,:)).^2);
indx = find(s==0); 
s(indx) = 10000;
corr = corr./s;

% features ranked based on the best correlation scores
indx = find(abs(corr)>10000);
corr(indx) = 0;
[values rankIndx] = sort(-abs(corr));

model.featureIndex = rankIndx;
model.rankValue = values;
model.featureRankingMethod = 'fishersFeatureRanking';