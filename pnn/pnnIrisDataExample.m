% This code is for educational and research purposes of comparisons. This
% is a Probabilistic Neural Network with four layers on a three class 
% iris data set.
%
% This code is converted from Jupyter Notebook implemented by
% Manan Ahuja student in Algorithms for Data Science Spring 2021. The
% source of the algorithm is from Donald F. Specht, Probabalistic Neural
% Networks for Classification, Mapping, or Associative Memory, 1988

clear;
clc;
close all;

irisData = readmatrix('iris.csv','Range','A2:D151');
X = irisData;
y = [ones(1,50) ones(1,50)*2 ones(1,50)*3]';


% This normalizes the data to unit length 0 to 1 by observation as noted by
% Specht. 
for i=1:150
    % x(i,:) = irsData.X(i,3:4)/sqrt(irsData.X(i,3:4)*irsData.X(i,3:4)');
    x(i,:) = X(i,:)/sqrt(X(i,:)*X(i,:)');
end
w = x;
w1 = w(1:50,:);
w2 = w(51:100,:);
w3 = w(101:150,:);

temp = zeros(1,3);
% sigma = 0.0001; % spread use with the petal length and petal width
sigma = 0.5; % spread (smoothing parameter) used for use with all features
ypred = [];

m1 = length(w1); % The number of weight vectors for w1
m2 = length(w2); % The number of weight vectors for w2
m3 = length(w3); % The number of weight vectors for w3

for i=1:150 % All data to determine which class the observations belong to
    sum1 = 0;
    for j=1:m1 % Setosa Class
        z1 = w1(j,:)*x(i,:)';
        sum1 = sum1 + exp((z1-1)/(sigma^2));
    end
    temp(i,1) = sum1/m1;
    sum2 = 0;
    for j=1:m2 % Versicoloe class
        z2 = w2(j,:)*x(i,:)';
        sum2 = sum2 + exp((z2-1)/(sigma^2));
    end
    temp(i,2) = sum2/m2;
    sum3 = 0;
    for j=1:m3 % Virginica class
        z3 = w3(j,:)*x(i,:)';
        sum3 = sum3 + exp((z3-1)/(sigma^2));
    end
    temp(i,3) = sum3/m3;
    [value(i,:) ypred(i,:)] = max(temp(i,:)); % the maximum value is 
                                              % selected to assign the 
                                              % class for the observation.
end

accuracy = (length(find(ypred == y))/150)*100;