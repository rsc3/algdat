% Ben Rodriguez 
% EN.685.621
% This code is for eduational purposes
% Simple example of expectation maximization

clear
clc
close all
K = 2;
x = [1     2
     4     2
     1     3
     4     3];

m = mean(x)';
sigma = std(x)';
initializeK = ones(1, K);
%rnd = randn(1, K);
rnd = [-0.18671,0.72579]; % Used for example
m = m * initializeK + sigma * rnd;
sigma = mean(sigma) * initializeK;
prob = initializeK / K;
[pUpdate, mUpdate, sigmaUpdate,  prob_ikn, numberIterations] = ...
   simpleExpectationMaximization(x', K, prob, m, sigma, sigma(1) * 1.0e-6);