% This is a two and three class example using the iris data set.

clear;
clc;
close all;

numericalData = readmatrix('trainFeatures.xls');
numData.X = numericalData(:,2:end); %[dim x num_data] training vectors.
numData.Y = numericalData(:,1); %[1 x num_data] labels (class) of training data {1, 2, ...}

irisData = readmatrix('iris.csv','Range','A2:D151');
irsData.X = irisData;
irsData.Y = [ones(1,50) ones(1,50)*2 ones(1,50)*3]';

modelMultiClassNum =  fishersMultiClassFeatureRanking(numData,1)
modelMultiClassIrs =  fishersMultiClassFeatureRanking(irsData,1)

Data.X = irsData.X(1:100,:);
Data.Y = [ones(1,50)*-1 ones(1,50)];
model2Class =  fishersFeatureRanking(Data)

Data.X = irsData.X(1:100,:);
Data.Y = [ones(1,50)*-1 ones(1,50)];
model2Class =  bhattacharyyaFeatureRanking(Data)