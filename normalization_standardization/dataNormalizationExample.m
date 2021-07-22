% This code is for educational and research purposes of comparisons. This
% is code that provides the use of the z-score normalization 
% (standarization).

clear;
clc;
close all;

numericalData = readmatrix('trainFeatures.xls');
irisData = readmatrix('iris.csv','Range','A2:D151');

[numericalDataNormalization, indx] = dataNormalization(numericalData(:,2:35));
[numericalDataStandard, indx] = dataStandard(numericalData(:,2:35));

[irisDataNormalization, indx] = dataNormalization(irisData);
[irisDataStandard, indx] = dataStandard(irisData);