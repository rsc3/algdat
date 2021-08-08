% Ben Rodriguez 
% EN.685.621
% This code is for eduational purposes
% Simple example of Expectation Maximization using the Iris data set

clear;
clc;
close all;

% weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
% iris_data.X(:,1) = weka_data.sepallength.values';
% iris_data.X(:,2) = weka_data.sepalwidth.values';
% iris_data.X(:,3) = weka_data.petallength.values';
% iris_data.X(:,4) = weka_data.petalwidth.values';
% iris_data.Y = [ones(1,50) ones(1,50).*2 ones(1,50).*3]';
irisData = readmatrix('iris.csv','Range','A2:D151');
y = [ones(1,50) ones(1,50).*2 ones(1,50).*3];
%              setosa = 1, versicolor = 2, viginica = 3
%              red         green           blue

iris_data.X = irisData;
iris_data.Y = y;

K = 3; % number of clusters
m = mean(iris_data.X)';
sigma = std(iris_data.X)';
initializeK = ones(1, K);
%rnd = randn(1, K);
rnd =[0.5377    1.8339   -2.2588];
m = m * initializeK + sigma * rnd; % initial mean for EM
sigma = mean(sigma) * initializeK; % initial standard deviation for EM
prob = initializeK / K; % intial probability mixture 
convergence_threshold = sigma(1) * 1.0e-6;

[pUpdate, mUpdate, sigmaUpdate,  prob_ikn, numberIterations] = ...
   simpleExpectationMaximization(iris_data.X(:,:)', K, prob, m(:,:), sigma, convergence_threshold);

[tmp,  classEM] = max( prob_ikn); % classEM gives

class2 = find(classEM == 1);  % versicolor
class3 = find(classEM == 2);  % virginica
class1 = find(classEM == 3);  % setosa

figure,plot(iris_data.X(class1,3), iris_data.X(class1,4),'xb',...
        iris_data.X(class2,3), iris_data.X(class2,4),'dr',...
        iris_data.X(class3,3), iris_data.X(class3,4),'og')
    title('Iris data reassgined with EM algorithm')
    

figure,plot(iris_data.X(1:50,3), iris_data.X(1:50,4),'xb',...
        iris_data.X(51:100,3), iris_data.X(51:100,4),'dr',...
        iris_data.X(101:150,3), iris_data.X(101:150,4),'og')
    title('Original Iris data')
    
figure,plot(iris_data.X(class1,3), iris_data.X(class1,4),'ob','LineWidth',2,'MarkerSize',12)
hold on;plot(iris_data.X(class2,3), iris_data.X(class2,4),'or','LineWidth',2,'MarkerSize',12)
hold on;plot(iris_data.X(class3,3), iris_data.X(class3,4),'og','LineWidth',2,'MarkerSize',12)

hold on;plot(iris_data.X(1:50,3), iris_data.X(1:50,4),'xb','LineWidth',2)
hold on;plot(iris_data.X(51:100,3), iris_data.X(51:100,4),'dr','LineWidth',2)
hold on;plot(iris_data.X(101:150,3), iris_data.X(101:150,4),'og','LineWidth',2)
    title('Original Iris data vs EM Algorithm results')
 
% Below we are using the origninal Iris data to calculate the mean, cov, 
% and prior probabilities      
 model.Prior = [(50/150) (50/150) (50/150)];   
 model.Pclass{1}.Prior = 1;
 model.Pclass{2}.Prior = 1;
 model.Pclass{3}.Prior = 1;
 model.Pclass{1}.Mean = mean(iris_data.X(1:50,:))';
 model.Pclass{2}.Mean = mean(iris_data.X(51:100,:))';
 model.Pclass{3}.Mean = mean(iris_data.X(101:150,:))';
 model.Pclass{1}.Cov = cov(iris_data.X(1:50,:));
 model.Pclass{2}.Cov = cov(iris_data.X(51:100,:));
 model.Pclass{3}.Cov = cov(iris_data.X(101:150,:));
 ypred = bayesClassifier(iris_data.X',model);
 CA_OriginalData = length(find(iris_data.Y==ypred))/length(iris_data.Y);
 
 % Below we are using the newly assigned class labels of the Iris data 
 % from the EM algoerithm to calculate the mean, cov, 
 % and prior probabilities      
 model.Prior = [(length(class1)/150) (length(class2)/150) (length(class3)/150)];   
 model.Pclass{1}.Prior = 1;
 model.Pclass{2}.Prior = 1;
 model.Pclass{3}.Prior = 1;
 model.Pclass{1}.Mean = mean(iris_data.X(class1,:))';
 model.Pclass{2}.Mean = mean(iris_data.X(class2,:))';
 model.Pclass{3}.Mean = mean(iris_data.X(class3,:))';
 model.Pclass{1}.Cov = cov(iris_data.X(class1,:));
 model.Pclass{2}.Cov = cov(iris_data.X(class2,:));
 model.Pclass{3}.Cov = cov(iris_data.X(class3,:));
 ypred = bayesClassifier(iris_data.X',model);
 CA_EMAssigned = length(find(iris_data.Y==ypred))/length(iris_data.Y);