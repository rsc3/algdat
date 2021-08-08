% This code is for educational and research purposes of comparisons. This
% is a FFNN three class example using the iris data set.

clc
clear
close all

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(:,1) = weka_data.sepallength.values';
iris_data.X(:,2) = weka_data.sepalwidth.values';
iris_data.X(:,3) = weka_data.petallength.values'; % Petal Length
iris_data.X(:,4) = weka_data.petalwidth.values';  % Petal Width
iris_data.Y = [ones(1,50) ones(1,50).*2 ones(1,50).*3];
%              setosa = 1, versicolor = 2, virginica = 3
%              red         green           blue

X = iris_data.X'; % creates a [4x150] matrix
Y = iris_data.Y;  % creates a [1x150] vector

[model, L] = mlpClass(X, Y, 4); % can also use the mlpClass(X, Y, [4 2]) 
                                % for two hidden layers with the first 
                                % layer having 4 nodes and the second 
                                % having 2, the output would hstill have 
                                % three for the three class decision.
[y_hat, P] = mlpClassPred(model, X);
CA = length(find(y_hat == Y))/150;

net = feedforwardnet(4); % The number of hidden layers assigned are 4
net = train(net,X,Y);

yhat = round(sim(net,X));
% The above is equivalent to the following
X_normalized = mapminmax('apply',X,net.inputs{1}.processSettings{1}); 
% The above is normalized from -1 to 1
Y_hat = purelin(net.b{2} + net.LW{2,1} * tansig(net.b{1} + (net.IW{1,1} * X_normalized)));
% The above produces the output of the network
Y_hat = mapminmax('reverse',Y_hat,net.outputs{2}.processSettings{1});
% The above reverses the normalization

X_hat = mapminmax('apply',X,net.inputs{1}.processSettings{1});
X_wieghted = (net.b{1} + (net.IW{1,1} * X_hat)); % if the full 
X_mapped = mapminmax('reverse',X_wieghted,net.outputs{2}.processSettings{1});

Data.X = X_mapped';
Data.Y = Y';
model =  fishersMultiClassFeatureRanking(Data,1);
numFeatures = model.featureIndex(1:2);

figure;plot(X(4,1:50),X(3,1:50),'x','Color',[0 0 1],'LineWidth',2,'MarkerSize',8)
hold on;plot(X(4,51:100),X(3,51:100),'ro','LineWidth',2,'MarkerSize',7)
hold on;plot(X(4,101:150),X(3,101:150),'g^','LineWidth',2,'MarkerSize',7)

figure;plot(X_mapped(numFeatures(1),1:50),X_mapped(numFeatures(2),1:50),'x','Color',[0 0 1],'LineWidth',2,'MarkerSize',8)
hold on;plot(X_mapped(numFeatures(1),51:100),X_mapped(numFeatures(2),51:100),'ro','LineWidth',2,'MarkerSize',7)
hold on;plot(X_mapped(numFeatures(1),101:150),X_mapped(numFeatures(2),101:150),'g^','LineWidth',2,'MarkerSize',7)