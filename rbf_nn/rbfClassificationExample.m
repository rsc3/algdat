% This code is for educational and research purposes of comparisons. This
% is a RBF Neural Network with four layers on a three class 
% iris data set.

clear;
clc;
close all;

irisData = readmatrix('iris.csv','Range','A2:D151');
irsData.X = irisData;

spread = 0.21;

X=[0 0; 0 1.25; 1 0; 1 1.25; 1 .75; 1 2; 2 0.75; 2 2]';
y = [1 1 1 1 -1 -1 -1 -1]';

irsData.Y = [ones(1,50) ones(1,50)*(-1) ones(1,50)*(-1)]';
model_1 = rbfTrain(irsData.X, irsData.Y, spread);
irsData.Y = [ones(1,50)*(-1) ones(1,50) ones(1,50)*(-1)]';
model_2 = rbfTrain(irsData.X, irsData.Y, spread);
irsData.Y = [ones(1,50)*(-1) ones(1,50)*(-1) ones(1,50)]';
model_3 = rbfTrain(irsData.X, irsData.Y, spread); 

x0 = [5.1,3.5,1.4,0.2];
[yt0 ypred0] = rbfClassify(x0, model_1);

[yt1 ypred1] = rbfClassify(irsData.X, model_1);
[yt2 ypred2] = rbfClassify(irsData.X, model_2);
[yt3 ypred3] = rbfClassify(irsData.X, model_3);

tmp = [yt1;yt2;yt3];
[value ypred] = max(tmp);
irsData.Y = [ones(1,50) ones(1,50)*2 ones(1,50)*3]';
accuracy = (length(find(ypred' == irsData.Y))/150)*100;

% The following ax and ay variables test the RBF NN with the Iris data to
% determine the boundaries for the classes

irsData.Y = [ones(1,50) ones(1,50)*(-1) ones(1,50)*(-1)]';
model_1 = rbfTrain(irsData.X(:,3:4), irsData.Y, spread);
irsData.Y = [ones(1,50)*(-1) ones(1,50) ones(1,50)*(-1)]';
model_2 = rbfTrain(irsData.X(:,3:4), irsData.Y, spread);
irsData.Y = [ones(1,50)*(-1) ones(1,50)*(-1) ones(1,50)]';
model_3 = rbfTrain(irsData.X(:,3:4), irsData.Y, spread);

[Ay,Ax] = meshgrid(linspace(-1,3,101), linspace(0,7,101));
Ax = Ax(:)';
Ay = Ay(:)';
Axy = [Ax; Ay]';

[yt1 ypred1] = rbfClassify(Axy, model_1);
[yt2 ypred2] = rbfClassify(Axy, model_2);
[yt3 ypred3] = rbfClassify(Axy, model_3);

tmp = [yt1;yt2;yt3];
[value ypred] = max(tmp);

indx1 = find(ypred==1);
indx2 = find(ypred==2);
indx3 = find(ypred==3);
figure,plot(Axy(indx1,1),Axy(indx1,2),'.','Color',...
                   [249/255 219/255 219/255],'LineWidth',6,'MarkerSize',20)
hold on;plot(Axy(indx2,1),Axy(indx2,2),'.','Color',...
                   [219/255 249/255 219/255],'LineWidth',6,'MarkerSize',20)
hold on;plot(Axy(indx3,1),Axy(indx3,2),'.','Color',...
                         [204/255 204/255 1],'LineWidth',6,'MarkerSize',20)
hold on;plot(irsData.X(1:50,3),irsData.X(1:50,4),'ro','LineWidth',2,'MarkerSize',8)
hold on;plot(irsData.X(51:100,3),irsData.X(51:100,4),'go','LineWidth',2,'MarkerSize',7)
hold on;plot(irsData.X(101:150,3),irsData.X(101:150,4),'bo','LineWidth',2,'MarkerSize',7)
hold on;contour(reshape(Axy(:,1),101,101), reshape(Axy(:,2),101,101),...
                   reshape(ypred,101,101),'LineColor','k','LineWidth',1.5);
% hold on;contour(reshape(Axy(:,1),101,101), reshape(Axy(:,2),101,101),...
%                reshape(ypred1,101,101),'LineColor','r','LineWidth',1.5);
% hold on;contour(reshape(Axy(:,1),101,101), reshape(Axy(:,2),101,101),...
%                reshape(ypred2,101,101),'LineColor','g','LineWidth',1.5);
% hold on;contour(reshape(Axy(:,1),101,101), reshape(Axy(:,2),101,101),...
%                reshape(ypred3,101,101),'LineColor','b','LineWidth',1.5);
title('RBF Neural Network Example using the Iris Data Set')
xlabel('Normalized Petal Width')
ylabel('Normalized Petal Length')