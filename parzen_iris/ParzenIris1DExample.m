% This code is for educational and research purposes of comparisons. This
% is a Parzen three class example using the iris data set.

clear;
clc;
close all;

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(:,1) = weka_data.sepallength.values';
iris_data.X(:,2) = weka_data.sepalwidth.values';
iris_data.X(:,3) = weka_data.petallength.values'; % Petal Length
iris_data.X(:,4) = weka_data.petalwidth.values';  % Petal Width
iris_data.Y = [ones(1,50) ones(1,50).*2 ones(1,50).*3]';
%              setosa = 1, versicolor = 2, virginica = 3
%              red         green           blue

figure,plot(iris_data.X(1:50,3),zeros(1,50),'ro','MarkerSize',8,'LineWidth',1.5)
hold on;plot(mean(iris_data.X(1:50,3)),0,'rx','MarkerSize',10,'LineWidth',2)
hold on;plot(iris_data.X(51:100,3),zeros(1,50),'go','MarkerSize',8,'LineWidth',1.5)
hold on;plot(mean(iris_data.X(51:100,3)),0,'gx','MarkerSize',10,'LineWidth',2)
hold on;plot(iris_data.X(101:150,3),zeros(1,50),'bo','MarkerSize',8,'LineWidth',1.5)
hold on;plot(mean(iris_data.X(101:150,3)),0,'bx','MarkerSize',10,'LineWidth',2)

ax = 0:0.05:8; % used to test the space from 0 to 8 in increments of 0.05
spread = 0.5; % this is the h value in the equation
p1=zeros(size(ax));  
for i=1:50,
   p1 = p1 + gaussianKernel(iris_data.X(i,3), ax, spread);
end
p1=p1;
hold on,plot(ax,p1,'r','MarkerSize',8,'LineWidth',1)

p2=zeros(size(ax));
for i=51:100,
   p2 = p2 + gaussianKernel(iris_data.X(i,3), ax, spread);
end
p2=p2;
hold on,plot(ax,p2,'g','LineWidth',1)

p3=zeros(size(ax));
for i=101:150,
   p3 = p3 + gaussianKernel(iris_data.X(i,3), ax, spread);
end
p3=p3;
hold on,plot(ax,p3,'b','LineWidth',1)