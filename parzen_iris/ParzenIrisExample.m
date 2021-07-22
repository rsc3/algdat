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

spread = 0.5;

model =  fishersMultiClassFeatureRanking(iris_data,1);% Rank features
numFeatures = model.featureIndex(1:2);%Select the top two ranked features
X = iris_data.X(:,[4 3])'; % ues the petal length and petal width
%X = iris_data.X(:,numFeatures)'; % use the top two features
%X = iris_data.X'; % use all of the features
y = iris_data.Y';

spread = 0.5; % this is the h value in the equation
p1 = zeros(1,size(X,2)); 
for i=1:50
    p1 = p1 + gaussianKernel(X(:,i), X, spread);
end
p1 = p1/50;

p2=zeros(1,size(X,2)); 
for i=51:100
    p2 = p2 + gaussianKernel(X(:,i), X, spread);
end
p2=p2/50;

p3=zeros(1,size(X,2)); 
for i=101:150
    p3 = p3 + gaussianKernel(X(:,i), X, spread);
end
p3=p3/50;

ytmp = [p1; p2; p3];

[value ypred]= max(ytmp);
inx1 = find(y==1);
inx2 = find(y==2);
inx3 = find(y==3);
CA = length(find(y==ypred))/length(y); % This gives use the classification
                                       % accuracy.

% The following ax and ay variables test the kernel with the Iris data to
% determine the boundaries for the classes
ax=-1:0.04:3;
ay=0:0.07:7;
[Ax,Ay] = meshgrid(linspace(-1,3,101), linspace(0,7,101));
Ax = Ax(:)';
Ay = Ay(:)';

p1=zeros(size(Ax)); 
for i=1:50
    p1 = p1 + gaussianKernel(X(:,i), [Ax; Ay], spread);
end
p1=p1/50;

p2=zeros(size(Ax)); 
for i=51:100
    p2 = p2 + gaussianKernel(X(:,i), [Ax; Ay], spread);
end
p2=p2/50;

p3=zeros(size(Ax)); 
for i=101:150
    p3 = p3 + gaussianKernel(X(:,i), [Ax; Ay], spread);
end
p3=p3/50;

figure;plot(X(1,1:50),X(2,1:50),'or');
hold on;plot(mean(X(1,1:50)),mean(X(2,1:50)),...
                            'xr','MarkerSize',12,'LineWidth',2)
hold on;plot(mean(X(1,1:50)),mean(X(2,1:50)),...
                            'xr','MarkerSize',12,'LineWidth',2)
hold on;plot(X(1,51:100),X(2,51:100),'og')
hold on;plot(mean(X(1,51:100)),mean(X(2,51:100)),...
                          'xg','MarkerSize',12,'LineWidth',2)
hold on;plot(mean(X(1,51:100)),mean(X(2,51:100)),...
                          'xg','MarkerSize',12,'LineWidth',2)
hold on;plot(X(1,101:150),X(2,101:150),'ob')
hold on;plot(mean(X(1,101:150)),mean(X(2,101:150)),...
                         'xb','MarkerSize',12,'LineWidth',2)
hold on;plot(mean(X(1,101:150)),mean(X(2,101:150)),...
                         'xb','MarkerSize',12,'LineWidth',2)
                     
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101),reshape(p1,101,101),'LineColor','r');
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101),reshape(p2,101,101),'LineColor','g');
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101),reshape(p3,101,101),'LineColor','b');

title('Parzen Windowing with Gaussian Kernel')
xlabel('Petal Width')
ylabel('Petal Length')
axis([0 3 0.5 7])

ytmp = [p1; p2; p3];
[value ypred]= max(ytmp);
indx1 = find(ypred==1);
indx2 = find(ypred==2);
indx3 = find(ypred==3);
figure,plot(Ax(indx1),Ay(indx1),'.','Color',[249/255 219/255 219/255],'LineWidth',6,'MarkerSize',20)
hold on;plot(Ax(indx2),Ay(indx2),'.','Color',[219/255 249/255 219/255],'LineWidth',6,'MarkerSize',20)
hold on;plot(Ax(indx3),Ay(indx3),'.','Color',[204/255 204/255 1],'LineWidth',6,'MarkerSize',20)
hold on;plot(X(1,inx1),X(2,inx1),'ro','LineWidth',2,'MarkerSize',8)
hold on;plot(X(1,inx2),X(2,inx2),'go','LineWidth',2,'MarkerSize',7)
hold on;plot(X(1,inx3),X(2,inx3),'bo','LineWidth',2,'MarkerSize',7)
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101),reshape(ypred,101,101),'LineColor','k','LineWidth',1.5);
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101),reshape(p1,101,101),'LineColor','r','LineWidth',1.5);
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101), reshape(p2,101,101),'LineColor','g','LineWidth',1.5);
hold on;contour(reshape(Ax,101,101), reshape(Ay,101,101), reshape(p3,101,101),'LineColor','b','LineWidth',1.5);
title('Parzen Windowing with Gaussian Kernel')
xlabel('Petal Width')
ylabel('Petal Length')
axis([0 3 0.5 7])