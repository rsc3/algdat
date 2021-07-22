% This code is for educational and research purposes of comparisons. This
% code will show the process to generate synthetic data.
clear
clc
close all

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(1,:) = weka_data.sepallength.values;
iris_data.X(2,:) = weka_data.sepalwidth.values;
iris_data.X(3,:) = weka_data.petallength.values;
iris_data.X(4,:) = weka_data.petalwidth.values;
iris_data.y = [ones(1,50) ones(1,50).*2 ones(1,50).*3];

iris_class_1 = cov(iris_data.X(:,1:50)');
%rng(5),rnd_data = rand(4,100); % uses rng() for a seed to genrate the same
%random values
rnd_data = rand(4,100); 
rnd_data = iris_class_1*rnd_data;

iris_synthetic = [];

sepal_length_min = min(iris_data.X(1,1:50));
sepal_length_max = max(iris_data.X(1,1:50));

X = (rnd_data(1,:))';
[l, n] = size(X);
Pmin = min(rnd_data(1,:));
Pmax = max(rnd_data(1,:));
a = sepal_length_min; 
b = sepal_length_max;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(1,:) = X';
          
sepal_width_min = min(iris_data.X(2,1:50));
sepal_width_max = max(iris_data.X(2,1:50));

X = (rnd_data(2,:))';
[l, n] = size(X);
Pmin = min(rnd_data(2,:));
Pmax = max(rnd_data(2,:));
a = sepal_width_min; 
b = sepal_width_max;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(2,:) = X';

petal_length_min = min(iris_data.X(3,1:50));
petal_length_max = max(iris_data.X(3,1:50));

X = (rnd_data(3,:))';
[l, n] = size(X);
Pmin = min(rnd_data(3,:));
Pmax = max(rnd_data(3,:));
a = petal_length_min; 
b = petal_length_max;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(3,:) = X';

petal_width_min = min(iris_data.X(4,1:50));
petal_width_max = max(iris_data.X(4,1:50));

X = (rnd_data(4,:))';
[l, n] = size(X);
Pmin = min(rnd_data(4,:));
Pmax = max(rnd_data(4,:));
a = petal_width_min; 
b = petal_width_max;
X = ((X - Pmin)./(Pmax - Pmin)).*(b-a)+ a;
iris_synthetic(4,:) = X';

mu = (mean(iris_synthetic')-mean(iris_data.X(:,1:50)'));
mu_100 = [ones(1,100)*mu(1); ones(1,100)*mu(2); ones(1,100)*mu(3); ones(1,100)*mu(4)];
tmp = iris_synthetic - mu_100;


figure(1),plot(iris_data.X(1,1:50),iris_data.X(4,1:50),'or')
hold on;
plot(tmp(1,:),tmp(4,:),'xb')
p = findobj(gcf,'Type','line');
set(p,'LineWidth',3);
axis([4 6 0 0.65])
hold off;

figure(2); hold on;
ID.X = tmp([1 4],:);
ID.y = ones(1,100);
ppatterns(ID);
model = mlcgmm(ID); % ML estimate of GMM
pgauss(model);
pgmm(model,struct('visual','contour'));

ID.X = iris_data.X([1 4],1:50);
ID.y = iris_data.y(51:100);
ppatterns(ID);
model = mlcgmm(ID); % ML estimate of GMM
pgauss(model);
pgmm(model,struct('visual','contour'));
p = findobj(gcf,'Type','line');
set(p,'LineWidth',3);
axis([4 6 0 0.65])