% This code is for educational and research purposes of comparisons. This
% is code that provides the use of the z-score normalization 
% (standarization).

clear;
clc;
close all;

numericalData = readmatrix('trainFeatures.xls');
classLabels = numericalData(:,1);
irisData = readmatrix('iris.csv','Range','A2:D151');

wilksMultivariateOutlier(irisData(1:50,:),0.5)
wilksMultivariateOutlier(irisData(51:100,:),0.5)
wilksMultivariateOutlier(irisData(101:150,:),0.5)

indx0 = find(classLabels == 0);
indx1 = find(classLabels == 1);
indx2 = find(classLabels == 2);
indx3 = find(classLabels == 3);
indx4 = find(classLabels == 4);
indx5 = find(classLabels == 5);
indx6 = find(classLabels == 6);
indx7 = find(classLabels == 7);
indx8 = find(classLabels == 8);
indx9 = find(classLabels == 9);

wilksMultivariateOutlier(numericalData(indx0,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx1,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx2,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx3,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx4,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx5,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx6,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx7,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx8,2:35),0.05)
wilksMultivariateOutlier(numericalData(indx9,2:35),0.05)

setosaMean = mean(irisData(1:50,:));
setosaCov = cov(irisData(1:50,:));
setosaDist = mahalan(irisData(1:50,:)',setosaMean',setosaCov);
[setosaDistSorted,SetosaIndx] = sort(setosaDist,'descend');

versicolorMean = mean(irisData(51:100,:));
versicolorCov = cov(irisData(51:100,:));
versicolorDist = mahalan(irisData(51:100,:)',versicolorMean',versicolorCov);
[versicolorDistSorted,versicolorIndx] = sort(versicolorDist,'descend');

virginicaMean = mean(irisData(101:150,:));
virginicaCov = cov(irisData(101:150,:));
virginicaDist = mahalan(irisData(101:150,:)',virginicaMean',virginicaCov);
[virginicaDistSorted,virginicaIndx] = sort(virginicaDist,'descend');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-D example
fprintf('--------------------------------------------------------------\n')
fprintf('2-D Example using feature 1 and 4 only\n')
fprintf('--------------------------------------------------------------\n')

wilksMultivariateOutlier(irisData(1:50,[1 4]),0.5)
wilksMultivariateOutlier(irisData(51:100,[1 4]),0.6)
wilksMultivariateOutlier(irisData(101:150,[1 4]),0.99)

setosaMean = mean(irisData(1:50,[1 4]));
setosaCov = cov(irisData(1:50,[1 4]));
setosaDist = mahalan(irisData(1:50,[1 4])',setosaMean',setosaCov);
[setosaDistSorted,setosaIndx] = sort(setosaDist,'descend');

versicolorMean = mean(irisData(51:100,[1 4]));
versicolorCov = cov(irisData(51:100,[1 4]));
versicolorDist = mahalan(irisData(51:100,[1 4])',versicolorMean',versicolorCov);
[versicolorDistSorted,versicolorIndx] = sort(versicolorDist,'descend');

virginicaMean = mean(irisData(101:150,[1 4]));
virginicaCov = cov(irisData(101:150,[1 4]));
virginicaDist = mahalan(irisData(101:150,[1 4])',virginicaMean',virginicaCov);
[virginicaDistSorted,virginicaIndx] = sort(virginicaDist,'descend');

ID.X = irisData(1:50,[1 4])';
ID.y = ones(50,1)';
ppatterns(ID);
model = mlcgmm(ID); % ML estimate of GMM
pgauss(model);
pgmm(model,struct('visual','contour'));
p = findobj(gcf,'Type','line');
set(p,'LineWidth',3);

ID.X = irisData(51:100,[1 4])';
ID.y = ones(50,1)*2';
ppatterns(ID);
model = mlcgmm(ID); % ML estimate of GMM
pgauss(model);
pgmm(model,struct('visual','contour'));
p = findobj(gcf,'Type','line');
set(p,'LineWidth',3);

ID.X = irisData(101:150,[1 4])';
ID.y = ones(50,1)*3';
ppatterns(ID);
model = mlcgmm(ID); % ML estimate of GMM
pgauss(model);
pgmm(model,struct('visual','contour'));
p = findobj(gcf,'Type','line');
set(p,'LineWidth',3);

hold on;
plot(irisData(setosaIndx(1),1),irisData(setosaIndx(1),4),'ok','LineWidth',2,'MarkerSize',12)
plot(irisData(versicolorIndx(1)+50,1),irisData(versicolorIndx(1)+50,4),'dr','LineWidth',2)
plot(irisData(versicolorIndx(1)+50,1),irisData(versicolorIndx(1)+50,4),'ok','LineWidth',2,'MarkerSize',12)
plot(irisData(virginicaIndx(1)+100,1),irisData(virginicaIndx(1)+100,4),'ok','LineWidth',2,'MarkerSize',12)
axis([4 8 0 2.7])