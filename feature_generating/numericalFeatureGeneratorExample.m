% This code is for educational and research purposes of comparisons. This
% is a numerical reader and processing for generating features. The end
% result will be a 10 class number data set where each observation will
% have d_f features from the 28 x 28 pixel size images, where d_f is not 
% the size of 28x28. 
%
% References:
%    https://www.kaggle.com/c/digit-recognizer/data

clear;
clc;
close all;

train = readmatrix('train.csv');

classLabels = train(:,1);
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

diagMask = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0;
            0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
indxD = find(diagMask == 1);
vertMask = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
indxV = find(vertMask == 1);        
horizMask = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
             1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
indxH = find(horizMask == 1);         

figure,imagesc(diagMask), colormap('gray')
figure,imagesc(vertMask), colormap('gray')
figure,imagesc(horizMask), colormap('gray')
         
dataD = [];
dataV = [];
dataH = [];
for i = 1:1000
    img = reshape(train(i,2:end),[28,28])';
    imgDCT = dct2(img);
%     imgD = imgDCT.*diagMask;
%     imgV = imgDCT.*vertMask;
%     imgH = imgDCT.*horizMask;
%     
%     dataD = [dataD imgD(indxD)'];
%     dataV = [dataV imgV(indxV)'];
%     dataH = [dataH imgH(indxH)'];
    dataD = [dataD; imgDCT(indxD)'];
    dataV = [dataV; imgDCT(indxV)'];
    dataH = [dataH; imgDCT(indxH)'];
end

indx0 = find(classLabels(1:1000) == 0);
indx1 = find(classLabels(1:1000) == 1);
indx2 = find(classLabels(1:1000) == 2);
indx3 = find(classLabels(1:1000) == 3);
indx4 = find(classLabels(1:1000) == 4);
indx5 = find(classLabels(1:1000) == 5);
indx6 = find(classLabels(1:1000) == 6);
indx7 = find(classLabels(1:1000) == 7);
indx8 = find(classLabels(1:1000) == 8);
indx9 = find(classLabels(1:1000) == 9);

% The following uses the correlation matrix to determine how many
% principal components to keep
rD = corrcoef(dataD);
[R_eigVec, R_eigVal, R_explained] = pcacov(rD);
R_eigVal_diag = diag(R_eigVal);
R_CumulativeSum = cumsum(R_explained);
indxD = find(R_eigVal >= 4); % indx = 1:15;

rV = corrcoef(dataV);
[R_eigVec, R_eigVal, R_explained] = pcacov(rV);
R_eigVal_diag = diag(R_eigVal);
R_CumulativeSum = cumsum(R_explained);
indxV = find(R_eigVal >= 4); % indx = 1:10;

rV = corrcoef(dataV);
[R_eigVec, R_eigVal, R_explained] = pcacov(rV);
R_eigVal_diag = diag(R_eigVal);
R_CumulativeSum = cumsum(R_explained);
indxH = find(R_eigVal >= 4); % indx = 1:10;

% The following uses the covariance matrix to reduce the dimensionality
% of the data. Note, when using the covariance instead of the
% correlation matrix allows the new data to have the same variance of
% the calculated eigen values. Also, when calculatig the covariance the
% data is zero meaned. 
CD = cov(dataD);
[C_eigVecD, C_eigVal, C_explained] = pcacov(CD);
CV = cov(dataV);
[C_eigVecV, C_eigVal, C_explained] = pcacov(CV);
CH = cov(dataH);
[C_eigVecH, C_eigVal, C_explained] = pcacov(CH);
% Now we use the top principal components from above. 
pcaFeatures = [];
pcaFeatures = [pcaFeatures dataD*C_eigVecD(:,indxD)]; 
pcaFeatures = [pcaFeatures dataV*C_eigVecV(:,indxV)];
pcaFeatures = [pcaFeatures dataH*C_eigVecH(:,indxH)];
figure,plot(pcaFeatures(indx0,3), pcaFeatures(indx0,16),'or',...
        pcaFeatures(indx1,3), pcaFeatures(indx1,16),'+g',...
        pcaFeatures(indx2,3), pcaFeatures(indx2,16),'*b',...
        pcaFeatures(indx3,3), pcaFeatures(indx3,16),'.k',...
        pcaFeatures(indx4,3), pcaFeatures(indx4,16),'xm',...
        pcaFeatures(indx5,3), pcaFeatures(indx5,16),'sr',...
        pcaFeatures(indx6,3), pcaFeatures(indx6,16),'dg',...
        pcaFeatures(indx7,3), pcaFeatures(indx7,16),'^b',...
        pcaFeatures(indx8,3), pcaFeatures(indx8,16),'pk',...
        pcaFeatures(indx9,3), pcaFeatures(indx9,16),'hm')
legend('0','1','2','3','4','5','6','7','8','9')
axis([-1000 1000 -1000 1000])
title('Numerical data transformed using DCT and reduced to 2-dimensions using Matlab pcacov()')

figure,plot(pcaFeatures(indx0,3), pcaFeatures(indx0,16),'or',...
        pcaFeatures(indx1,3), pcaFeatures(indx1,16),'+g',...
        pcaFeatures(indx3,3), pcaFeatures(indx3,16),'.k',...
        pcaFeatures(indx4,3), pcaFeatures(indx4,16),'xm')
legend('0','1','3','4')
axis([-1000 1000 -1000 1000])
title('Numerical data 0, 1, and 4 transformed using DCT and reduced to 2-dimensions using Matlab pcacov()')

writematrix([classLabels(1:1000) pcaFeatures],'trainFeatures.xls');