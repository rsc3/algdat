% Example 1: Linear Discriminant Analysis
% The LDA is applied to extract 2 features from the Iris data set iris.mat which
% consists of the labeled 4-dimensional data. The data after the feature extraction step
% are visualized in Figure 3.2.

clear
clc
close all

weka_data = arffparser('read', 'iris.arff'); % load data provided by WEKA
iris_data.X(1,:) = weka_data.sepallength.values;
iris_data.X(2,:) = weka_data.sepalwidth.values;
iris_data.X(3,:) = weka_data.petallength.values;
iris_data.X(4,:) = weka_data.petalwidth.values;
iris_data.y = [ones(1,50) ones(1,50).*2 ones(1,50).*3];
%stprtools_data = load('iris'); % load input data provided by STPRTools
%iris_data = stprtools_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply princomp and rotatefactors
% Do not standardize the features when the princomp is used for extraction
% since princomp has its own standardization process within.
%  - princomp.m centers the data, it uses the raw data
%  - pcacov.m does not center the data but requires the covariance (C) or
%    correlation (R) matirx, the C and R matrix are taken from the raw 
%    data. pcacov.m uses svd.m to determine the eigen vectors.
%  - rotatefactors.m default is varimax but equimax or quartimax.
%    -> varimax reduces the error by column (factors)
%    -> quartimax reduces the error by row (features)
%    -> equamax reduces the error by row and column simultaneously
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The following uses the correlation matrix to determine how many
    % principal components to keep
    R = corrcoef(iris_data.X');
    [R_eigVec, R_eigVal, R_explained] = pcacov(R);
    R_eigVal_diag = diag(R_eigVal);
    R_CumulativeSum = cumsum(R_explained);
    indx = []; %indx = find(R_eigVal >= 0.1);
    indx = 1:3;
    % The following uses the covariance matrix to reduce the dimensionality
    % of the data. Note, when using the covariance instead of the
    % correlation matrix allows the new data to have the same variance of
    % the calculated eigen values. Also, when calculatig the covariance the
    % data is zero meaned. 
    C = cov(iris_data.X');
    Dinv = sqrt(inv(diag(diag(C))));
    [C_eigVec, C_eigVal, C_explained] = pcacov(C);
    % Here we use the top three principal components
    X_pcacov = iris_data.X'*C_eigVec; 
    figure,plot(X_pcacov(1:50,1), X_pcacov(1:50,2),'xb',...
        X_pcacov(51:100,1), X_pcacov(51:100,2),'dr',...
        X_pcacov(101:150,1), X_pcacov(101:150,2),'*g')
    title('Iris data reduced to 2-dimensions using Matlab pcacov()')
    % Notice the results of the transforming to the new space is the same
    % when plotting the first two new vectors regardless of using all the
    % eigen vectors or the first two. 
    X_pcacov = iris_data.X'*C_eigVec(:,1:2); 
    figure,plot(X_pcacov(1:50,1), X_pcacov(1:50,2),'xb',...
        X_pcacov(51:100,1), X_pcacov(51:100,2),'dr',...
        X_pcacov(101:150,1), X_pcacov(101:150,2),'*g')
    title('Iris data reduced to 2-dimensions using Matlab pcacov()')
    
    [V, E] = eig(C);
    % After evaluating the eigen values in E it is seen that diagnal values
    % are in assending order so flip the corresponsing eigne vectors to be
    % the decending order of importance (variance high to low).
    u = fliplr(V); % this simply flips the matrix from left to right
    % Here we use the top three principal components
    X_eig = iris_data.X'*u; 
    figure,plot(X_eig(1:50,1), X_eig(1:50,2),'xb',...
        X_eig(51:100,1), X_eig(51:100,2),'dr',...
        X_eig(101:150,1), X_eig(101:150,2),'*g')
    title('Iris data reduced to 2-dimensions using Matlab eig()')
    
    % If you look at the varance of X_pcacov and X_eig, it is the same as
    % the variance shown in the first three eigenvalues of C_eigVal and E.
    % var(X_pcacov)
    % var(X_eig)
    % C_eigVal
    % E
    
    % The following uses the covariance matrix to find the initial factor 
    % loadings of the covariance matrix C to determine the features to 
    % group.
    C_eigValDiag = diag(C_eigVal);
    C_CumulativeSum = cumsum(C_explained);
    LambdaC = sqrt(C_eigValDiag(indx,indx));
    LoadingsC = Dinv*C_eigVec(:,indx)*LambdaC;
    [RotatedLoadingsC] = rotatefactors(LoadingsC,'Method','varimax');
    Features.X = (iris_data.X'*RotatedLoadingsC)';
    figure,plot(Features.X(1,1:50), Features.X(2,1:50),'xb',...
        Features.X(1,51:100), Features.X(2,51:100),'dr',...
        Features.X(1,101:150), Features.X(2,101:150),'*g')
    title('Iris data reduced to 2-dimensions using Matlab rotatefactors()')
    
    %Features.X = Features.X*LoadingsC;
    % NOTE: When using the ranked features it seems to be unnecessary to
    %       rotate the LoadingsC.

%     % The following is an alternative method
%     [eig_vec,SCORE,eig_val,tsquare] = princomp(Features.X);
%     %explained = eig_val./sum(eig_val);
%     %cumulativeSum = cumsum(explained);
%     %indx = find(cumulativeSum  <= 0.99);
%     indx = find(eig_val >= 1);
%     eigVec = eig_vec(:,indx);
%     [L,T] = rotatefactors(eigVec);
%     Features.X = Features.X*L;
%     % The above is the same mathematical concept
%     % [L,T] = rotatefactors(Features.X*eigVec);
%     % Features.X = L;

% The following kernel pca is from Mathworks developed by Ambarish Jash. 
% This is a great example for learning how to use the kernel trick with PCA 
% https://www.mathworks.com/matlabcentral/fileexchange/27319-kernel-pca
% You will notice that code is commented out where the sigma value of the
% kernel can be changed. By adjusting the value you will notice better
% separation in all three classes.
data_out = kernelpca_tutorial(iris_data.X,2);
figure,plot(data_out(1,1:50), data_out(2,1:50),'xb',...
        data_out(1,51:100), data_out(2,51:100),'dr',...
        data_out(1,101:150), data_out(2,101:150),'*g')
    title('Iris data reduced to 2-dimensions using Matlab kernelpca\_tutorial()')
  
% Note since the data is not separated between the versicolor and virginica
% when processing all three classea only two classes are processed to see 
% how this may potentially improve separation in the new space.   

data_out = kernelpca_tutorial(iris_data.X(:,51:150),2);
figure,plot(data_out(1,1:50), data_out(2,1:50),'xb',...
        data_out(1,51:100), data_out(2,51:100),'dr')
    title('Iris data (versicolor and virginica) reduced to 2-dimensions using Matlab kernelpca\_tutorial()')
