function K = gaussianKernel(X1, X2, spread)
% This code is for educational and research purposes of comparisons. 
% Credit will be given to the referenced developer and Emanuel Parzen.
%
% This function evaluates the Gaussian on the input data and 
% returns the kernal values. In (Parzen,1962) the kernal function are 
% defined as the weighting functions.   
% 
% Reference:
%    [1] Parzen, E., On the Estimation of a Probability Density Function 
%        and Mode, 1962
%    [2] Duin, R.P.W and Pekalska, E., Pattern Recognition Tools, 
%        http://37steps.com/37-steps/

[row1,col1]=size(X1);
[row2,col2]=size(X2);
N = col2;
D = row2;
K = zeros(col1,col2);
for i = 1:col1
    for j = 1:col2
        % The ... in Matlab is a continuation of the code
        K(i,j) = ...
            (1/N)*(1/((sqrt(2*pi)*spread))^D)*...
            exp(-0.5*(((X1(:,i)-X2(:,j))'*(X1(:,i)-X2(:,j)))/(spread^2)));
    end
end