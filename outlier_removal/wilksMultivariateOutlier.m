function wilksMultivariateOutlier(X,alpha)
% MOUTLIER1 Detection of Outlier in Multivariate Samples Test.
% This test is based on the Wilks'method (1963) designed for detection of a
% single outlier from a normal multivariate sample and approaching the 
% maximun squared Mahalanobis distance to a F distribution function by the
% Yang and Lee (1987) formulation. A significative squared Mahalanobis 
% distance means an outlier. To test the outlier, this function calls the 
% ACR function.
%
% Syntax: function wilksMultivariateOutlier(X,alpha) 
%      
% Inputs:
%     X - multivariate data matrix. 
%     alpha - significance level (default = 0.05). 
%
% Output:
%     - Table of outliers detected in a multivariate sample.
%
% Additional Files Needed:
%    ACR.m
%
% Example: From the example of Rencher (2002, p. 79). Consisting of 
% measurements of the ramus bone length at four different ages 
% (8, 8.5, 9 & 9.5 yr) on each of 20 boys. We are interesting to detect any
% multivariate outliers. We use a significance of 0.1.
%
%                ---------------------------------------------------
%                                             Age
%                         ------------------------------------------
%                Subject     8          8.5          9          9.5   
%                ---------------------------------------------------
%                    1     47.8        48.8        49.0        49.7
%                    2     46.4        47.3        47.7        48.4
%                    3     46.3        46.8        47.8        48.5
%                    4     45.1        45.3        46.1        47.2
%                    5     47.6        48.5        48.9        49.3
%                    6     52.5        53.2        53.3        53.7
%                    7     51.2        53.0        54.3        54.4
%                    8     49.8        50.0        50.3        52.7
%                    9     48.1        50.8        52.3        54.4
%                   10     45.0        47.0        47.3        48.3
%                   11     51.2        51.4        51.6        51.9
%                   12     48.5        49.2        53.0        55.5
%                   13     52.1        52.8        53.7        55.0
%                   14     48.2        48.9        49.3        49.8
%                   15     49.6        50.4        51.2        51.8
%                   16     50.7        51.7        52.7        53.3
%                   17     47.2        47.7        48.4        49.5
%                   18     53.3        54.6        55.1        55.3
%                   19     46.2        47.5        48.1        48.4
%                   20     46.3        47.6        51.3        51.8
%                ---------------------------------------------------                                        
%
% Total data matrix must be:
%     X = [47.8 48.8 49 49.7;
%          46.4 47.3 47.7 48.4;
%          46.3 46.8 47.8 48.5;
%          45.1 45.3 46.1 47.2;
%          47.6 48.5 48.9 49.3;
%          52.5 53.2 53.3 53.7;
%          51.2 53 54.3 54.4;
%          49.8 50 50.3 52.7;
%          48.1 50.8 52.3 54.4;
%          45 47 47.3 48.3;
%          51.2 51.4 51.6 51.9;
%          48.5 49.2 53 55.5;
%          52.1 52.8 53.7 55;
%          48.2 48.9 49.3 49.8;
%          49.6 50.4 51.2 51.8;
%          50.7 51.7 52.7 53.3;
%          47.2 47.7 48.4 49.5;
%          53.3 54.6 55.1 55.3;
%          46.2 47.5 48.1 48.4;
%          46.3 47.6 51.3 51.8];
% 
% Calling on Matlab the function: 
%                wilksMultivariateOutlier(X,0.10)
%
% Answer is:
%
% Table of observation(s) resulting as multivariate outlier(s).
% ----------------------------------------------
%                             D2
% Observation              observed
% ----------------------------------------------
%       9                  11.0301
% ----------------------------------------------
% With a given significance level of: 0.10
% Critical value for the maximum squared Mahalanobis distance: 10.9645
% D2 = squared Mahalanobis distance.
%
% Created by A. Trujillo-Ortiz, R. Hernandez-Walls, A. Castro-Perez and 
%            K. Barba-Rojo
%            Facultad de Ciencias Marinas
%            Universidad Autonoma de Baja California
%            Apdo. Postal 453
%            Ensenada, Baja California
%            Mexico.
%            atrujo@uabc.mx
%
% Copyright. September 13, 2006.
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and 
%    K. Barba-Rojo. (2006).
%    MOUTLIER1:Detection of Outlier in Multivariate Samples Test. A MATLAB
%    file. [WWW document].
%    URL http://www.mathworks.com/matlabcentral/fileexchange/
%        loadFile.do?objectId=12252
%
% References:
% Rencher, A. C. (2002), Methods of Multivariate Analysis. 2nd. ed.
%           New-Jersey:John Wiley & Sons. Chapter 4, pp. 101-104.
% Wilks, S. S. (1963), Multivariate Statistical Outliers. Sankhya, 
%           Series A, 25: 407-426.
% Yang, S. S. and Lee, Y. (1987), Identification of a Multivariate
%           Outlier. Presented at the Annual  Meeting of the American
%           Statistical Association, San Francisco, August 1987.
%

if nargin < 2, 
   alpha = 0.05;  %(default)
end 

if nargin < 1, 
   error('Requires at least one input arguments.');
end

mX = mean(X); %Means vector from data matrix X.
[n,p] = size(X);
difT = [];

for j = 1:p;
   eval(['difT=[difT,(X(:,j)-mean(X(:,j)))];']); %squared Mahalanobis dist.
end

S = cov(X);
D2T = difT*inv(S)*difT'; 
[D2,cc] = sort(diag(D2T));  %Ascending squared Mahalanobis distances.

D2C = ACR(p,n,alpha);

idx = find(D2 >= D2C);
o = cc(idx);
io = D2(idx);

if isempty(o);
    disp(' ')
    fprintf('With a given significance level of: %.2f\n', alpha);
    disp('Non observation(s) resulting as multivariate outlier(s).');
else
    disp(' ')
    disp('Table of observation(s) resulting as multivariate outlier(s).')
    fprintf('----------------------------------------------\n');  
    disp('                            D2');
    disp('Observation              observed');
    fprintf('----------------------------------------------\n');  
    fprintf(' %6.0f               %10.4f\n',[o,io].');
    fprintf('----------------------------------------------\n');  
    fprintf('With a given significance level of: %.2f\n', alpha);
    fprintf('Critical value for the maximum squared Mahalanobis')
    fprintf(' distance: %.4f\n', D2C);
    disp('D2 = squared Mahalanobis distance.');
end
 
return,

function x = ACR(p,n,alpha);
%ACR Upper percentiles critical value for test of single multivariate 
% normal outlier. From the method given by Wilks (1963) and approaching to
% a F distribution function by the Yang and Lee (1987) formulation, we 
% provide an m-file to get the critical value of the maximun squared 
% Mahalanobis dist. to detect outliers from a normal multivariate sample.
%
% Syntax: function x = ACR(p,n,alpha) 
%   $$ The function's name is giving as a gratefull to Dr. Alvin C. Rencher
%      for his unvaluable contribution to multivariate statistics with his 
%      text 'Methods of Multivariate Analysis'.$$
%      
%     Inputs:
%          p - number of independent variables.
%          n - sample size.
%      alpha - significance level (default = 0.05).
%
%     Output:
%          x - critical value of the maximun squared Mahalanobis distance.
%
% We can generate all the critical values of the maximun squared 
% Mahalanobis distance presented on the Table XXXII of by Barnett and 
% Lewis (1978) and Table A.6 of Rencher (2002). Also with any given 
% significance level (alpha).
%
% Example: For p = 3; n = 25; alpha=0.01;
%
% Calling on Matlab the function: 
%                ACR(p,n,alpha)
%
% Answer is:
%
%     13.1753
%
%  Created by A. Trujillo-Ortiz, R. Hernandez-Walls, A. Castro-Perez and 
%             K. Barba-Rojo
%             Facultad de Ciencias Marinas
%             Universidad Autonoma de Baja California
%             Apdo. Postal 453
%             Ensenada, Baja California
%             Mexico.
%             atrujo@uabc.mx
%
%  Copyright. August 20, 2006.
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and
%    K. Barba-Rojo. (2006).
%    ACR:Upper percentiles critical value for test of single multivariate 
%    normal outlier. A MATLAB file. [WWW document]. 
%    URL http://www.mathworks.com/matlabcentral/
%    fileexchange/loadFile.do?objectId=12161
%
%  References:
%  Barnett, V. and Lewis, T. (1978), Outliers on Statistical Data.
%           New-York:John Wiley & Sons.
%  Rencher, A. C. (2002), Methods of Multivariate Analysis. 2nd. ed.
%            New-Jersey:John Wiley & Sons. Chapter 13 (pp. 408-450).
%  Wilks, S. S. (1963), Multivariate Statistical Outliers. Sankhya, 
%            Series A, 25: 407-426.
%  Yang, S. S. and Lee, Y. (1987), Identification of a Multivariate
%            Outlier. Presented at the Annual  Meeting of the American
%            Statistical Association, San Francisco, August 1987.
%

if nargin < 3,
   alpha = 0.05; %(default)
end; 

if (alpha <= 0 | alpha >= 1)
   fprintf('Warning: significance level must be between 0 and 1\n');
   return;
end;

if nargin < 2, 
   error('Requires at least two input arguments.');
   return;
end;

a = alpha;
Fc = finv(1-a/n,p,n-p-1); %F distribution critical value with p and n-p-1 
                      % degrees of freedom using the Bonferroni correction. 
ACR = (p*(n-1)^2*Fc)/(n*(n-p-1)+(n*p*Fc)); 
  % = ((-1*((1/(1+(Fc*p/(n-p-1))))-1))*((n-1)^2))/n;
x = ACR;

return,