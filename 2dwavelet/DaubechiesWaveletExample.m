% This code is for educational and research purposes of comparisons. This
% is code that provides the use of the Daubechies wavelet for transforming
% a spacial domain image to a wavelet domain. It also shows the inverse of
% the transform.

clear;
close all;
clc;

I = imread('Trolley512x512.bmp');
%I = imread('squareX.bmp');
figure;image(I);colormap(gray(256))
[row,col,depth] = size(I);

ID = double(I);
if depth == 3
    IN = 0.299.*ID(:,:,1) + 0.587.*ID(:,:,2) + 0.114.*ID(:,:,3);
    clear ID;
else
    IN = ID;
    clear ID;
end

Daub4 = Daubechies4(row);
IDTemp = zeros(row,col);
for i = 1:row
    IDTemp(i,:) = (inv(sqrt(2)).*(Daub4*IN(i,:)'))';
end
clear IN;
IDaub = zeros(row,col);
for j = 1:col
    IDaub(:,j) = (inv(sqrt(2)).*(Daub4*IDTemp(:,j)));
end
clear IDTemp;
clear i;
clear j;
figure;image(IDaub);colormap(gray(512))
invDaub4 = invDaubechies4(IDaub,1);
figure;image(invDaub4);colormap(gray(256))

Daub4 = Daubechies4(row/2);
IN = IDaub(1:row/2,1:col/2);
IDTemp = zeros(row/2,col/2);
for i = 1:row/2
    IDTemp(i,:) = (inv(sqrt(2)).*(Daub4*IN(i,:)'))';
end
clear IN;
IDaub2 = zeros(row/2,col/2);
for j = 1:col/2
    IDaub2(:,j) = (inv(sqrt(2)).*(Daub4*IDTemp(:,j)));
end
clear IDTemp;
IDaub(1:row/2,1:col/2) = IDaub2;
figure;image(IDaub);colormap(gray(1048))

invDaub4 = invDaubechies4(IDaub,2);
figure;image(invDaub4);colormap(gray(256))