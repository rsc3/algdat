% Notes on fftshift
% http://matlab.izmiran.ru/help/techdoc/ref/fftshift.html
% https://www.mathworks.com/matlabcentral/fileexchange/25473-why-use-fftshift-fft-fftshift-x-in-matlab-instead-of-fft-x
clear;
clc;
close all;

sX = imread('squareX.bmp');
GG = imread('GoldenGate512x512.bmp');
Trolley = imread('Trolley512x512.bmp');

DFT_128 = DFT_DS(128);
DFT_512 = DFT_DS(512);

DFT_sX = (DFT_128*double(sX(:,:,1)))*DFT_128';
figure; mesh (fftshift(abs(DFT_sX)));
DFT_GG = (DFT_512*double(GG(:,:,1)))*DFT_512';
figure; mesh (fftshift(abs(DFT_GG)));
DFT_Trolley = (DFT_512*double(Trolley(:,:,1)))*DFT_512';
figure; mesh (fftshift(abs(DFT_Trolley)));

figure; mesh (log2(fftshift(abs(DFT_sX+0.001))));
figure; mesh (log2(fftshift(abs(DFT_GG+0.001))));
figure; mesh (log2(fftshift(abs(DFT_Trolley+0.001))));