clear;
clc;
close all;

sX = imread('squareX.bmp');
GG = imread('GoldenGate512x512.bmp');
Trolley = imread('Trolley512x512.bmp');

DCT_128 = NPointDCT2(128);
DCT_512 = NPointDCT2(512);

DCT_sX = (DCT_128*double(sX(:,:,1)))*DCT_128';
figure; mesh(DCT_sX);
figure;imagesc(DCT_sX);colormap(gray(64))
DCT_GG = (DCT_512*double(GG(:,:,1)))*DCT_512';
figure; mesh(DCT_GG);
figure;imagesc(DCT_GG);colormap(gray(128))
DCT_Trolley = (DCT_512*double(Trolley(:,:,1)))*DCT_512';
figure; mesh(DCT_Trolley);
figure;imagesc(DCT_Trolley);colormap(gray(128))