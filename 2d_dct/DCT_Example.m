% This function produces the 8 point DCT and transforms a set of input images using the 8 point DCT
close all,
clc,
clear
N1 = 8;
N2 = 8;
C = DCT_2(ones(N1,N2));
D = dct2(ones(N1,N2)); % Matlab built in functio
DM = dctmtx(8);% Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))

NPDCT = NPointDCT2(N1);
% Matlab built in function
subplot(2,2,3);imagesc(NPDCT);colormap(gray(256))
subplot(2,2,4);imagesc(DM);colormap(gray(256))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DCT 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
for i = 1:N1
    subplot(2,4,i);stem(NPDCT(i,:));axis([0 9 -0.5 0.5]);
end
hold off;

figure;
hold on;
for i = 1:N1
    subplot(2,4,i);stairs(NPDCT(i,:));axis([0 9 -0.5 0.5]);
end
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab DCT Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
for i = 1:N1
    subplot(2,4,i);stem(DM(i,:));axis([0 9 -0.5 0.5]);
end
hold off;

figure;
hold on;
for i = 1:N1
    subplot(2,4,i);stairs(DM(i,:));axis([0 9 -0.5 0.5]);
end
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imHorizontal = [0   0   0   0   0   0   0   0;
           0   0   0   0   0   0   0   0;
           0   0   0   0   0   0   0   0;
           0   0   0   0   0   0   0   0;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255];
C = DCT_2(imHorizontal);
D = dct2(imHorizontal); % Matlab built in function
NPTD = (NPDCT*imHorizontal)*NPDCT';
figure;subplot(2,3,1);imagesc(imHorizontal);colormap(gray(256))
subplot(2,3,4);imagesc(C);colormap(gray(256))
       
imVert = [0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255;
           0  0   0   0 255 255 255 255];
C = DCT_2(imVert);
D = dct2(imVert); % Matlab built in function
NPTD = NPDCT*imVert*NPDCT';
subplot(2,3,2);imagesc(imVert);colormap(gray(256))
subplot(2,3,5);imagesc(D);colormap(gray(256))

imDiag =[255  0   0   0   0   0   0   0;
           0  255  0   0   0   0   0   0;
           0   0  255  0   0   0   0   0;
           0   0   0  255  0   0   0   0;
           0   0   0   0  255  0   0   0;
           0   0   0   0   0  255  0   0;
           0   0   0   0   0   0  255  0;
           0   0   0   0   0   0   0  255];
C = DCT_2(imDiag);
D = dct2(imDiag); % Matlab built in function
NPTD = NPDCT*imDiag*NPDCT';
subplot(2,3,3);imagesc(imDiag);colormap(gray(256))
subplot(2,3,6);imagesc(NPTD);colormap(gray(256))

