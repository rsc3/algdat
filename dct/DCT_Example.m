% This function produces the 8 point DCT and transforms a set of input images using the 8 point DCT
close all,
clc,
clear
N1 = 8;
N2 = 8;
C = DCT_2(ones(N1,N2));
D = dct2(ones(N1,N2)); % Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))

NPDCT = NPointDCT2_Rodriguez(N1);
% Matlab built in function
subplot(2,2,3);imagesc(NPDCT);colormap(gray(256))
subplot(2,2,4);imagesc(DM);colormap(gray(256))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DCT 4
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
imHorizontal1 = [0   0   0   0   0   0   0   0;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255];
C = DCT_2(imHorizontal1);
D = dct2(imHorizontal1); % Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))
       
imHorizontal2 = [255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           0   0   0   0   0   0   0   0;
           255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255];
C = DCT_2(imHorizontal2);
D = dct2(imHorizontal2-127); % Matlab built in function
subplot(2,2,3);imagesc(C);colormap(gray(256))
subplot(2,2,4);imagesc(D);colormap(gray(256))

imVert1 = [0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255;
           0  255 255 255 255 255 255 255];
C = DCT_2(imVert1);
D = dct2(imVert1); % Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))

imVert2 = [255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255;
           255 255 255 255 0 255 255 255];
C = DCT_2(imVert2);
D = dct2(imVert2); % Matlab built in function
subplot(2,2,3);imagesc(C);colormap(gray(256))
subplot(2,2,4);imagesc(D);colormap(gray(256))

imDiag1 = [0   255 255 255 255 255 255 255;
           255 0   255 255 255 255 255 255;
           255 255 0   255 255 255 255 255;
           255 255 255 0   255 255 255 255;
           255 255 255 255 0   255 255 255;
           255 255 255 255 255 0   255 255;
           255 255 255 255 255 255 0   255;
           255 255 255 255 255 255 255 0 ];
C = DCT_2(imDiag1);
D = dct2(imDiag1); % Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))

imDiag2 = [255 255 255 255 255 255 255 255;
           255 255 255 255 255 255 255 255;
           0   255 255 255 255 255 255 255;
           255 0   255 255 255 255 255 255;
           255 255 0   255 255 255 255 255;
           255 255 255 0   255 255 255 255;
           255 255 255 255 0   255 255 255;
           255 255 255 255 255 0   255 255];
C = DCT_2(imDiag2);
D = dct2(imDiag2); % Matlab built in function
subplot(2,2,3);imagesc(C);colormap(gray(256))
subplot(2,2,4);imagesc(D);colormap(gray(256))

imRand1 = [128 255 255 128 255 255 255 128;
           255 255 255 128 255 255 255 255;
           255 255 128 255 255 255 128 255;
           128 128 255 255 128 255 255 255;
           255 255 128 128 255 255 128 255;
           255 128 255 128 128 128 255 255;
           255 255 128 125 255 128 255 255;
           128 255 255 255 255 255 255 128];
C = DCT_2(imRand1);
D = dct2(imRand1); % Matlab built in function
figure;subplot(2,2,1);imagesc(C);colormap(gray(256))
subplot(2,2,2);imagesc(D);colormap(gray(256))

imRand2 = [255 0 255 255 0 255 0 255;
           0   0 0   0   0 0   0 0;
           255 0 255 255 0 255 0 255;
           0   0 0   0   0 0   0 0;
           255 0 255 255 0 255 0 255;
           0   0 0   0   0 0   0 0;
           255 0 255 255 0 255 0 255;
           255 0 255 255 0 255 0 255];
C = DCT_2(imRand2);
D = dct2(imRand2); % Matlab built in function
subplot(2,2,3);imagesc(C);colormap(gray(256))
subplot(2,2,4);imagesc(D);colormap(gray(256))