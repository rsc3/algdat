% Notes on fftshift
% http://matlab.izmiran.ru/help/techdoc/ref/fftshift.html
% https://www.mathworks.com/matlabcentral/fileexchange/25473-why-use-fftshift-fft-fftshift-x-in-matlab-instead-of-fft-x
clear
clc
close all

f20 = 20;
f200 = 200;
f400 = 400;
amp2 = 2;
amp3 = 3;
fs = 2000;
ts = 1/fs;
T = 10;
t = 0:ts:T;
s20 = sin(2*pi*f20*t);
s200 = amp2*sin(2*pi*f200*t);
s400 = amp3*sin(2*pi*f400*t);
s = s20 + s200 + s400;

S = fftshift(fft(s));

w = -pi:(2*pi)/(length(t)-1):pi;
f = (fs*w)/(2*pi);
figure,plot(t(1:200),s(1:200),'k',t(1:200),s20(1:200),'r',t(1:200),s200(1:200),'b--',t(1:200),s400(1:200),'g')
legend('Sum of Signals','20Hz','200Hz','400Hz')
figure,plot(f,abs(S))

fa = 150;
fb = 250;
L = 101;
d_bs = ImpResCoeffBS(fs,fa,fb,L);

[h_hamm] = Hamming(fs,d_bs);

n = 0:(L-1);
p = f;

H = h_hamm*(exp(-i*pi/fs)).^(n'*p);
S_normalized = abs(S)/max(abs(S));
 
figure,plot(p,abs(H));
title('Magnitude Response Given Band Stop Impulse Response');
ylabel('|H(\omega)|');
xlabel('Frequency (Hz)');
figure,plot((p/fs)*2*pi,abs(H));
title('Magnitude Response Given Band Stop Impulse Response');
ylabel('|H(\omega)|');
xlabel('\omega in unts of \pi');

S_hat = S.*H;
figure,plot(f,abs(S_hat))

s_hat = ifft(ifftshift(S_hat));
figure,plot(real(s_hat))
figure,plot(t(1:200),s(1:200),'b',t(1:200),real(s_hat(26:225)),'r--')
figure,plot(t(1:200),(s20(1:200) + s400(1:200)),'b',t(1:200),real(s_hat(26:225)),'r--')