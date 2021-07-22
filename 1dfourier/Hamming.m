function [h] = Hamming(fs,d)

% This function implements the Hamming window.  
%
% inputs:
%    d - contains the coefficients of the impulse response
%    fs - sampling frequency
% output: 
%    h - filter coefficients of w_hamm*d 
%
% Note: w_hamm is to be multiplied by the signal in the frequency domain. 

L = length(d);
M = (L-1);

n = 0:(L-1);
p = -fs:fs;

k = 0; % Create window values.
for kk = 1:L
    w_hamm(kk) = (0.54-0.46*cos((2*pi*k)/M));
    k = k+1;
end

h = w_hamm.*d;   % Window and impulse response of the desired filter.
H = h*(exp(-j*pi/fs)).^(n'*p);

figure,subplot(3,2,1); plot(d);
title(['Rectangular Window Impulse Response: N=' num2str(L)]);
ylabel('d(n)')
xlabel('Samples (n)');

% magnitude response given the impuse responce
D = d*(exp(-j*pi/fs)).^(n'*p);
subplot(3,2,2); plot(p,abs(D));
title('Magnitude Response given impulse responce');
ylabel('|D(\omega)|');
xlabel('Frequency (Hz)');

subplot(3,2,3); plot(w_hamm);
title(['Hamming Window: N=' num2str(L)]);
ylabel('w(n)')
xlabel('Samples (n)');

% magnitude response Blackman window
W = w_hamm*(exp(-j*pi/fs)).^(n'*p);
subplot(3,2,4); plot(p/fs,abs(W));
title('Hamming Magnitude Response');
ylabel('|W(\omega)|');
xlabel('Frequency (Hz)');

subplot(3,2,5),plot(h)
title('Hamming and Rectangular Window');
ylabel('h(n)')
xlabel('Samples (n)');


subplot(3,2,6);plot(p,abs(D));
hold on
plot(p,abs(H),'r');
title('Hamming and Rectangular Window');
ylabel('|D(\omega)| and W(|\omega|');
xlabel('Frequency (Hz)');
hold off