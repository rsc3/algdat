function DFT_Matrix = DFT_DS(N)
% This algorithm is built for creating Discrete Fourier
% Transform Matrix.  The algorithm takes N input, which 
% creates NxN DFT matrix. The input value has to 2^n.

DFT_Matrix = [];

for k = 1:N,
    for n = 1:N,
        DFT_Matrix(k,n) = exp((-2*i*pi*(n-1)*(k-1))/N);
    end
end