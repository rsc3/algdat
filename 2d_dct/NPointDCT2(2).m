function NPDCT = NPointDCT2(N)

% This function produces the n-point DCT_2 of
% a size received.  The n-point DCT_2 matrix will 
% be a square matrix.

for k = 0:N-1
    for n = 0:N-1
        if k == 0
            NPDCT(k+1,n+1) = sqrt(inv(N));
        else
            temp = cos((pi*(2*n + 1)*k)/(2*N));
            NPDCT(k+1,n+1) = sqrt(2/N)*temp;%1/sqrt(N) = sqrt(2/N)
        end
    end
end
NPDCT;