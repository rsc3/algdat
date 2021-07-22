function Daub4 = Daubechies4(size)
% This code is for educational and research purposes of comparisons. This
% is code that calculates the Daubechies-4 wavelet coefficients. 

H(1) = (1+sqrt(3))/4;
H(2) = (3+sqrt(3))/4;
H(3) = (3-sqrt(3))/4;
H(4) = (1-sqrt(3))/4;
G(1) = (-1+sqrt(3))/4;
G(2) = (3-sqrt(3))/4;
G(3) = (-3-sqrt(3))/4;
G(4) = (1+sqrt(3))/4;

j = 1;
for i = 1:size/2
    if (j+2) > size
        Daub4(i,j) = H(1);
        Daub4(i,j+1) = H(2);
        Daub4(i,1) = H(3);
        Daub4(i,2) = H(4);
        Daub4(i+(size/2),j) = G(1);
        Daub4(i+(size/2),j+1) = G(2);
        Daub4(i+(size/2),1) = G(3);
        Daub4(i+(size/2),2) = G(4);
    else
        Daub4(i,j) = H(1);
        Daub4(i,j+1) = H(2);
        Daub4(i,j+2) = H(3);
        Daub4(i,j+3) = H(4);
        Daub4(i+(size/2),j) = G(1);
        Daub4(i+(size/2),j+1) = G(2);
        Daub4(i+(size/2),j+2) = G(3);
        Daub4(i+(size/2),j+3) = G(4);
    end
    j = j + 2;
end
        