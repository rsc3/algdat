function invDaub4 = invDaubechies4(I,level)
% Inverse Daubechies Wavelet
% This code is for educational and research purposes of comparisons. This
% is code that calculates the inverse Daubechies wavelet using the 
% Daubechies-4 wavelet coefficients. 

[rows,cols] = size(I);
row = rows/(2^(level-1));
col = cols/(2^(level-1));

IDaub = zeros(rows,cols);

for L = 1:level
    IDTemp = zeros(row,col);
    IDT = zeros(row,col);
    Daub4 = Daubechies4(row);	
	for j = 1:col
        temp1 = (inv(sqrt(2)).*(Daub4(1:row/2,:)'*I(1:row/2,j)));
        temp2 = (inv(sqrt(2)).*(Daub4((row/2)+1:row,:)'*I((row/2)+1:row,j)));        
        IDTemp(:,j) = temp1+temp2;
	end
	for i = 1:row
        temp1 = (inv(sqrt(2)).*(Daub4(1:row/2,:)'*IDTemp(i,1:col/2)'))';
        temp2 = (inv(sqrt(2)).*(Daub4((row/2)+1:row,:)'*IDTemp(i,(col/2)+1:col)'))';        
        IDT(i,:) = temp1+temp2;
	end
    IDaub(1:row,1:col) = IDT;
    I(1:row,1:col) = IDT;
	clear IDTemp;
	clear i;
	clear j;
    row = row*2;
    col = col*2;
    clear IDT;
end

invDaub4 = IDaub;