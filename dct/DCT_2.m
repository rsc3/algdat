function B_DCT = DCT_2(B) 
    [N1, N2] = size(B);
    B_DCT = zeros(N1, N2);
    for k1 = 0:N1-1
        for k2 = 0:N2-1
            tmp = 0;
            for n1 = 0:N1-1
                for n2 = 0:N2-1
                    tmp = tmp + B(n1+1,n2+1)*cos((pi*(2*n1+1)*k1)/(2*N1))*...
                                cos((pi*(2*n2+1)*k2)/(2*N2));
                end
            end
            b1 = get_b(k1,N1);
            b2 = get_b(k2,N2);
            B_DCT(k1+1,k2+1) = b1*b2*tmp;
        end
    end
end
function b = get_b(k,N)
    if k == 0
        b = 1/sqrt(N);
    else
        b = sqrt(2/N);
    end
end
