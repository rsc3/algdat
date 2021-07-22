function d_bs = ImpResCoeffBS(fs,fa,fb,L)
% fs is the sampling frequency. 
% fa and fb are the frequency that define the stopband and pass band. The
% region between the two frequency is the stop band. 
% L is the length you desire for your filter.

wc1 = pi*(fa/fs);
wc2 = pi*(fb/fs);
k = 1;
M = (L-1)/2;
for n = 0:(L-1)
    if n == M                      % Calculate the center coefficient value
        d_bs(k) = 1-((wc2 - wc1)/pi);
        k = k+1;
    else
        d_bs(k) = (sin(pi*(n-M)))/(pi*(n-M))-...
                            ((sin(wc2*(n-M)))-(sin(wc1*(n-M))))/(pi*(n-M));

        k = k+1;
    end
end