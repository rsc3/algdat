function [prob, m, sigma,  prob_ikn, numberOfIterations] = ...
             simpleExpectationMaximization(x, K, prob, m, sigma, threshold)
% Ben Rodriguez 
% EN.685.621
% This code is for eduational purposes
% Simple example of expectation maximization

N = size(x, 2);
D = size(x, 1);

initializeD = ones(D, 1);
initializeN = ones(1, N);
numberOfIterations = 0;

figure; 
axis([0.5 4.5 1.5 3.5])
hold on;
colors = lines(K);
while 1
    % Save previous values to use for determining if EM has converged
    meanPrevious = m;
    sigmaPrevious = sigma;
    probabilityPrevious = prob;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation Step
    for k = 1:K
        g(k, :) = (exp(- sum((x - m(:, k) * ...
            ones(1, size(x, 2))) .^ 2) / (sigma(k)^2) / 2) / ...
            (sqrt(2 * pi) * sigma(k)) ^ D);
         probKg(k, :) = prob(k) * g(k, :);
    end
     prob_ikn =  probKg ./ (ones(K, 1) * sum( probKg));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Maximization Step
     sum_prob_ikn = sum( prob_ikn');
    
    for k = 1:K
        m(:, k) = sum(((initializeD *  prob_ikn(k, :)) .* x / ...
                                                       sum_prob_ikn(k))')';
        sigma(k) = sqrt(sum(sum((x - m(:, k) * initializeN) .^ 2) .*  ...
                                   prob_ikn(k, :)) /  sum_prob_ikn(k) / D);
    end
    prob =  sum_prob_ikn / sum( sum_prob_ikn);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Here should be an output to the command window to show the values at
    % each iteration.
    
    % Plot the results for visualization analysis
    numberOfIterations = numberOfIterations + 1;
    
    colors = lines(K);
    for k = 1:K
        plot(m(1, k), m(2, k), 'o','Color',colors(k,:), ...
        'MarkerSize', 6, 'LineWidth', 2)
    end
    iterEM = sprintf('%d', numberOfIterations);
    title(sprintf('EM iteration number % s', iterEM))
    % pause placed to see the points converge on the figure.
    % pause
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The following code is to determine if the values have converged.

    meanDelta = max(sqrt(sum((m - meanPrevious) .^ 2)));
    sMean = mean(sqrt(sum(m .^ 2)));
    convMean = (meanDelta <= sMean * threshold);

    sigmaDelta = max(sqrt(sum((sigma - sigmaPrevious) .^ 2)));
    sSignma = mean(sqrt(sum(sigma .^ 2)));
    convSigma = (sigmaDelta <= sSignma * threshold);

    pDelta = max(sqrt(sum((prob - probabilityPrevious) .^ 2)));
    sProbability = mean(sqrt(sum(prob .^ 2)));
    convProbability = (pDelta <= sProbability * threshold);
    
    if convMean & convSigma & convProbability
        break;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

[tmp,  classEM] = max( prob_ikn);
ck = find( classEM == 1);
plot(x(1, ck), x(2, ck), 'x','Color',colors(1,:))
ck = find( classEM == 2);
plot(x(1, ck), x(2, ck), 'x','Color',colors(2,:))
iterEM = sprintf('%d', numberOfIterations);
title(sprintf('EM convergance % s', iterEM))
hold off;