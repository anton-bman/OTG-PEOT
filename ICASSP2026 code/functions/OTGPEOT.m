function [f0s, f0s_iter, M] =  OTGPEOT(y, prior_est, gamma, beta, varargin)
%%%%
%   OTGPEOT - OFF-THE-GRID MULTI-PITCH ESTIMATION USING OPTIMAL TRANSPORT
%
%   [f0s, alpha, M] = OTGPEOT(y, prior_est, gamma, beta, maxIter, allFreqs)
%
%   Algorithm solving the problem 
%   min_{M, omega} <C_omega, M> + gamma||rhat - AM1||_2^2
%
%   INPUTS: 
%   y           multipitch input signal (or single-pitch), size 1 x N
%   prior_est   prior estimate of the pitch, in [0, 2pi), size 1 x Np
%   gamma       regularization parameter, gamma > 0 
%   beta        sparsity regularization parameter, beta > 0
%
%   OPTIONAL INPUT:
%   freqGrid    frequency grid, vector of size 1xNf
%   maxHarm     maximum number of assumed harmonics of the signal, default
%               set to infinity
%   
%
%   OUTPUTS: 
%   f0s         vector consisting of the angular fundamental
%               frequencies.
%   f0s_iter    same as above, with estimates for each iteration.
%   M           final transport plan. Note: with input y normalized to unit
%               variance
%
%
%   Reference:  "Off-The-Grid Multi-Pitch Estimation using Optimal
%               Transport", submitted to ICASSP 2026.
%
%   Implemented by: Anton Björkman
%   Date: October 8, 2025

% normalize input
y = y/std(y);


if nargin > 4
    allFreqs = varargin{1};
else
    allFreqs =  linspace(0.015, pi, 2260);
end
if nargin > 5
    maxHarm = varargin{2};
end

f0s_iter = [];
f0s_iter(1,:) = prior_est;
currentEst = prior_est';
currentEst = currentEst(currentEst>0);

N = length(y);
nDelays = round(N*3/4);
delays = 1:nDelays;

% Sampled covariance 
R_y = xcorr(y, 'unbiased');
r = (R_y(N+1:(N+nDelays)));
r_real = [real(r); imag(r)]; % Split into real and imaginary part
nPitches = length(currentEst);


% Initialize the dual variable
lambda = zeros(length(r_real),1);
lambda_old = lambda;

tol = 1e-7;
tol2 = 1e-9;

K = 15;
for iterIdx = 1:100

    


    C_small = arrayfun(@(f) generateCostMatrix(allFreqs, f, maxHarm), currentEst, ...
        'UniformOutput', false);
    % Add the constant beta to promote 
    % sparsity in the transport plan.
    C_small = [C_small{:}] + beta;

    % Split into real and imaginary parts.
    A_full = exp(1j*delays(:)*allFreqs);
    A_real = [real(A_full); imag(A_full)];

    % Calculate the mapping between frequencies and their estimated
    % harmonic number, see h_hat in the paper.
    for pitchidx = 1:nPitches
        h(pitchidx, :) = h_map(allFreqs, currentEst(pitchidx));
    end

    if iterIdx == 1
        M = 1e-3*ones(size(C_small));
    end


    %%%%% Optimize for transport (Equation (4)) %%%%%
    M_old = M;
    [M, lambda] = transport_newton(r_real, C_small, lambda, M, A_real, gamma, K);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%% Optimize for pitch (Equation (8)) %%%%%%%
    temp = [];
    for pitchIdx = 1:nPitches
        temp(pitchIdx) = (allFreqs.^2)*M(:,pitchIdx)/( (h(pitchIdx,:).*allFreqs)*(M(:,pitchIdx)) );
    end
    currentEst = temp;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if numel(temp) < size(f0s_iter, 2)
        temp = [temp, zeros(1, size(f0s_iter, 2) - numel(temp))];
    end

    f0s_iter(iterIdx+1,:) = temp;


    %%%%%%% Convergence stuff %%%%%%%
    errorVal(iterIdx) = abs(sum(sum(C_small.*M - C_small.*M_old)) + gamma*sum(abs(lambda.^2)))/abs(sum(sum(C_small.*M_old)) + gamma*sum(abs(lambda_old.^2)));
    lambda_old = lambda;
    maxPeriod = 2;
    converged = false;
    if errorVal(iterIdx) <= tol2
        converged = true;
    end
    % Check for relative error between two nearby values
    for P = 1:min(maxPeriod, iterIdx-1)
        if abs(errorVal(iterIdx) - errorVal(iterIdx-P)) <= tol
            converged = true;
            cycleStart = iterIdx-P+1;
            cycleEnd = iterIdx;
            cycleErrors = errorVal(cycleStart:cycleEnd);

            % Find the iteration with the minimum error in this cycle
            [~, idxMin] = min(cycleErrors);
            bestIter = cycleStart + idxMin - 1;  % map back to absolute iteration
            break;  % no need to check other periods
        end
    end
    if converged
        break;
    end
    if errorVal(iterIdx) <= tol
        break;
    end
    %%%%%%% Convergence stuff %%%%%%%

end





% pitch_est = saved_est;

if exist('bestIter', 'var')
    f0s = f0s_iter(bestIter,:);
else
    f0s = f0s_iter(end,:);
end
end


function h = h_map(x, y)
% h_map computes integer h_i minimizing (x_i / y - h)^2
% for x: vector, y: scalar
h = max(1, round(x ./ y));  % Element-wise operation
end
