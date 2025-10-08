clear all
clc

%%
fs = 20000;
N = 500;
t = (0:N)/fs;
pitches = [221 325 264];% 467 221];
P = length(pitches);

% to ensure an error of at least 1 Hz, and at most 2 Hz
err = sign(rand(1,P) - 0.5).*(1 + rand(1,P));
prior_est =  pitches + err;
nHarmonics = 6+randi(4,1,P); % Harmonic order of each pitch, random between
                             % 6 and 10
inharm = 0.6*1e-3; % Inharmonicity parameter


%%%% Generate the signal %%%%
y = zeros(size(t));
for p = 1:length(pitches)
    H = (1:nHarmonics(p))'; % total number of harmonics
    F = pitches(p)*H + randn(size(H))*inharm*fs/2/pi; % harmonics
    a = exp(-abs(H-2)/nHarmonics(p)); % amplitudes
    phi = 2*pi*rand(nHarmonics(p),1); % random phase
    y = y + sum(a .* exp(1j*(2*pi*F.*t + phi)), 1);
end
%%% Adding the noise
P_signal = (mean(abs(y).^2));
SNR_dB = 20; % desired SNR
P_noise = P_signal / 10^(SNR_dB/10);
n = (randn(size(y)) + 1j*randn(size(y))) / sqrt(2)* sqrt(P_noise);
y_noisy = (y + n).';


%%
%%% Frequencies on the frequency grid, converted to angular frequency
nGridPoints = 500; % Total number of grid points on the frequency grid
allFreqs =  linspace(min(pitches)-10, max(pitches.*nHarmonics)+10, nGridPoints)/fs*2*pi;


% Estimation
%%% Hyper-parameters
gamma = 1e-2;
beta = 1e-4;
[est, saved_est, M] = OTGPEOT(y_noisy, prior_est/fs*2*pi, gamma, beta, allFreqs, Inf);
est = est*fs/2/pi;
saved_est = saved_est*fs/2/pi;
%%


f = figure(1); clf(f)
titles = {'Pitch 1','Pitch 2','Pitch 3'};
for i = 1:P
    subplot(P,1,i)
    plot(saved_est(:,i), 'LineWidth', 1.4); hold on
    yline(prior_est(i), '--', 'LineWidth', 1.2);
    yline(pitches(i), ':', 'LineWidth', 1.4);
    ylim([min([saved_est(:,i); prior_est(i); pitches(i)])-0.1, ...
          max([saved_est(:,i); prior_est(i); pitches(i)])+0.1])
    xlim([1,length(saved_est)])
    xlabel('Iteration')
    ylabel('Pitch [Hz]')
    title(titles{i})
    legend({'Estimate','Prior','True pitch'}, 'Location','best')
    grid on
end
sgtitle('Pitch Estimation per Iteration')



