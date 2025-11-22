clc; clear; close all;

rng(10);

%%% https://ch.mathworks.com/matlabcentral/fileexchange/37514-stbl-alpha-stable-distributions-for-matlab
function [x,p]=runthefft_single(n,h,alpha,beta,sigma,mu) % from Part I
% From fftnoncentralchi2.m (Intermediate Propability ch2)
    N=2^n;
    x=(0:N-1)'*h - N*h/2;
    s=1/(h*N);
    t=2*pi*s*((0:N-1)' - N/2);
    sgn=ones(N,1);
    sgn(2:2:N)=-1;
    phi_term = (sigma^alpha)*(abs(t).^alpha);
    skew_term = 1 - 1i*beta*sign(t)*tan(pi*alpha/2);
    CF = exp(-phi_term .* skew_term + 1i*mu.*t);
    phi = sgn .* CF;
    phi(N/2+1) = sgn(N/2+1);
    p = s .* abs(fft(phi));
end

function pdf = stablepdf_fft(z,alpha,beta,sigma,mu) % from Part I
% From fftnoncentralchi2.m (Intermediate Propability ch2)
    pmax = 18;
    step = 0.01;
    p = 14;
    maxz = round(max(abs(z))) + 5;
    while ((maxz/step + 1) > 2^(p-1)), p = p + 1; end
    if p > pmax, p = pmax; end
    if maxz/step + 1 > 2^(p-1)
        step = (maxz + 1)*1.001 / (2^(p-1));
    end
    [xgrd, bigpdf] = runthefft_single(p, step, alpha,beta, sigma, mu);
    % pdf = wintp1(xgrd, bigpdf, z); [Old wintp1 is broken]
    pdf = interp1(xgrd, bigpdf, z, 'linear');
end

function mle_uni = neglog_sym(theta,x)
    t_alpha = theta(1);
    t_logSigma = theta(2);
    mu = theta(3);
    alpha = 2 * 1./(1+exp(-t_alpha));
    sigma = exp(t_logSigma);
    beta = 0; % symmetr

    f = stablepdf_fft(x,alpha,beta,sigma,mu);

    tol = 1e-16;
    f(f<tol) = tol;
    mle_uni = -sum(log(f));
end

function [alpha_hat,sigma_hat,mu_hat,theta_hat] = mle_symstable(x)
    neglog = @(theta) neglog_sym(theta,x);
    alpha0 = 1.5;
    t_alpha0 = -log(2/alpha0 - 1);
    sigma0 = std(x);
    t_logSigma0 = log(sigma0);

    theta0 = [t_alpha0;t_logSigma0;mean(x)];
    % from http://mathworks.com/help/optim/ug/fminunc.html
    theta_hat = fminunc(neglog, theta0, ...
        optimoptions('fminunc','Algorithm','quasi-newton',...
                         'Display','off','MaxIterations',200,'MaxFunctionEvaluations',400));
    alpha_hat = 2 ./ (1 + exp(-theta_hat(1)));
    sigma_hat = exp(theta_hat(2));
    mu_hat    = theta_hat(3);

end




%%
alpha_grid = linspace(1.2,1.8,6);
m = length(alpha_grid);
n = 10;
nsim = 30;
sigma_true = 1;
mu_true = 0;

alpha_hat = zeros(nsim,m);
sigma_hat= zeros(nsim,m);
mu_hat= zeros(nsim,m);

total_iterations = m * nsim;
counter = 0;
tic;
for j = 1:m
    pd_true = makedist("Stable","alpha",alpha_grid(j),"beta",0,"gam",sigma_true,"delta",mu_true);
    for s = 1:nsim
        x = random(pd_true,n,1);
        [a,sig,mu] = mle_symstable(x);
        alpha_hat(s,j) = a;
        sigma_hat(s,j) = sig;
        mu_hat(s,j)    = mu;
        counter = counter + 1;
    
        if mod(counter, 1) == 0  
            fprintf('Progress: %5.1f%%   (alpha1 = %.2f, n = %d, sim = %d/%d)\n', ...
                100 * counter / total_iterations, alpha_grid(j), n, s, nsim);
        end
    end
end
elapsed_time = toc;
fprintf("Total computation time: %.f s",elapsed_time)
%%
figure;
subplot(1,3,1)
boxplot(alpha_hat, 'Labels', string(alpha_grid));
title('$\alpha$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\alpha}$', 'Interpreter','latex');
grid on

subplot(1,3,2)
boxplot(mu_hat, 'Labels', string(alpha_grid));
title('$\mu$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\mu}$', 'Interpreter','latex');
yline(mu_true,'--k',Label='True value')

grid on

subplot(1,3,3)
boxplot(sigma_hat, 'Labels', string(alpha_grid));
title('$\sigma$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\sigma}$', 'Interpreter','latex');
yline(sigma_true,'--k',Label='True value')

grid on



%% b

% function mle_di = neglog_sym_di(theta,x)
%     pi_ = 1/(1 + exp(-theta(1)));
%     alpha1 = 2 * 1./(1+exp(-theta(2)));
%     alpha2 = 2 * 1./(1+exp(-theta(3)));
%     sigma1 = exp(theta(4));
%     sigma2 = exp(theta(5));
%     mu1 = theta(6);
%     mu2 = theta(7);
%     beta = 0; %symetric
%     f1 = stablepdf_fft(x,alpha1,beta,sigma1,mu1);
%     f2 = stablepdf_fft(x,alpha2,beta,sigma2,mu2);
% 
%     fmix = pi_*f1+(1-pi_)*f2;
%     tol = 1e-16;
%     fmix(fmix<tol) = tol;
%     mle_di = -sum(log(fmix));
% end

function nll = neglog_sym_di(theta,x)

    % Parameter transforms
    pi_     = 1 ./ (1 + exp(-theta(1)));

    % Constrain α to realistic range to avoid instability
    alpha1  = 0.5 + 1.45 ./ (1 + exp(-theta(2)));
    alpha2  = 0.5 + 1.45 ./ (1 + exp(-theta(3)));

    sigma1  = exp(theta(4));
    sigma2  = exp(theta(5));

    mu1     = theta(6);
    mu2     = theta(7);

    % Compute PDFs
    f1 = stablepdf_fft(x,alpha1,0,sigma1,mu1);
    f2 = stablepdf_fft(x,alpha2,0,sigma2,mu2);

    % Mixture density
    f = pi_.*f1 + (1-pi_).*f2;

    % Stabilize
    f = max(f, 1e-14);

    % Soft regularization (prevents divergence)
    reg = 1e-4*(mu1^2 + mu2^2) + 1e-4*(sigma1^2 + sigma2^2);

    nll = -sum(log(f)) + reg;
end

% 
% function [pi_hat,alpha_hat1,alpha_hat2,sigma_hat1,sigma_hat2,mu_hat1,mu_hat2] = mle_symstable_di(x)
%     neglog = @(theta) neglog_sym_di(theta,x);
%     eps_ = 0.1;
%     pi_t = 0.5;
%     alpha_t1 = 1.5;
%     alpha_t2 = 1.5;
%     sigma_t1 = eps_;
%     sigma_t2 = 2*eps_;
%     mu_t1 = eps;
%     mu_t2 = -eps_;
%     theta0 = [pi_t;-log(2/alpha_t1-1);-log(2/alpha_t2-1);log(sigma_t1);log(sigma_t2);mu_t1;mu_t2];
%     % from http://mathworks.com/help/optim/ug/fminunc.html
%     theta_hat = fminunc(neglog, theta0, ...
%         optimoptions('fminunc','Algorithm','quasi-newton',...
%                          'Display','off','MaxIterations',200,'MaxFunctionEvaluations',400));
%     pi_hat   = 1/(1+exp(-theta_hat(1)));
%     alpha_hat1   = 2./(1+exp(-theta_hat(2)));
%     alpha_hat2   = 2./(1+exp(-theta_hat(3)));
%     sigma_hat1   = exp(theta_hat(4));
%     sigma_hat2   = exp(theta_hat(5));
%     mu_hat1  = theta_hat(6);
%     mu_hat2  = theta_hat(7);
% 
% end

function [pi_hat,a1,a2,s1,s2,m1,m2] = mle_symstable_di(x)

    % Robust initial location
    m = trimmean(x,20);
    s = mad(x,1)/0.6745;

    % Initial guess
    pi0 = 0.5;
    alpha0 = 1.5;

    theta0 = [
        log(pi0/(1-pi0));     % t_pi
        log(alpha0/(2-alpha0));  % t_alpha1
        log(alpha0/(2-alpha0));  % t_alpha2
        log(s);               % log sigma1
        log(s*1.5);           % log sigma2
        m + s;                % mu1
        m - s                 % mu2
    ];

    % Optimize
    opt = optimoptions('fminunc','Algorithm','quasi-newton',...
                       'Display','none','MaxIterations',300,...
                       'MaxFunctionEvaluations',600);

    theta_hat = fminunc(@(th) neglog_sym_di(th,x), theta0, opt);

    % Convert back
    pi_hat = 1./(1 + exp(-theta_hat(1)));

    % α constrained to [0.5,1.95] automatically
    a1  = 0.5 + 1.45 ./ (1 + exp(-theta_hat(2)));
    a2  = 0.5 + 1.45 ./ (1 + exp(-theta_hat(3)));

    s1  = exp(theta_hat(4));
    s2  = exp(theta_hat(5));

    m1  = theta_hat(6);
    m2  = theta_hat(7);
end


% function [pi_hat,alpha_hat1,alpha_hat2,sigma_hat1,sigma_hat2,mu_hat1,mu_hat2] = mle_symstable_di(x)
%     % The negative log-likelihood function
%     neglog = @(theta) neglog_sym_di(theta,x);
% 
%     % --- SMART INITIALIZATION using k-Means Clustering ---
% 
%     % 1. K-means clustering (k=2) to get initial assignments
%     % 'Replicates', 5 increases robustness against poor local optima in k-means
%     % Note: x must be a column vector for k-means.
%     X_data = x(:);
% 
%     try
%         [idx, ~] = kmeans(X_data, 2, 'Replicates', 5);
%         C1 = (idx == 1);
%         C2 = (idx == 2);
% 
%         % Handle potential empty clusters (unlikely for large N but good practice)
%         if sum(C1) == 0 || sum(C2) == 0
%             warning('K-means resulted in an empty cluster. Falling back to robust moment matching initialization.');
%             is_kmeans_fail = true;
%         else
%             is_kmeans_fail = false;
%         end
%     catch ME
%         warning(['K-means failed with error: ', ME.message, '. Falling back to robust moment matching initialization.']);
%         is_kmeans_fail = true;
%     end
% 
% 
%     if is_kmeans_fail
%         % Fallback: Robust initial guess for moments, assuming equal weight and spread
%         pi_t = 0.5;
%         % Use median and IQR for robustness
%         median_x = median(x);
%         iqr_x = iqr(x);
% 
%         % Split means slightly around the median
%         mu_t1 = median_x + 0.25 * iqr_x;
%         mu_t2 = median_x - 0.25 * iqr_x;
% 
%         % Split scales equally (based on overall IQR)
%         sigma_t1 = iqr_x / 4; % A fraction of overall IQR
%         sigma_t2 = iqr_x / 4; 
% 
%     else
%         % Initialization based on k-Means clusters
%         pi_t = sum(C1) / length(x);
%         mu_t1 = mean(x(C1));
%         mu_t2 = mean(x(C2));
% 
%         % Use robust scale estimation (std is fine as a start)
%         sigma_t1 = std(x(C1));
%         sigma_t2 = std(x(C2));
%     end
% 
%     % 3. Fixed or Simple Alpha Guess: 1.5 is a safe, middle-ground start
%     alpha_t1 = 1.5;
%     alpha_t2 = 1.5;
% 
%     % Ensure scales are positive and not too small for log function
%     min_sigma = 1e-4;
%     sigma_t1 = max(sigma_t1, min_sigma);
%     sigma_t2 = max(sigma_t2, min_sigma);
% 
%     % --- 4. Convert to Transformed Space (theta0) ---
% 
%     % Note on parameter order: [pi_t; t_alpha1; t_alpha2; t_logSigma1; t_logSigma2; mu1; mu2]
%     theta0 = [-log(1/pi_t - 1);        ...  % t_pi: enforces pi in (0,1)
%               -log(2/alpha_t1-1);      ...  % t_alpha1: enforces alpha1 in (0,2)
%               -log(2/alpha_t2-1);      ...  % t_alpha2: enforces alpha2 in (0,2)
%                log(sigma_t1);          ...  % t_logSigma1: enforces sigma1 in (0,inf)
%                log(sigma_t2);          ...  % t_logSigma2: enforces sigma2 in (0,inf)
%                mu_t1;                  ...  % mu1: unconstrained
%                mu_t2];                 ...  % mu2: unconstrained
% 
%     % --- Optimization ---
% 
%     % from http://mathworks.com/help/optim/ug/fminunc.html
%     theta_hat = fminunc(neglog, theta0, ...
%         optimoptions('fminunc','Algorithm','quasi-newton',...
%                          'Display','off','MaxIterations',200,'MaxFunctionEvaluations',400));
% 
%     % --- Convert back to natural parameters ---
%     pi_hat       = 1/(1+exp(-theta_hat(1)));
%     alpha_hat1   = 2./(1+exp(-theta_hat(2)));
%     alpha_hat2   = 2./(1+exp(-theta_hat(3)));
%     sigma_hat1   = exp(theta_hat(4));
%     sigma_hat2   = exp(theta_hat(5));
%     mu_hat1      = theta_hat(6);
%     mu_hat2      = theta_hat(7);
% end


%%

alpha1_grid = linspace(1.25,1.45,6);
alpha2_grid = alpha1_grid+0.5;

sigma1_true = 1; sigma2_true = 2; 
mu1_true = 0; mu2_true = 0; 
pi_true = 0.8;
% 
% sigma1_true = 1; sigma2_true = 2; 
% mu1_true = 0; mu2_true = 0; 
% pi_true = 0.8;

pi_hat = zeros(nsim,m);
alpha1_hat = zeros(nsim,m);
alpha2_hat = zeros(nsim,m);
sigma1_hat = zeros(nsim,m);
sigma2_hat = zeros(nsim,m);
mu1_hat = zeros(nsim,m);
mu2_hat = zeros(nsim,m);


total_iterations_b = m * nsim;
counter_b = 0;
tic;
for j = 1:m
    pd1_true = makedist("Stable","alpha",alpha1_grid(j),"beta",0,"gam",sigma1_true,"delta",mu1_true);
    pd2_true = makedist("Stable","alpha",alpha2_grid(j),"beta",0,"gam",sigma2_true,"delta",mu2_true);

    for s = 1:nsim
        comp = rand(n,1) < pi_true;
        x = zeros(n,1);
        x(comp)  = random(pd1_true,sum(comp),1);
        x(~comp) = random(pd2_true,n - sum(comp),1);
        [pi_hat(s,j), alpha1_hat(s,j), alpha2_hat(s,j), sigma1_hat(s,j),sigma2_hat(s,j), mu1_hat(s,j), mu2_hat(s,j)] = mle_symstable_di(x);
        counter_b = counter_b + 1;

        if mod(counter_b, 1) == 0  
            fprintf('Progress: %5.1f%%   (n = %d, sim = %d/%d)\n', ...
                100 * counter_b / total_iterations_b,n, s, nsim);
        end
    end
end
elapsed_time = toc;
fprintf("Total computation time: %.f s\n",elapsed_time)

%% 

figure;
subplot(2,3,1)
boxplot(alpha1_hat, 'Labels', string(alpha1_grid));
title('$\alpha_1$ estimates', 'Interpreter','latex');
xlabel('$\alpha_1$ grid', 'Interpreter','latex');
ylabel('$\hat{\alpha_1}$', 'Interpreter','latex');
grid on


% figure;
subplot(2,3,2)
boxplot(alpha2_hat, 'Labels', string(alpha2_grid));
title('$\alpha_2$ estimates', 'Interpreter','latex');
xlabel('$\alpha_2$ grid', 'Interpreter','latex');
ylabel('$\hat{\alpha_2}$', 'Interpreter','latex');
grid on

subplot(2,3,3)
boxplot(mu1_hat, 'Labels', string(alpha1_grid));
title('$\mu_1$ estimates', 'Interpreter','latex');
xlabel('$\alpha_1$ grid', 'Interpreter','latex');
ylabel('$\hat{\mu_1}$', 'Interpreter','latex');
yline(mu_true,'--k',Label='True value')

grid on

subplot(2,3,4)
boxplot(mu2_hat, 'Labels', string(alpha2_grid));
title('$\mu_2$ estimates', 'Interpreter','latex');
xlabel('$\alpha_2$ grid', 'Interpreter','latex');
ylabel('$\hat{\mu_2}$', 'Interpreter','latex');
yline(sigma_true,'--k',Label='True value')

grid on

subplot(2,3,5)
boxplot(sigma1_hat, 'Labels', string(alpha1_grid));
title('$\sigma_1$ estimates', 'Interpreter','latex');
xlabel('$\alpha_1$ grid', 'Interpreter','latex');
ylabel('$\hat{\sigma_1}$', 'Interpreter','latex');
yline(mu_true,'--k',Label='True value')
ylim([0,20])

grid on

subplot(2,3,6)
boxplot(sigma2_hat, 'Labels', string(alpha2_grid));
title('$\sigma_2$ estimates', 'Interpreter','latex');
xlabel('$\alpha_2$ grid', 'Interpreter','latex');
ylabel('$\hat{\sigma_2}$', 'Interpreter','latex');
yline(sigma_true,'--k',Label='True value')
grid on


%% c
r_data = importdata("DJIA30stockreturns.mat");
[nobs, nStocks] = size(r_data);

% Preallocate — now the variables exist!
pi_hat_data      = zeros(1, nStocks);
alpha1_hat_data  = zeros(1, nStocks);
alpha2_hat_data  = zeros(1, nStocks);
sigma1_hat_data  = zeros(1, nStocks);
sigma2_hat_data  = zeros(1, nStocks);
mu1_hat_data     = zeros(1, nStocks);
mu2_hat_data     = zeros(1, nStocks);

for j = 1:nStocks
    x = r_data(:, j);

    try
        [pi_hat_data(j), alpha1_hat_data(j), alpha2_hat_data(j), ...
         sigma1_hat_data(j), sigma2_hat_data(j), ...
         mu1_hat_data(j), mu2_hat_data(j)] = mle_symstable_di(x);

    catch
        % If it fails, we leave zeros.
        % That's exactly what you said you want.
    end
end

figure;
xgrid = linspace(-0.15, 0.15, 400)';

for i = 1:nStocks
    subplot(5,5,i);
    histogram(r_data(:,i), 40, 'Normalization','pdf', ...
              'DisplayStyle','stairs');
    hold on;

    % Only plot mixture if parameters are non-zero  
    if pi_hat_data(i) > 0 && ...
       alpha1_hat_data(i) > 0 && alpha2_hat_data(i) > 0 && ...
       sigma1_hat_data(i) > 0 && sigma2_hat_data(i) > 0

        pi_ = pi_hat_data(i);
        a1  = alpha1_hat_data(i);
        a2  = alpha2_hat_data(i);
        s1  = sigma1_hat_data(i);
        s2  = sigma2_hat_data(i);
        m1  = mu1_hat_data(i);
        m2  = mu2_hat_data(i);

        f1 = stablepdf_fft(xgrid, a1, 0, s1, m1);
        f2 = stablepdf_fft(xgrid, a2, 0, s2, m2);
        fmix = pi_ .* f1 + (1 - pi_) .* f2;

        plot(xgrid, fmix, 'LineWidth', 1.5);
        legend('Histogram','Mixture Fit','Location','best');
    else
        legend('Histogram','Location','best');
    end

    title(['Stock ', num2str(i)]);
    xlabel('Return');
    ylabel('Density');
    axis tight;
end

sgtitle('DJIA Stock Returns with 2-Component Symmetric Stable Mixture Fits');

%%
disp("mu1")
disp(mu1_hat_data)
disp("\n")
disp("mu2")
disp(mu2_hat_data)
disp("\n")
disp("sigma1")
disp(sigma1_hat_data)
disp("\n")
disp("sigma2")
disp(sigma2_hat_data)
disp("\n")

