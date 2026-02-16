%% FFT
clc;clear;close all;
function [x,p]=runthefft(n,h,cf_handle) 
    %%% Updated from take home assignment 1
    % allow a cf handle
    N=2^n;
    x=(0:N-1)'*h - N*h/2;
    s=1/(h*N);
    t=2*pi*s*((0:N-1)' - N/2);
    sgn=ones(N,1);
    sgn(2:2:N)=-1;

    CF = cf_handle(t);
    phi = sgn .* CF;
    phi(N/2+1) = sgn(N/2+1);
    p = s .* abs(fft(phi));
end

function pdf = pdf_fft(z,cf_handle) 
    pmax = 18;
    step = 0.01;
    p = 14;
    maxz = round(max(abs(z))) + 5;
    while ((maxz/step + 1) > 2^(p-1)), p = p + 1; end
    if p > pmax, p = pmax; end
    if maxz/step + 1 > 2^(p-1)
        step = (maxz + 1)*1.001 / (2^(p-1));
    end
    [xgrd, bigpdf] = runthefft(p, step, cf_handle);
    pdf = interp1(xgrd, bigpdf, z, 'linear');
end

function fS_res = num_int(integrand, s, x) 
    %%% Updated from take home assignment 1
    % (vectorized version)

    % Compute all values at once 
    val_matrix = integrand(s); 
    
    % Integrate along dimension 2 
    fS_res = trapz(x, val_matrix, 2);
end


%% MLE
function nll = neglog_model(theta, x, model)
    params = model.trans(theta);
    cf_handle_for_params = @(t) model.cf(t, params);
    f = pdf_fft(x, cf_handle_for_params);
    tol = 1e-16;
    f(f < tol) = tol;
    nll = -sum(log(f));
end

function [params_hat, theta_hat, fit_stats] = mle_model(x, model)
    neglog = @(theta) neglog_model(theta, x, model);
    
    theta0 = model.guess_theta0(x);
    
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
        'Display', 'off', 'MaxIterations', 300, 'MaxFunctionEvaluations', 400);

    [theta_hat, nll] = fminunc(neglog, theta0, options);
    
    params_hat = model.trans(theta_hat);
    
    k = numel(theta_hat);
    n = numel(x);
    fit_stats.nll = nll;
    fit_stats.aic = 2*k + 2*nll;
    fit_stats.bic = k*log(n) + 2*nll;
end


%% Data
stockdata = importdata("DJIA30stockreturns.csv");
s = size(stockdata); n = s(1); k = s(2);

m = 1000; % gridsize (used globally for other parts)
%% Implementation A

function model = gaussian_single()
    model.name = 'Gaussian';
    model.trans = @(theta) [theta(1), exp(theta(2))];
    model.pdf = @(x, p) normpdf(x, p(1), p(2));
    model.guess_theta0 = @(x) [mean(x); log(std(x))];
end

% Numerical implementation

gauss_model = gaussian_single();
x_grids_A = cell(k, 1);
all_params_A = cell(k, 1);
all_stats_A = cell(k, 1);
pdf_hat_A = cell(k, 1);

for j = 1:k
    x = stockdata(:, j);
    x_grid = linspace(min(x), max(x), m);
    x_grids_A{j} = x_grid;
    [params_hat, ~, stats] = mle_model_std(x, gauss_model);
    all_params_A{j} = params_hat;
    all_stats_A{j} = stats;
    pdf_hat_A{j} = gauss_model.pdf(x_grid, params_hat);
end


%% Implementation B
function model = model_mix_gauss(K)
    
    function Phi = cf_mix_gauss(t, p)
        pi_ = p(1:K);
        mu_ = p(K+1:2*K);
        sigma_ = p(2*K+1:3*K);
    
        t_col = t(:);
        t_expanded = repmat(t_col, 1, K);
        exponents = 1i * t_expanded .* repmat(mu_', size(t, 1), 1) - ...
                    0.5 * (t_expanded .* repmat(sigma_', size(t, 1), 1)).^2;
        
        component_CFs = exp(exponents);
        Phi = component_CFs * pi_;
    end
    
    function p = transform_mix_gauss(theta)
        theta_pi = [theta(1:K-1); 0];
        pi_ = exp(theta_pi) ./ sum(exp(theta_pi));
        
        mu = theta(K:2*K-1);
        
        sigma = exp(theta(2*K:3*K-1)); 
        
        p = [pi_; mu; sigma];
    end
    
    function theta0 = initial_mix_gauss(x)
        theta_pi_init = zeros(K-1, 1);
        
        data_mean = mean(x);
        data_std = std(x);
        
        mus = zeros(K, 1);
        sigmas = zeros(K, 1);
        
        if K == 2
            % Central (Low Vol)
            mus(1) = data_mean;
            sigmas(1) = data_std * 0.5; 
            % Wider (High Vol)
            mus(2) = data_mean;
            sigmas(2) = data_std * 1.5;
        
        elseif K == 3
            % Central Peak (Low Vol)
            mus(1) = data_mean;
            sigmas(1) = data_std * 0.5;
            % Negative Tail
            mus(2) = data_mean - data_std; 
            sigmas(2) = data_std * 1.5;
            % Positive Tail
            mus(3) = data_mean + data_std;
            sigmas(3) = data_std * 1.5;

        else % Fallback for other K
            % (Not optimized)
            mus = linspace(data_mean - data_std, data_mean + data_std, K)';
            sigmas = ones(K, 1) * data_std * 0.5;
        end
        tol = 1e-16;
        sigmas(sigmas < tol) = tol; 
        theta0 = [theta_pi_init; mus; log(sigmas)];
    end
    model.name = ['Mixture Gaussian (K=', num2str(K), ')'];
    model.trans = @transform_mix_gauss;
    model.cf = @cf_mix_gauss;
    model.guess_theta0 = @initial_mix_gauss;
    model.K = K;
end

% Numerical implementation

% k = 2
mix2_model = model_mix_gauss(2);
all_params_B2 = cell(k,1);
all_stats_B2 = cell(k,1);
pdf_hat_B2 = cell(k,1);

for j = 1:k
    x = stockdata(:,j);
    x_grid = linspace(min(x), max(x), m);
    [params_hat,~,stats] = mle_model(x,mix2_model);
    all_params_B2{j} = params_hat;
    all_stats_B2{j} = stats;
    cf_fit = @(t) mix2_model.cf(t, params_hat);
    pdf_hat_B2{j} = pdf_fft(x_grid, cf_fit);
end

% k = 3
mix3_model = model_mix_gauss(3);
all_params_B3 = cell(k,1);
all_stats_B3 = cell(k,1);
pdf_hat_B3 = cell(k,1);

for j = 1:k
    x = stockdata(:,j);
    x_grid = linspace(min(x), max(x), m);
    [params_hat,~,stats] = mle_model(x,mix3_model);
    all_params_B3{j} = params_hat;
    all_stats_B3{j} = stats;
    cf_fit = @(t) mix3_model.cf(t, params_hat);
    pdf_hat_B3{j} = pdf_fft(x_grid, cf_fit);
end



%% Implemntation C
function model = chisq_sum(K)
    model.name = ['Chi-Square Sum (K=', num2str(K), ')'];
    model.K = K;
    model.trans = @(theta) theta;

    model.cf = @cf_chisq_sum;
    function Phi = cf_chisq_sum(t, p)
        mu = p(1);
        w = p(2:end);
        
        term_loc = exp(1i * mu * t);
        term_chi = ones(size(t));
        for j = 1:length(w)
            term_chi = term_chi .* (1 - 2i * w(j) * t).^(-0.5);
        end
        Phi = term_loc .* term_chi;
    end

    model.guess_theta0 = @guess_chisq_init;
    function theta0 = guess_chisq_init(x)
        
        data_mean = mean(x);
        data_var = var(x);
        
        w_init = zeros(K, 1);
        target_w_sq = (data_var / 2) / K; 
        w_mag = sqrt(target_w_sq);
        
        for j = 1:K
            if mod(j, 2) == 1
                w_init(j) = w_mag; % Positive weight
            else
                w_init(j) = -w_mag; % Negative weight
            end
        end
        
        mu_init = data_mean - sum(w_init);
        theta0 = [mu_init; w_init];
    end
end

% Numerical Implementation
chi_Ks = [2, 3, 4, 5];
pdf_hat_C_all = cell(k, 5); 
all_params_C = cell(k, 5);
all_stats_C  = cell(k, 5);

for K_val = chi_Ks
    model_C = chisq_sum(K_val);
    
    for j = 1:k
        x = stockdata(:,j);        
        [params_hat, ~, stats] = mle_model(x, model_C);
        
        all_params_C{j, K_val} = params_hat;
        all_stats_C{j, K_val} = stats;
        
        cf_fit = @(t) model_C.cf(t, params_hat);
        pdf_vals = pdf_fft(x_grids_A{j}, cf_fit);
        
        pdf_hat_C_all{j, K_val} = pdf_vals;
    end
end

%% Implementation D
function model = model_nct(m)
    model.name = 'NCT (FFT)';
    model.trans = @(theta) [theta(1); exp(theta(2)); exp(theta(3)); theta(4)];
    model.cf = @cf_nct_vectorized;
    model.guess_theta0 = @guess_nct;

    function phi = cf_nct_vectorized(t, p)
        mu = p(1);
        sig = p(2);
        nu = p(3);
        del = p(4);
        
        std_v = sqrt(2*nu);
        v_min = max(1e-3, nu - 8*std_v); 
        v_max = nu + 12*std_v;
        if v_max < 50, v_max = 50; end
        
        v_grid = linspace(v_min, v_max, m); 
        integrand_func = @(t_vec) integrand_nct_matrix(v_grid, t_vec*sig, nu, del);
        phi_std = num_int(integrand_func, t, v_grid);
        
        % Add location shift
        phi = exp(1i * t * mu) .* phi_std;
    end

    function val_matrix = integrand_nct_matrix(v, t_scaled, nu, del)      
        log_pdf_v = (nu/2-1).*log(v) - v/2 - (nu/2)*log(2) - gammaln(nu/2);
        pdf_v = exp(log_pdf_v);
        
        sqrt_nu_v = sqrt(nu ./ v);   %
        nu_v = nu ./ v;              
        
        term1 = (1i *del) .* t_scaled .* sqrt_nu_v; 
        term2 = (-0.5) .* (t_scaled.^2) .* nu_v;
        
        term_cf = exp(term1 + term2);
        val_matrix = term_cf .* pdf_v;
        val_matrix(isnan(val_matrix)) = 0; 
    end

    function theta0 = guess_nct(x)
        mu_g = mean(x); sig_g = std(x); nu_g = 10; del_g = 0; 
        theta0 = [mu_g; log(sig_g); log(nu_g); del_g];
    end
end

% Numerical Implementation

nct_mdl = model_nct(m);
all_params_D = cell(k,1); 
all_stats_D = cell(k,1); 
pdf_hat_D = cell(k,1);

hBar = waitbar(0, 'Initializing NCT Model Fit...');
for j = 1:k
    waitbar((j-1)/k, hBar, sprintf('Fitting NCT: Stock %d of %d', j, k));
    x = stockdata(:,j);
    [params_hat, ~, stats] = mle_model(x, nct_mdl);
    
    all_params_D{j} = params_hat;
    all_stats_D{j} = stats;
    pdf_hat_D{j} = pdf_fft(x_grids_A{j}, @(t) nct_mdl.cf(t, params_hat));
end

waitbar(1, hBar, 'Done!');
pause(0.5);
close(hBar);


%%
save('results.mat', 'all_params_A', 'all_params_B2', 'all_params_B3', 'all_params_C', 'all_params_D', ...
                    'all_stats_A', 'all_stats_B2', 'all_stats_B3', 'all_stats_C', 'all_stats_D', ...
                    'pdf_hat_A', 'pdf_hat_B2', 'pdf_hat_B3', 'pdf_hat_C_all', 'pdf_hat_D');

%%%%%%%%%%

%%

%% Plotting
figure('Color', 'w');
% 1. Plot the Gaussian (A)
plot(x_grids_A{2}, pdf_hat_A{2}, 'r', 'LineWidth', 1.5, 'DisplayName', 'Gaussian');
hold on;

% 2. Plot Mixtures (B)
plot(x_grids_A{2}, pdf_hat_B2{2}, 'b', 'LineWidth', 1.5, 'DisplayName', 'Mix Gaussian (K=2)');
plot(x_grids_A{2}, pdf_hat_B3{2}, 'k', 'LineWidth', 1.5, 'DisplayName', 'Mix Gaussian (K=3)');

% 3. Plot Chi-Squares (C)
plot(x_grids_A{2}, pdf_hat_C_all{2,2}, '--r', 'LineWidth', 1.5, 'DisplayName', 'Chi-Sq Sum (K=2)');
plot(x_grids_A{2}, pdf_hat_C_all{2,3}, '--k', 'LineWidth', 1.5, 'DisplayName', 'Chi-Sq Sum (K=3)');
plot(x_grids_A{2}, pdf_hat_C_all{2,4}, '--b', 'LineWidth', 1.5, 'DisplayName', 'Chi-Sq Sum (K=4)');
plot(x_grids_A{2}, pdf_hat_C_all{2,5}, '--g', 'LineWidth', 1.5, 'DisplayName', 'Chi-Sq Sum (K=5)');

% 4. Plot NCT (D)
plot(x_grids_A{2}, pdf_hat_D{2}, 'm-', 'LineWidth', 2, 'DisplayName', 'NCT');

% Formatting
legend('show', 'Location', 'best');
title('Fitted Densities (Stock 1)');
xlabel('Returns');
ylabel('PDF');
grid on;
hold off;

%% Detailed Model Stats Table
% Define the model names for the header columns
models = {'Gaussian', 'MixG(2)', 'MixG(3)', 'Chi(2)', 'Chi(3)', 'Chi(4)', 'Chi(5)', 'NCT'};
metrics = {'BIC', 'AIC', 'NLL'};
fields = {'bic', 'aic', 'nll'};

fprintf('\n=== DETAILED MODEL STATS BY STOCK ===\n');

for j = 1:k
    % 1. Collect all stats for the current stock j
    stats = [ ...
        all_stats_A{j}, ...        % Gaussian
        all_stats_B2{j}, ...       % MixG(2)
        all_stats_B3{j}, ...       % MixG(3)
        all_stats_C{j,2}, ...      % Chi(2)
        all_stats_C{j,3}, ...      % Chi(3)
        all_stats_C{j,4}, ...      % Chi(4)
        all_stats_C{j,5}, ...      % Chi(5)
        all_stats_D{j}    ...      % NCT
    ];

    % 2. Print Header Row (Stock Number + Model Names)
    fprintf('\n%-10s', sprintf('Stock %d', j));
    for m = 1:length(models)
        fprintf('%12s', models{m});
    end
    fprintf('\n');
    
    % 3. Print Data Rows (BIC, AIC, NLL)
    for i = 1:length(metrics)
        fprintf('%-10s', metrics{i}); % Row Label
        
        for m = 1:length(stats)
            val = stats(m).(fields{i});
            fprintf('%12.2f', val);
        end
        fprintf('\n');
    end
    
    % Separator line
    fprintf('%s\n', repmat('-', 1, 10 + 12*length(models))); 
end

