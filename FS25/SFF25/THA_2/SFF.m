clc; clear; close all;

function [x, p] = runthefft(n, h, cf_handle)
    N = 2^n;
    x = (0:N-1)'*h - N*h/2;
    s = 1/(h*N);
    t = 2*pi*s*((0:N-1)' - N/2);
    sgn = ones(N, 1);
    sgn(2:2:N) = -1;
    CF = cf_handle(t);
    phi = sgn .* CF;
    phi(N/2+1) = sgn(N/2+1);
    p = s .* abs(fft(phi));
end

function pdf = pdf_fft(z, cf_handle)
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
    pdf = interp1(xgrd, bigpdf, z, 'linear', 0);
end

%% MLE
function nll = neglog_model(theta, x, model)
    params = model.trans(theta);
    if isfield(model, 'pdf')
        f = model.pdf(x, params);
    else
        cf_handle_for_params = @(t) model.cf(t, params);
        f = pdf_fft(x, cf_handle_for_params);
    end
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

%%
stockdata = importdata("DJIA30stockreturns.csv");
s = size(stockdata); n = s(1); k = s(2);
m = 1000;

%% Implementation A
function model = gaussian_exact()
    model.name = 'Gaussian (Exact)';
    model.trans = @(theta) [theta(1), exp(theta(2))];
    model.pdf = @(x, p) normpdf(x, p(1), p(2));
    model.guess_theta0 = @(x) [mean(x); log(std(x))];
end

% Numerical Implementation A
gauss_model = gaussian_exact();
x_grids_A = cell(k, 1);
all_params_A = cell(k, 1);
all_stats_A = cell(k, 1);
pdf_hat_A = cell(k, 1);

for j = 1:k
    x = stockdata(:, j);
    x_grid = linspace(min(x), max(x), m);
    x_grids_A{j} = x_grid;
    [params_hat, ~, stats] = mle_model(x, gauss_model);
    all_params_A{j} = params_hat;
    all_stats_A{j} = stats;
    pdf_hat_A{j} = gauss_model.pdf(x_grid, params_hat);
end

%% Implementation B
function model = model_mix_gauss_exact(K)
    function pdf_val = pdf_mix(x, p)
        pi_ = p(1:K);
        mu_ = p(K+1:2*K);
        sigma_ = p(2*K+1:3*K);
        pdf_val = zeros(size(x));
        for i = 1:K
            pdf_val = pdf_val + pi_(i) * normpdf(x, mu_(i), sigma_(i));
        end
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
            mus(1) = data_mean; sigmas(1) = data_std * 0.5;
            mus(2) = data_mean; sigmas(2) = data_std * 1.5;
        elseif K == 3
            mus(1) = data_mean; sigmas(1) = data_std * 0.5;
            mus(2) = data_mean - data_std; sigmas(2) = data_std * 1.5;
            mus(3) = data_mean + data_std; sigmas(3) = data_std * 1.5;
        else
            mus = ctranspose(linspace(data_mean - data_std, data_mean + data_std, K));
            sigmas = ones(K, 1) * data_std * 0.5;
        end
        tol = 1e-6;
        sigmas(sigmas < tol) = tol;
        theta0 = [theta_pi_init; mus; log(sigmas)];
    end
    
    model.name = ['Mixture Gaussian (K=', num2str(K), ')'];
    model.trans = @transform_mix_gauss;
    model.pdf = @pdf_mix;
    model.guess_theta0 = @initial_mix_gauss;
    model.K = K;
end

% Numerical Implementation B

mix2_model = model_mix_gauss_exact(2);
all_params_B2 = cell(k, 1);
all_stats_B2 = cell(k, 1);
pdf_hat_B2 = cell(k, 1);

for j = 1:k
    x = stockdata(:, j);
    x_grid = linspace(min(x), max(x), m);
    [params_hat, ~, stats] = mle_model(x, mix2_model);
    all_params_B2{j} = params_hat;
    all_stats_B2{j} = stats;
    pdf_hat_B2{j} = mix2_model.pdf(x_grid, params_hat);
end

mix3_model = model_mix_gauss_exact(3);
all_params_B3 = cell(k, 1);
all_stats_B3 = cell(k, 1);
pdf_hat_B3 = cell(k, 1);

for j = 1:k
    x = stockdata(:, j);
    x_grid = linspace(min(x), max(x), m);
    [params_hat, ~, stats] = mle_model(x, mix3_model);
    all_params_B3{j} = params_hat;
    all_stats_B3{j} = stats;
    pdf_hat_B3{j} = mix3_model.pdf(x_grid, params_hat);
end


%% Implementation C
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
                w_init(j) = w_mag;
            else
                w_init(j) = -w_mag;
            end
        end
        mu_init = data_mean - sum(w_init);
        theta0 = [mu_init; w_init];
    end
end


% Numerical Implementation C
chi_Ks = [2, 3, 4, 5];
pdf_hat_C_all = cell(k, 5);
all_params_C = cell(k, 5);
all_stats_C = cell(k, 5);

for K_val = chi_Ks
    model_C = chisq_sum(K_val);
    for j = 1:k
        x = stockdata(:, j);
        [params_hat, ~, stats] = mle_model(x, model_C);
        all_params_C{j, K_val} = params_hat;
        all_stats_C{j, K_val} = stats;
        
        cf_fit = @(t) model_C.cf(t, params_hat);
        pdf_vals = pdf_fft(x_grids_A{j}, cf_fit);
        pdf_hat_C_all{j, K_val} = pdf_vals;
    end
end


%% D
function model = model_nct_exact()
    model.name = 'NCT (Exact)';
    model.trans = @(theta) [theta(1); exp(theta(2)); exp(theta(3)); theta(4)];
    model.pdf = @pdf_nct_exact;
    model.guess_theta0 = @guess_nct;
    
    function f = pdf_nct_exact(x, p)
        mu = p(1);
        sig = p(2);
        nu = p(3);
        del = p(4);
        z = (x - mu) ./ sig;
        f = (1 ./ sig) .* nctpdf(z, nu, del);
    end

    function theta0 = guess_nct(x)
        mu_g = mean(x); sig_g = std(x); nu_g = 10; del_g = 0;
        theta0 = [mu_g; log(sig_g); log(nu_g); del_g];
    end
end

% Numerical Implementaition D
nct_mdl = model_nct_exact();
all_params_D = cell(k, 1);
all_stats_D = cell(k, 1);
pdf_hat_D = cell(k, 1);
hBar = waitbar(0, 'Fitting NCT Model...');

for j = 1:k
    waitbar((j-1)/k, hBar, sprintf('Fitting NCT: Stock %d of %d', j, k));
    x = stockdata(:, j);
    [params_hat, ~, stats] = mle_model(x, nct_mdl);
    all_params_D{j} = params_hat;
    all_stats_D{j} = stats;
    pdf_hat_D{j} = nct_mdl.pdf(x_grids_A{j}, params_hat);
end
waitbar(1, hBar, 'Done!');
pause(0.5);
close(hBar);

%% Save
save('results.mat', 'all_params_A', 'all_params_B2', 'all_params_B3', 'all_params_C', 'all_params_D', ...
                    'all_stats_A', 'all_stats_B2', 'all_stats_B3', 'all_stats_C', 'all_stats_D', ...
                    'pdf_hat_A', 'pdf_hat_B2', 'pdf_hat_B3', 'pdf_hat_C_all', 'pdf_hat_D');