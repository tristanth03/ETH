function main_driver()
    clc; clear; close all;
    
    stockdata = importdata('DJIA30stockreturns.csv');
    if isstruct(stockdata)
        data = stockdata.data;
    else
        data = stockdata;
    end
    
    [n_obs, n_stocks] = size(data);
    
    % -------------------------------------------------------------------------
    % 1. Setup Models
    % -------------------------------------------------------------------------
    models = {};
    
    models{end+1} = model_gaussian();
    
    models{end+1} = model_mix_gauss(2);
    models{end+1} = model_mix_gauss(3);
    
    models{end+1} = model_chisq_sum(2);
    models{end+1} = model_chisq_sum(3);
    models{end+1} = model_chisq_sum(4);
    models{end+1} = model_chisq_sum(5);
    
    models{end+1} = model_nct(); 
    
    num_models = length(models);
    
    % -------------------------------------------------------------------------
    % 2. Estimation Loop
    % -------------------------------------------------------------------------
    results = struct('nll', {}, 'aic', {}, 'bic', {}, 'params', {}, 'var1', {});
    
    fprintf('Estimating models for %d stocks...\n', n_stocks);
    
    for i = 1:n_stocks
        r = data(:, i);
        fprintf('Processing Stock %d...\n', i);
        
        for m = 1:num_models
            mdl = models{m};
            [p_hat, ~, stats] = mle_model(r, mdl);
            
            results(i, m).nll = stats.nll;
            results(i, m).aic = stats.aic;
            results(i, m).bic = stats.bic;
            results(i, m).params = p_hat;
            
            results(i, m).var1 = compute_var_1pct(mdl, p_hat);
        end
    end
    
    % -------------------------------------------------------------------------
    % 3. Plotting (First Stock)
    % -------------------------------------------------------------------------
    r1 = data(:, 1);
    x_grid = linspace(min(r1)*1.2, max(r1)*1.2, 1000);
    
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    [f_kde, xi_kde] = ksdensity(r1);
    plot(xi_kde, f_kde, 'k-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');
    
    colors = lines(num_models);
    for m = 1:num_models
        mdl = models{m};
        p_hat = results(1, m).params;
        
        y_vals = zeros(size(x_grid));
        
        if strcmp(mdl.type, 'fft')
             cf_fun = @(t) mdl.cf(t, p_hat);
             y_vals = pdf_fft(x_grid, cf_fun);
        elseif strcmp(mdl.type, 'analytical')
             y_vals = mdl.pdf(x_grid, p_hat);
        end
        
        plot(x_grid, y_vals, '--', 'Color', colors(m,:), 'LineWidth', 1.5, ...
             'DisplayName', mdl.name);
    end
    
    title('Fitted Densities vs Kernel Density (Stock 1)');
    xlabel('Returns');
    ylabel('Density');
    legend('show', 'Location', 'best');
    grid on;
    hold off;
    
    % -------------------------------------------------------------------------
    % 4. Reporting: Best Models (LaTeX)
    % -------------------------------------------------------------------------
    model_names = cell(1, num_models);
    for m = 1:num_models
        model_names{m} = models{m}.name;
    end
    
    generate_best_fit_table(results, model_names, n_stocks);
    
    % -------------------------------------------------------------------------
    % 5. Reporting: VaR Table (LaTeX)
    % -------------------------------------------------------------------------
    empirical_var = quantile(data, 0.01); 
    generate_var_table(results, empirical_var, model_names, n_stocks);
    
end

% -------------------------------------------------------------------------
% Core Logic: MLE
% -------------------------------------------------------------------------
function [params_hat, theta_hat, fit_stats] = mle_model(x, model)
    neglog = @(theta) neglog_wrapper(theta, x, model);
    theta0 = model.guess_theta0(x);
    
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
        'Display', 'off', 'MaxIterations', 2000, 'MaxFunctionEvaluations', 20000);
    
    try
        [theta_hat, nll] = fminunc(neglog, theta0, options);
    catch
        theta_hat = theta0;
        nll = neglog(theta0);
    end
    
    params_hat = model.trans(theta_hat);
    
    k = numel(theta_hat);
    n = numel(x);
    fit_stats.nll = nll;
    fit_stats.aic = 2*k + 2*nll;
    fit_stats.bic = k*log(n) + 2*nll;
end

function nll = neglog_wrapper(theta, x, model)
    params = model.trans(theta);
    
    if strcmp(model.type, 'fft')
        cf_handle = @(t) model.cf(t, params);
        f = pdf_fft(x, cf_handle);
    else
        f = model.pdf(x, params);
    end
    
    tol = 1e-100; 
    f(f < tol) = tol;
    f(isnan(f)) = tol;
    nll = -sum(log(f));
    
    if isnan(nll) || isinf(nll)
        nll = 1e10;
    end
end

% -------------------------------------------------------------------------
% VaR Computation (1%)
% -------------------------------------------------------------------------
function var_val = compute_var_1pct(model, params)
    alpha = 0.01;
    
    if strcmp(model.name, 'Gaussian')
        mu = params(1);
        sig = params(2);
        var_val = norminv(alpha, mu, sig);
        
    elseif contains(model.name, 'NCT')
        mu = params(1);
        sig = params(2);
        nu = params(3);
        delta = params(4);
        var_val = mu + sig * nctinv(alpha, nu, delta);
        
    else
        % FFT Based (Mixtures, ChiSq) - Solve CDF(x) = alpha
        % Create a wide grid, compute PDF, integrate to CDF
        L = 1000; 
        % Heuristic range for returns
        x_min = -0.5; 
        x_max = 0.5;
        
        cf_handle = @(t) model.cf(t, params);
        
        % Coarse search to find range
        x_search = linspace(x_min, x_max, 4096);
        pdf_vals = pdf_fft(x_search, cf_handle);
        dx = x_search(2) - x_search(1);
        cdf_vals = cumsum(pdf_vals) * dx;
        
        % Normalize total mass to 1 (fix numerical leak)
        cdf_vals = cdf_vals / cdf_vals(end);
        
        [~, idx] = min(abs(cdf_vals - alpha));
        var_val = x_search(idx);
    end
end

% -------------------------------------------------------------------------
% FFT Utilities
% -------------------------------------------------------------------------
function [x,p] = runthefft(n, h, cf_handle)
    N = 2^n;
    x = (0:N-1)'*h - N*h/2;
    s = 1/(h*N);
    t = 2*pi*s*((0:N-1)' - N/2);
    sgn = ones(N,1);
    sgn(2:2:N) = -1;
    CF = cf_handle(t);
    phi = sgn .* CF;
    phi(N/2+1) = sgn(N/2+1);
    p = s .* abs(fft(phi));
end

function pdf = pdf_fft(z, cf_handle)
    pmax = 20;
    step = 0.001; 
    p = 14;
    maxz = max(abs(z)) + 1; 
    
    while ((maxz/step + 1) > 2^(p-1))
        p = p + 1;
    end
    if p > pmax
        p = pmax;
    end
    
    % Re-adjust step if bounds exceeded
    if (maxz/step + 1) > 2^(p-1)
       step = (maxz + 1)*1.001 / (2^(p-1));
    end
    
    [xgrd, bigpdf] = runthefft(p, step, cf_handle);
    pdf = interp1(xgrd, bigpdf, z, 'linear', 0);
end

% -------------------------------------------------------------------------
% Model Definitions
% -------------------------------------------------------------------------

% 1. Gaussian
function m = model_gaussian()
    m.name = 'Gaussian';
    m.type = 'fft'; % Using FFT as requested for consistent interface
    m.trans = @(theta) [theta(1), exp(theta(2))];
    m.cf = @(t,p) exp(1i * p(1) * t - 0.5 * (p(2) * t).^2);
    m.guess_theta0 = @(x) [mean(x); log(std(x))];
end

% 2. Mixture of Gaussians
function m = model_mix_gauss(K)
    m.name = sprintf('MixGauss %d', K);
    m.type = 'fft';
    m.K = K;
    
    m.trans = @(theta) transform_mix(theta, K);
    m.cf = @(t,p) cf_mix(t, p, K);
    m.guess_theta0 = @(x) guess_mix(x, K);
end

function p = transform_mix(theta, K)
    w_unconstrained = theta(1:K-1);
    w_rest = [w_unconstrained; 0];
    pi_ = exp(w_rest) ./ sum(exp(w_rest));
    
    mu = theta(K : 2*K - 1);
    sigma = exp(theta(2*K : 3*K - 1));
    p = [pi_; mu; sigma];
end

function Phi = cf_mix(t, p, K)
    pi_ = p(1:K);
    mu_ = p(K+1:2*K);
    sig_ = p(2*K+1:3*K);
    Phi = zeros(size(t));
    for k=1:K
        term = pi_(k) .* exp(1i*mu_(k)*t - 0.5*(sig_(k)*t).^2);
        Phi = Phi + term;
    end
end

function theta0 = guess_mix(x, K)
    mx = mean(x); sx = std(x);
    w_init = zeros(K-1,1);
    mu_init = linspace(mx - sx, mx + sx, K)';
    sig_init = log(repmat(sx, K, 1) * 0.6);
    theta0 = [w_init; mu_init; sig_init];
end

% 3. Weighted Chi-Square Sum
function m = model_chisq_sum(K)
    m.name = sprintf('ChiSq Sum %d', K);
    m.type = 'fft';
    m.K = K;
    
    % Params: Loc (1) + Weights (K)
    m.trans = @(theta) theta; % No transformation needed
    m.cf = @(t,p) cf_chisq(t, p, K);
    m.guess_theta0 = @(x) guess_chisq(x, K);
end

function Phi = cf_chisq(t, p, K)
    mu = p(1);
    w = p(2:end);
    
    % CF of w*Chi2(1) is (1 - 2*i*w*t)^(-0.5)
    % CF of Sum + mu is exp(i*mu*t) * Prod(...)
    
    term_prod = ones(size(t));
    for j=1:K
        % Use complex power carefully
        base = 1 - 2i * w(j) * t;
        term_prod = term_prod .* (base .^ (-0.5));
    end
    Phi = exp(1i * mu * t) .* term_prod;
end

function theta0 = guess_chisq(x, K)
    % Heuristic initialization
    % To span negative and positive returns, we need mixed signs in weights.
    mu_guess = mean(x); 
    sx = std(x);
    
    % Split weights positive and negative
    w_guess = randn(K, 1) * (sx / sqrt(K));
    % Ensure at least one negative and one positive to allow 2-sided tails
    if K >= 2
        w_guess(1) = abs(w_guess(1));
        w_guess(2) = -abs(w_guess(2));
    end
    
    theta0 = [mu_guess; w_guess];
end

% 4. Location-Scale Non-Central t
function m = model_nct()
    m.name = 'NCT';
    m.type = 'analytical';
    
    m.trans = @(theta) [theta(1); exp(theta(2)); exp(theta(3)); theta(4)];
    m.pdf = @pdf_nct;
    m.guess_theta0 = @guess_nct;
end

function y = pdf_nct(x, p)
    mu = p(1);
    sig = p(2);
    nu = p(3);
    delta = p(4);
    
    z = (x - mu) ./ sig;
    y = (1/sig) * nctpdf(z, nu, delta);
end

function theta0 = guess_nct(x)
    % mu, log(sigma), log(nu), delta
    % Initialize nu around 5 (fat tails), delta near 0
    theta0 = [mean(x); log(std(x)); log(10); 0.0];
end

% -------------------------------------------------------------------------
% Output Generators
% -------------------------------------------------------------------------
function generate_best_fit_table(results, model_names, n_stocks)
    fprintf('\n\n%% --- Best Fitting Models ---\n');
    fprintf('\\begin{table}[h!]\n\\centering\n');
    fprintf('\\begin{tabular}{l c c c}\n\\hline\n');
    fprintf('Stock & Best (Likelihood) & Best (AIC) & Best (BIC) \\\\ \\hline\n');
    
    for i = 1:n_stocks
        row = results(i, :);
        nlls = [row.nll];
        aics = [row.aic];
        bics = [row.bic];
        
        [~, iL] = min(nlls);
        [~, iA] = min(aics);
        [~, iB] = min(bics);
        
        fprintf('Stock %d & %s & %s & %s \\\\ \n', ...
            i, model_names{iL}, model_names{iA}, model_names{iB});
    end
    fprintf('\\hline\n\\end{tabular}\n\\caption{Best fitting models per stock.}\n\\end{table}\n');
end

function generate_var_table(results, emp_var, model_names, n_stocks)
    fprintf('\n\n%% --- 1%% VaR Comparison ---\n');
    fprintf('\\begin{table}[h!]\n\\centering\n');
    fprintf('\\small\n');
    
    % Header
    header = 'Stock & Empirical';
    for m = 1:length(model_names)
        header = [header, ' & ', model_names{m}];
    end
    header = [header, ' \\\\ \\hline'];
    fprintf('\\begin{tabular}{l c | c c c c c c c c}\n\\hline\n');
    fprintf('%s\n', header);
    
    for i = 1:n_stocks
        ev = emp_var(i);
        fprintf('%d & %.4f', i, ev);
        
        % Collect model VaRs
        row_vars = zeros(1, length(model_names));
        for m = 1:length(model_names)
            row_vars(m) = results(i, m).var1;
        end
        
        % Find closest
        diffs = abs(row_vars - ev);
        [~, best_idx] = min(diffs);
        
        for m = 1:length(model_names)
            val = row_vars(m);
            if m == best_idx
                fprintf(' & \\textbf{%.4f}', val);
            else
                fprintf(' & %.4f', val);
            end
        end
        fprintf(' \\\\ \n');
    end
    
    fprintf('\\hline\n\\end{tabular}\n\\caption{1\\%% Value at Risk estimates. Bold indicates closest to empirical.}\n\\end{table}\n');
end