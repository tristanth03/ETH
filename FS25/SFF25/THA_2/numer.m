%% Numerical results
clc;clear;close all;
stockdata = importdata("DJIA30stockreturns.csv");
s = size(stockdata); n = s(1); k = s(2);
m = 1000;
res = importdata("results.mat");

%%
x = stockdata(:,1);
x_grid = linspace(min(x), max(x), m);
[f_kde, x_kde] = ksdensity(x, x_grid);

c_blue = [0 0.4470 0.7410];
c_orange = [0.8500 0.3250 0.0980];
c_yellow = [0.9290 0.6940 0.1250];
c_purple = [0.4940 0.1840 0.5560];

figure; set(gcf,'Color','w');
lw = 2; fs = 18;
plot(x_kde, f_kde, 'k-', 'LineWidth', lw+0.5); 
hold on;
plot(x_grid, res.pdf_hat_A{1}, 'Color', c_blue, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_B2{1}, '--', 'Color', c_orange, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_B3{1}, ':', 'Color', c_yellow, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_D{1}, 'Color', c_purple, 'LineWidth', lw+0.5);
grid on;
xlabel('Log Return (%)', 'FontSize', fs);
ylabel('Density', 'FontSize', fs);
title('Gaussian / Mixture Gaussian / Non-central t (Full Range)', 'FontSize', 14);
legend({'Kernel density (true)','Gaussian','Gaussian 2-mixture','Gaussian 3-mixture','Non-central t'}, ...
        'Location','best','FontSize',fs);
set(gca,'FontSize',fs,'LineWidth',1);

figure; set(gcf,'Color','w');
lw = 2; fs = 12;
plot(x_kde, f_kde, 'k-', 'LineWidth', lw+0.5); hold on;
plot(x_grid, res.pdf_hat_A{1}, 'Color', c_blue, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_B2{1}, '--', 'Color', c_orange, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_B3{1}, ':', 'Color', c_yellow, 'LineWidth', lw);
plot(x_grid, res.pdf_hat_D{1}, 'Color', c_purple, 'LineWidth', lw+0.5);
xlim([4 10]);
grid on;
xlabel('Log Return (%)','FontSize',fs);
ylabel('Density','FontSize',fs);
title('Gaussian / Mixture Gaussian / Non-central t (Zoom: 4–10)', 'FontSize', 14);
legend({'Kernel density (true)','Gaussian','Gaussian 2-mixture','Gaussian 3-mixture','Non-central t'}, ...
        'Location','best','FontSize',fs);
set(gca,'FontSize',fs,'LineWidth',1);

figure; set(gcf, 'Color', 'w');
lw = 2; fs = 12;
colors = [ ...
    0.0000 0.4470 0.7410;  
    0.4660 0.6740 0.1880;   
    0.3010 0.7450 0.9330;   
    0.6350 0.0780 0.1840]; 

plot(x_kde, f_kde, 'k-', 'LineWidth', lw+0.5); 
hold on;
plot(x_grid, res.pdf_hat_C_all{1,2}, 'Color', colors(1,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,3}, 'Color', colors(2,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,4}, 'Color', colors(3,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,5}, 'Color', colors(4,:), 'LineWidth', lw);
grid on;
xlabel('Log Return (%)','FontSize',fs);
ylabel('Density','FontSize',fs);
title('Chi-square Sum Models C2–C5 (Full Range)', 'FontSize',14);
legend({'Kernel density (true)','Chi-Sq (2)','Chi-Sq (3)','Chi-Sq (4)','Chi-Sq (5)'}, ...
        'Location','best','FontSize',fs);
set(gca,'FontSize',fs,'LineWidth',1);

figure; set(gcf, 'Color', 'w');
lw = 2; fs = 12;

plot(x_kde, f_kde, 'k-', 'LineWidth', lw+0.5); hold on;
plot(x_grid, res.pdf_hat_C_all{1,2}, 'Color', colors(1,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,3}, 'Color', colors(2,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,4}, 'Color', colors(3,:), 'LineWidth', lw);
plot(x_grid, res.pdf_hat_C_all{1,5}, 'Color', colors(4,:), 'LineWidth', lw);
xlim([4 10]);
grid on;
xlabel('Log Return (%)','FontSize',fs);
ylabel('Density','FontSize',fs);
title('Chi-square Sum Models C2–C5 (Zoom: 4–10)', 'FontSize',14);
legend({'Kernel density (true)','Chi-Sq (2)','Chi-Sq (3)','Chi-Sq (4)','Chi-Sq (5)'}, ...
        'Location','best','FontSize',fs);
set(gca,'FontSize',fs,'LineWidth',1);


%%
models = {'Gaussian', 'MixG(2)', 'MixG(3)', 'Chi(2)', 'Chi(3)', 'Chi(4)', 'Chi(5)', 'NCT'};
metrics = {'BIC', 'AIC', 'NLL'};
fields = {'bic', 'aic', 'nll'};

fprintf('\n=== BY STOCK ===\n');

for j = 1:k
    stats = [ ...
        res.all_stats_A{j}, ...        % Gaussian
        res.all_stats_B2{j}, ...       % MixG(2)
        res.all_stats_B3{j}, ...       % MixG(3)
        res.all_stats_C{j,2}, ...      % Chi(2)
        res.all_stats_C{j,3}, ...      % Chi(3)
        res.all_stats_C{j,4}, ...      % Chi(4)
        res.all_stats_C{j,5}, ...      % Chi(5)
        res.all_stats_D{j}    ...      % NCT
    ];

    fprintf('\n%-10s', sprintf('Stock %d', j));
    for mm = 1:length(models)
        fprintf('%12s', models{mm});
    end
    fprintf('\n');
    for i = 1:length(metrics)
        fprintf('%-10s', metrics{i}); % Row Label
        
        for mm = 1:length(stats)
            val = stats(mm).(fields{i});
            fprintf('%12.2f', val);
        end
        fprintf('\n');
    end
        fprintf('%s\n', repmat('-', 1, 10 + 12*length(models))); 
end



%%
model_names = {'Gaussian', 'MixG(2)', 'MixG(3)', 'Chi(2)', 'Chi(3)', 'Chi(4)', 'Chi(5)', 'NCT'};
n_models = length(model_names);
var_table = zeros(k, 1 + n_models);
for j = 1:k
    x = stockdata(:, j);
    x_grid = linspace(min(x), max(x), m);
    var_table(j, 1) = prctile(x, 1);
    pdfs = cell(1, n_models);
    pdfs{1} = res.pdf_hat_A{j};
    pdfs{2} = res.pdf_hat_B2{j};
    pdfs{3} = res.pdf_hat_B3{j};
    pdfs{4} = res.pdf_hat_C_all{j, 2};
    pdfs{5} = res.pdf_hat_C_all{j, 3};
    pdfs{6} = res.pdf_hat_C_all{j, 4};
    pdfs{7} = res.pdf_hat_C_all{j, 5};
    pdfs{8} = res.pdf_hat_D{j};
    for i = 1:n_models
        pdf_vals = pdfs{i};
        cdf_vals = cumtrapz(x_grid, pdf_vals);
        if cdf_vals(end) > 0
            cdf_vals = cdf_vals / cdf_vals(end);
        end
        try
            [unique_cdf, idx] = unique(cdf_vals);
            unique_x = x_grid(idx);
            estimated_var = interp1(unique_cdf, unique_x, 0.01, 'pchip');
        catch
            estimated_var = NaN;
        end
        if isnan(estimated_var)
            estimated_var = min(x);
        end
        var_table(j, i+1) = estimated_var;
    end
end
var_names = [{'Empirical'}, model_names];
valid_var_names = matlab.lang.makeValidName(var_names);
T = array2table(var_table, 'VariableNames', valid_var_names);
row_names = arrayfun(@(x) sprintf('S%d', x), 1:k, 'UniformOutput', false);
T.Properties.RowNames = row_names;
disp(T);

empirical_vals = var_table(:, 1);
model_vals = var_table(:, 2:end);
mse_vals = mean((model_vals - empirical_vals).^2, 1);

fprintf('\nMSE of 1%% VaR across models:\n');
for i = 1:n_models
    fprintf('%s : mse = %.4e\n', model_names{i}, mse_vals(i));
end

%%
params_A = cell2mat(res.all_params_A); figure('Name', 'Gaussian');
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Gaussian Model', 'Interpreter', 'latex', 'FontSize', 18);
nexttile; boxplot(params_A(:, 1));
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18);
xticklabels({'$\mu$'}); grid on;nexttile;
boxplot(params_A(:, 2)); set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18);
xticklabels({'$\sigma$'}); grid on; mix_Ks = [2, 3];
for K = mix_Ks
    if K == 2 raw_B = res.all_params_B2; model_name = 'MixG(2)';
    else
        raw_B = res.all_params_B3; model_name = 'MixG(3)'; end
    mat_B = zeros(length(raw_B), 3*K);
    for i = 1:length(raw_B)
        mat_B(i, :) = ctranspose(raw_B{i}(:)); end
    labels_B = cell(1, 3*K); idx_lbl = 1;
    for i = 1:K, labels_B{idx_lbl} = sprintf('$\\pi_{%d}$',i); idx_lbl = idx_lbl + 1; end
    for i = 1:K, labels_B{idx_lbl} = sprintf('$\\mu_{%d}$',i); idx_lbl = idx_lbl + 1; end
    for i = 1:K, labels_B{idx_lbl} = sprintf('$\\sigma_{%d}$',i); idx_lbl = idx_lbl + 1; end
    figure('Name', model_name);
    t = tiledlayout(K, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Mixture Gaussian (K=%d)', K), 'Interpreter', 'latex', 'FontSize', 18);
    for i = 1:size(mat_B, 2)
        nexttile; boxplot(mat_B(:, i));
        set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18);
        xticklabels(labels_B(i)); grid on; end
end
chi_Ks = [2, 3, 4, 5];
for idx = 1:length(chi_Ks)
    K = chi_Ks(idx);
    raw_C = res.all_params_C(:, K); mat_C = zeros(length(raw_C), K+1);
    for i = 1:length(raw_C)
        mat_C(i, :) = ctranspose(raw_C{i}(:)); end
    labels_C = cell(1, K+1); labels_C{1} = '$\mu$';
    for i = 1:K
        labels_C{i+1} = sprintf('$w_{%d}$', i); end
    N_params = K + 1;
    n_cols = ceil(sqrt(N_params));
    n_rows = ceil(N_params / n_cols);
    figure('Name', sprintf('Chi(%d)', K));
    t = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Chi-Square Sum (K=%d)', K), 'Interpreter','latex','FontSize', 18);
    for i = 1:N_params
        nexttile; boxplot(mat_C(:, i));
        set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18);
        xticklabels(labels_C(i)); grid on; end 
end
raw_D = res.all_params_D; mat_D = zeros(length(raw_D), 4);
for i = 1:length(raw_D)
    mat_D(i, :) = ctranspose(raw_D{i}(:));
end
labels_D = {'$\mu$', '$\sigma$', '$\nu$', '$\delta$'}; figure('Name', 'NCT');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'NCT Model', 'Interpreter', 'latex', 'FontSize', 18);
for i = 1:4
    nexttile; boxplot(mat_D(:, i));
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18);
    xticklabels(labels_D(i)); grid on;
end