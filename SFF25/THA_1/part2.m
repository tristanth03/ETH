clc;clear;close all

alpha1_grid = linspace(1.2,1.8,6);
alpha2 = 1.8;
n = [1000,10000];
K = length(n);
m = length(alpha1_grid);
nsim = 500;

alpha_hat = zeros(nsim, m, K);
beta_hat  = zeros(nsim, m, K);
sigma_hat = zeros(nsim, m, K);
mu_hat    = zeros(nsim, m, K);

pd2 = makedist('Stable','alpha',alpha2,'beta',0,'gam',1,'delta',0);

total_iterations = m * K * nsim;
counter = 0;
B = 500;
CI_length = zeros(nsim, m, K);
for j = 1:m
    pd1 = makedist('Stable','alpha',alpha1_grid(j),'beta',0,'gam',1,'delta',0);
    for i = 1:K
        for sim = 1:nsim
            X1 = random(pd1,n(i),1);
            X2 = random(pd2,n(i),1);
            S = X1+X2;
    
            [alpha_hat(sim,j,i),beta_hat(sim,j,i),sigma_hat(sim,j,i),mu_hat(sim,j,i)] = stableregkw(S);

            % similar to the code in 
            % "APPLICATION OF THE NONPARAMETRIC BOOTSTRAP, p334-335,
            % Fundamental Statistical Inference"
            boot_alpha = zeros(B,1);
            for b = 1:B
                boot_S = S( randi(n(i), n(i), 1) );  
                boot_alpha(b) = stableregkw(boot_S); 
            end

            lb = quantile(boot_alpha, 0.05);
            ub = quantile(boot_alpha, 0.95);
            CI_length(sim,j,i) = ub - lb;

            counter = counter + 1;

            if mod(counter, 100) == 0  
                fprintf('Progress: %5.1f%%   (alpha1 = %.2f, n = %d, sim = %d/%d)\n', ...
                    100 * counter / total_iterations, alpha1_grid(j), n(i), sim, nsim);
            end

        end
    end
end
%%
for i = 1:K
    figure;
    t = tiledlayout(2, 2, ...
        'TileSpacing','compact', ...
        'Padding','compact');
    nexttile;
    boxplot(alpha_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\alpha$ estimates', 'Interpreter','latex', 'FontSize',24);
    xlabel('$\alpha_1$ grid', 'Interpreter','latex', 'FontSize',18);
    ylabel('$\hat{\alpha}$', 'Interpreter','latex', 'FontSize',24);
    grid on;

    nexttile;
    boxplot(beta_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\beta$ estimates', 'Interpreter','latex', 'FontSize',24);
    xlabel('$\alpha_1$ grid', 'Interpreter','latex', 'FontSize',18);
    ylabel('$\hat{\beta}$', 'Interpreter','latex', 'FontSize',24);
    yline(0,'--k','True Value',FontSize=14);
    grid on;

    nexttile;
    boxplot(sigma_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\sigma$ estimates', 'Interpreter','latex', 'FontSize',24);
    xlabel('$\alpha_1$ grid', 'Interpreter','latex', 'FontSize',18);
    ylabel('$\hat{\sigma}$', 'Interpreter','latex', 'FontSize',24);
    grid on;

    nexttile;
    boxplot(mu_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\mu$ estimates', 'Interpreter','latex', 'FontSize',24);
    xlabel('$\alpha_1$ grid', 'Interpreter','latex', 'FontSize',18);
    ylabel('$\hat{\mu}$', 'Interpreter','latex', 'FontSize',24);
    yline(0,'--k','True Value',FontSize=14);
    grid on;
    sgtitle(sprintf('Stable Parameter Estimates Across Simulations (n = %d, n-sim = %d)', n(i), nsim), ...
            'Interpreter','latex',fontsize=38);

end

%%
b500 = importdata("CI_1k_10k_B500.mat");

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for j = 1:m
    nexttile;
    hold on;

    for i = 1:K
        histogram(b500(:,j,i), 30); 
    end

    hold off;

    title(sprintf('$\\alpha_1 = %.2f$', alpha1_grid(j)),'Interpreter','latex',FontSize=24);
    xlabel('CI Length','Interpreter','latex',FontSize=24);
    ylabel('Frequency','Interpreter','latex',FontSize=24);
    grid on;
end

L = cell(K,1);
for i = 1:K
    L{i} = ['n = ' num2str(n(i))];
end

legend(L,'Interpreter','latex','Location','bestoutside',FontSize=24);


%%
b50 = importdata("CI_1k_10k_B50.mat");

%


for i=1:K

    ax = gobjects(m,1);   
    
    for j = 1:m
        ax(j) = subplot(2,3,j);
        histogram(b50(:,j,i), 30);
        title(sprintf('$\\alpha_1 = %.2f$', alpha1_grid(j)),'Interpreter','latex');
        xlabel('CI Length','Interpreter','latex');
        ylabel('Frequency','Interpreter','latex');
    end
    
    linkaxes(ax,'x');  
    
    sgtitle(sprintf('Bootstrap 90%% CI Lengths (n = %d)', n(i)), ...
            'Interpreter','latex');
end

% B = 50