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
B = 200;
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

for i=1:K
    figure;
    
    subplot(2,2,1)
    boxplot(alpha_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\alpha$ estimates', 'Interpreter','latex');
    xlabel('$\alpha_1$ grid', 'Interpreter','latex');
    ylabel('$\hat{\alpha}$', 'Interpreter','latex');
    grid on
    
    subplot(2,2,2)
    boxplot(beta_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\beta$ estimates', 'Interpreter','latex');
    xlabel('$\alpha_1$ grid', 'Interpreter','latex');
    ylabel('$\hat{\beta}$', 'Interpreter','latex');
    grid on
    
    subplot(2,2,3)
    boxplot(sigma_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\sigma$ estimates', 'Interpreter','latex');
    xlabel('$\alpha_1$ grid', 'Interpreter','latex');
    ylabel('$\hat{\sigma}$', 'Interpreter','latex');
    grid on
    
    subplot(2,2,4)
    boxplot(mu_hat(:,:,i), 'Labels', string(alpha1_grid));
    title('$\mu$ estimates', 'Interpreter','latex');
    xlabel('$\alpha_1$ grid', 'Interpreter','latex');
    ylabel('$\hat{\mu}$', 'Interpreter','latex');
    grid on
    
    sgtitle(sprintf('Stable Parameter Estimates Across Simulations (n = %d, n-sim = %d)', n(i), nsim), ...
            'Interpreter','latex');
        
    figure;
    ax = gobjects(m,1);   % store axes handles
    
    for j = 1:m
        ax(j) = subplot(2,3,j);
        histogram(CI_length(:,j,i), 30);
        title(sprintf('$\\alpha_1 = %.2f$', alpha1_grid(j)),'Interpreter','latex');
        xlabel('CI Length','Interpreter','latex');
        ylabel('Frequency','Interpreter','latex');
    end
    
    linkaxes(ax,'x');   % <-- this synchronizes all x-axes
    
    sgtitle(sprintf('Bootstrap 90%% CI Lengths (n = %d)', n(i)), ...
            'Interpreter','latex');
end

% B = 50