clc; clear; close all;

rng(12345);

function mle_uni = neglog_sym(theta,x)
    t_alpha = theta(1);
    t_logSigma = theta(2);
    mu = theta(3);
    alpha = 2 * 1./(1+exp(-t_alpha));
    sigma = exp(t_logSigma);
    beta = 0; % symmetr
    gamma = sigma; 
    delta = mu;

    pd = makedist("Stable","alpha",alpha,"beta",beta,"gam",gamma,"delta",delta);
    f = pdf(pd,x);

    tol = 1e-16;
    f(f<tol) = tol;
    mle_uni = -sum(log(f));
end

function [alpha_hat,sigma_hat,mu_hat,theta_hat] = mle_symstable(x)
    neglog = @(theta) neglog_sym(theta,x);
    theta0 = [1.5;0;mean(x)];
    
    opts = optimset('Display','off','MaxIter',200,'MaxFunEvals',200);
    
    theta_hat = fminsearch(neglog, theta0, opts);
    alpha_hat = 2 ./ (1 + exp(-theta_hat(1)));
    sigma_hat = exp(theta_hat(2));
    mu_hat    = theta_hat(3);

end




%%
rng(1);

alpha_grid = linspace(1.2,1.8,6);
m = length(alpha_grid);
n = 1000;
nsim = 10;
sigma_true = 1;
mu_true = 0;

alpha_hat = zeros(nsim,m);
sigma_hat = zeros(nsim,m);
mu_hat    = zeros(nsim,m);

for j = 1:m
    pd_true = makedist("Stable","alpha",alpha_grid(j),"beta",0,"gam",sigma_true,"delta",mu_true);
    for s = 1:nsim
        x = random(pd_true,n,1);
        [a,sig,mu] = mle_symstable(x);
        alpha_hat(s,j) = a;
        sigma_hat(s,j) = sig;
        mu_hat(s,j)    = mu;
    end
end

figure;
subplot(2,2,1)
boxplot(alpha_hat, 'Labels', string(alpha_grid));
title('$\alpha$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\alpha}$', 'Interpreter','latex');
grid on

subplot(2,2,2)
boxplot(mu_hat, 'Labels', string(alpha_grid));
title('$\mu$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\mu}$', 'Interpreter','latex');
grid on

subplot(2,2,3)
boxplot(sigma_hat, 'Labels', string(alpha_grid));
title('$\sigma$ estimates', 'Interpreter','latex');
xlabel('$\alpha$ grid', 'Interpreter','latex');
ylabel('$\hat{\sigma}$', 'Interpreter','latex');
grid on


% alpha_true = linspace(1.1,1.9,10);
% for i=1:length(alpha_true)
%     figure;
%     sigma_true = 1.0;
%     mu_true    = 0.0;
%     beta_true  = 0;
%     n = 10000;
%     pd_true = makedist("Stable", "alpha",alpha_true(i), "beta",beta_true, ...
%                                      "gam",sigma_true, "delta",mu_true);
% 
%     x = random(pd_true, n, 1);
%     [alpha_hat,sigma_hat,mu_hat,theta_hat] = mle_symstable(x);
%     pd_mle = makedist("Stable", ...
%         "alpha",alpha_hat, "beta",0, "gam",sigma_hat, "delta",mu_hat);
% 
%     X = linspace(prctile(x,0.5),prctile(x,99.5),500);
%     plt1 = plot(X,pdf(pd_mle,X));
%     hold on
%     plt2 = plot(X,pdf(pd_true,X));
%     title(sprintf('PDF of $\\alpha = %.1f$', ...
%       alpha_true(i)),'Interpreter','latex'); 
%     legend([plt1,plt2],{"MLE estimation","True"})
% 
% end
