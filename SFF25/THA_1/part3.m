clc; clear; close all;


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


%% MLE (Single-Component)
%%% https://ch.mathworks.com/matlabcentral/fileexchange/37514-stbl-alpha-stable-distributions-for-matlab
rng(12345);
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
    alpha0 = 1.8;
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


%% Numerical Experiment Part III - (a)
alpha_grid = linspace(1.2,1.8,7);
m = length(alpha_grid);
n = 10000;
nsim = 10;
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
%% Plotting Part III - (a)

figure;
t = tiledlayout(1, 3, ...
    'TileSpacing','compact', ...
    'Padding','compact');
nexttile;
boxplot(alpha_hat, 'Labels', string(alpha_grid));
title('$\alpha$ estimates', 'Interpreter','latex', 'FontSize',24);
xlabel('$\alpha$ grid', 'Interpreter','latex', 'FontSize',18);
ylabel('$\hat{\alpha}$', 'Interpreter','latex', 'FontSize',24);
grid on;

nexttile;
boxplot(sigma_hat, 'Labels', string(alpha_grid));
title('$\sigma$ estimates', 'Interpreter','latex', 'FontSize',24);
xlabel('$\alpha$ grid', 'Interpreter','latex', 'FontSize',18);
ylabel('$\hat{\sigma}$', 'Interpreter','latex', 'FontSize',24);
yline(sigma_true,'--k','True Value',FontSize=14);

grid on;

nexttile;
boxplot(mu_hat, 'Labels', string(alpha_grid));
title('$\mu$ estimates', 'Interpreter','latex', 'FontSize',24);
xlabel('$\alpha$ grid', 'Interpreter','latex', 'FontSize',18);
ylabel('$\hat{\mu}$', 'Interpreter','latex', 'FontSize',24);
yline(mu_true,'--k','True Value',FontSize=14);
grid on;

%% Numerical Experiment 2, Part III - (a)
clc;clear;close all;
rng(12345);
k = 4;
alpha_grid = linspace(1.2,1.8,k);
n = 10000;
sigma_true = 1;
mu_true = 0;

pdf_true = zeros(n,k);
pdf_hat = zeros(n,k);
s = linspace(-10,10,n);

for i=1:k
    pd_true = makedist("Stable","alpha",alpha_grid(i),"beta",0,"gam",sigma_true,"delta",mu_true);
    pdf_true(:,i) = stablepdf_fft(s,alpha_grid(i),0,sigma_true,mu_true);   
    x = random(pd_true,n,1);
    [a,sig,mu] = mle_symstable(x);
    pdf_hat(:,i) = stablepdf_fft(s,a,0,sig,mu);
   
end

%% Plotting (2) Part III - (a)
function pl = pl_res(x_lim, s, fT, fH,alpha_grid)
    figure;
    t = tiledlayout(2, 2, ...
        'TileSpacing','compact', ...
        'Padding','compact');
    for i=1:length(alpha_grid)
        nexttile;
        tP = plot(s,fT(:,i),'-b');
        title(sprintf('PDF comparison ($\\alpha$ = %.2f)', alpha_grid(i)), 'Interpreter','latex', 'FontSize',24);
        hold on
        tH = plot(s,fH(:,i),'--r');
        xlabel('s', 'Interpreter','latex', 'FontSize',18);
        ylabel('$f_S(s)$', 'Interpreter','latex', 'FontSize',24);
        legend([tP,tH],{"True FFT","MLE"},'Interpreter','latex', 'FontSize',24)
        grid on;
        xlim(x_lim)
    end
    pl = 1;
end

pl_res([-10,10],s,pdf_true,pdf_hat,alpha_grid);
pl_res([4,10],s,pdf_true,pdf_hat,alpha_grid);

%%


% nexttile;
% boxplot(sigma_hat, 'Labels', string(alpha_grid));
% title('$\sigma$ estimates', 'Interpreter','latex', 'FontSize',24);
% xlabel('$\alpha$ grid', 'Interpreter','latex', 'FontSize',18);
% ylabel('$\hat{\sigma}$', 'Interpreter','latex', 'FontSize',24);
% yline(sigma_true,'--k','True Value',FontSize=14);
% 
% grid on;
% 
% nexttile;
% boxplot(mu_hat, 'Labels', string(alpha_grid));
% title('$\mu$ estimates', 'Interpreter','latex', 'FontSize',24);
% xlabel('$\alpha$ grid', 'Interpreter','latex', 'FontSize',18);
% ylabel('$\hat{\mu}$', 'Interpreter','latex', 'FontSize',24);
% yline(mu_true,'--k','True Value',FontSize=14);
% grid on;




%% MLE two-component
function mle_di = neglog_sym_di(theta,x)
    pi_ = 1/(1 + exp(-theta(1)));
    alpha1 = 2 * 1./(1+exp(-theta(2)));
    alpha2 = 2 * 1./(1+exp(-theta(3)));
    sigma1 = exp(theta(4));
    sigma2 = exp(theta(5));
    mu1 = theta(6);
    mu2 = theta(7);
    beta = 0; %symetric
    f1 = stablepdf_fft(x,alpha1,beta,sigma1,mu1);
    f2 = stablepdf_fft(x,alpha2,beta,sigma2,mu2);

    fmix = pi_*f1+(1-pi_)*f2;
    tol = 1e-16;
    fmix(fmix<tol) = tol;
    mle_di = -sum(log(fmix));
end

function [pi_hat,alpha_hat1,alpha_hat2,sigma_hat1,sigma_hat2,mu_hat1,mu_hat2] = mle_symstable_di(x)
    neglog = @(theta) neglog_sym_di(theta,x);
    eps_ = 0.1;
    pi_t = 0.5;
    alpha_t1 = 1.5-eps_;
    alpha_t2 = 1.5+eps_;
    sigma_t1 = eps_;
    sigma_t2 = eps_;
    mu_t1 = eps;
    mu_t2 = -eps_;
    theta0 = [pi_t;-log(2/alpha_t1-1);-log(2/alpha_t2-1);log(sigma_t1);log(sigma_t2);mu_t1;mu_t2];
    % from http://mathworks.com/help/optim/ug/fminunc.html
    theta_hat = fminunc(neglog, theta0, ...
        optimoptions('fminunc','Algorithm','quasi-newton',...
                         'Display','off','MaxIterations',200,'MaxFunctionEvaluations',400));
    pi_hat   = 1/(1+exp(-theta_hat(1)));
    alpha_hat1   = 2./(1+exp(-theta_hat(2)));
    alpha_hat2   = 2./(1+exp(-theta_hat(3)));
    sigma_hat1   = exp(theta_hat(4));
    sigma_hat2   = exp(theta_hat(5));
    mu_hat1  = theta_hat(6);
    mu_hat2  = theta_hat(7);

end



%% Numerical Experiment Part III - (b)
clc;clear;close all;
rng(12345);
k = 4;
alpha1_grid = linspace(1.2,1.8,k);
alpha2 = 1.5;
n = 10000;
sigma_true = 1;
mu_true = 0;
pi_true = 0.6;

pdf_true = zeros(n,k);
pdf_hat = zeros(n,k);
s = linspace(-10,10,n);

pd2_true = makedist("Stable","alpha",alpha2,"beta",0,"gam",sigma_true,"delta",mu_true);
for i=1:k
    pd1_true = makedist("Stable","alpha",alpha1_grid(i),"beta",0,"gam",sigma_true,"delta",mu_true);

    pdf_true(:,i) = pi_true.*stablepdf_fft(s,alpha1_grid(i),0,sigma_true,mu_true) + (1-pi_true).*stablepdf_fft(s,alpha2,0,sigma_true,mu_true);   

    u = rand(n,1);
    idx1 = u < pi_true;
    idx2 = u >= pi_true;
    x = zeros(n,1);
    x(idx1) = random(pd1_true,sum(idx1),1);
    x(idx2) = random(pd2_true,sum(idx2),1);

    [pi_hat, alpha_hat1, alpha_hat2, sigma_hat1, sigma_hat2, mu_hat1, mu_hat2] = mle_symstable_di(x);
    f1_hat = stablepdf_fft(s, alpha_hat1, 0, sigma_hat1, mu_hat1);
    f2_hat = stablepdf_fft(s, alpha_hat2, 0, sigma_hat2, mu_hat2);
    pdf_hat(:, i) = pi_hat * f1_hat + (1 - pi_hat) * f2_hat;

end
%%
function pl = pl_res2(x_lim, s, fT,fH,alpha1_grid, alpha2,pi)
    figure;
    t = tiledlayout(2, 2, ...
        'TileSpacing','compact', ...
        'Padding','compact');
    for i=1:length(alpha1_grid)
        nexttile;
        tP = plot(s,fT(:,i),'-b');
        title(sprintf('PDF comparison ($\\alpha_1$ = %.2f, $\\alpha_2$ = %.2f, $\\pi$ = %.2f)', alpha1_grid(i),alpha2,pi), ...
            'Interpreter','latex', 'FontSize',24);
        hold on
        tH = plot(s,fH(:,i),'--r');
        xlabel('s', 'Interpreter','latex', 'FontSize',18);
        ylabel('$f_S(s)$', 'Interpreter','latex', 'FontSize',24);
        legend([tP,tH],{"True FFT","MLE"},'Interpreter','latex', 'FontSize',24)
        grid on;
        xlim(x_lim)
    end
    pl = 1;
end

pl_res2([-10,10],s,pdf_true,pdf_hat,alpha1_grid,alpha2,pi_true);
pl_res2([4,10],s,pdf_true,pdf_hat,alpha1_grid,alpha2,pi_true);



%% c
clear;close all;clc;
r_data = importdata("DJIA30stockreturns.mat");
[days, nStocks] = size(r_data);

pi_hat_dta = zeros(nStocks, 1);
alpha_hat1_dta = zeros(nStocks, 1);
alpha_hat2_dta = zeros(nStocks, 1);
sigma_hat1_dta = zeros(nStocks, 1);
sigma_hat2_dta = zeros(nStocks, 1);
mu_hat1_dta = zeros(nStocks, 1);
mu_hat2_dta = zeros(nStocks, 1);
pdf_hat = zeros(days, nStocks);
for stock = 1:nStocks
    stockDta = r_data(:, stock);
    [pi_hat, alpha_hat1, alpha_hat2, sigma_hat1, sigma_hat2, mu_hat1, mu_hat2] = mle_symstable_di(stockDta);
    pi_hat_dta(stock) = pi_hat;
    alpha_hat1_dta(stock) = alpha_hat1;
    alpha_hat2_dta(stock) = alpha_hat2;
    sigma_hat1_dta(stock) = sigma_hat1;
    sigma_hat2_dta(stock) = sigma_hat2;
    mu_hat1_dta(stock) = mu_hat1;
    mu_hat2_dta(stock) = mu_hat2;
    f1_hat = stablepdf_fft(stockDta, alpha_hat1, 0, sigma_hat1, mu_hat1);
    f2_hat = stablepdf_fft(stockDta, alpha_hat2, 0, sigma_hat2, mu_hat2);
    pdf_hat(:, stock) = pi_hat * f1_hat + (1 - pi_hat) * f2_hat;
end


%%
figure;
t = tiledlayout(5,5, ...
    'TileSpacing','compact', ...
    'Padding','compact');
for stock=1:nStocks
    nexttile;
    stockDta = r_data(:,stock);
    [sorted_stockDta, sort_idx] = sort(stockDta);
    sorted_pdf_hat = pdf_hat(sort_idx, stock);
    tH = plot(sorted_stockDta,sorted_pdf_hat,'-b',LineWidth=2);
    title(sprintf('Stock # %.d', stock),'Interpreter','latex', 'FontSize',12);
    hold on
    tP = histogram(stockDta,int32(sqrt(days)),'Normalization','pdf');
    xlabel('Returns', 'Interpreter','latex', 'FontSize',14);
    ylabel('Density', 'Interpreter','latex', 'FontSize',16);
    if stock == 1
        legend([tH, tP], {'Two-component','True'}, 'Location','northwest', ...
            'Interpreter','latex',fontsize=14);
    end
   
end

%%
parameterMatrix = [pi_hat_dta, alpha_hat1_dta, alpha_hat2_dta, ...
                   sigma_hat1_dta, sigma_hat2_dta, mu_hat1_dta,mu_hat2_dta];

labels = {'$\hat{\pi}$', '$\hat{\alpha}_1$', '$\hat{\alpha}_2$', ...
          '$\hat{\sigma}_1$', '$\hat{\sigma}_2$', '$\hat{\mu}_1$', '$\hat{\mu}_2$'};

figure;
boxplot(parameterMatrix, 'Labels', string(labels));
title('Distribution of Estimated Parameters Across Stocks', 'Interpreter', ...
    'latex', 'FontSize',24);
ylabel('Parameter Value', 'FontSize',24);
grid on;
ax = gca;
ax.TickLabelInterpreter = 'latex';  
ax.XAxis.FontSize = 24; 
ax.YAxis.FontSize = 24;

