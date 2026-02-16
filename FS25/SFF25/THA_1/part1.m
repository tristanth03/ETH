clc;clear; close all
function fS_res = num_int(integrand,s,x)
    fS_res = zeros(size(s));
    for i = 1:length(s)
            f = integrand(s(i));
            fS_res(i) = trapz(x,f(x));
    end
end


%% Setup
rng(12345);
alpha1 = 1.3;
alpha2 = 1.7;
pd1 = makedist('Stable','alpha',alpha1,'beta',0,'gam',1,'delta',0);
pd2 = makedist('Stable','alpha',alpha2,'beta',0,'gam',1,'delta',0);

n = 200;
s = linspace(-10,10,n);

m = 100;
xgrid = linspace(-20,20,m);

%% a

integrand = @(s) @(x) pdf(pd1,x) .* pdf(pd2,s-x);
fS_1 = num_int(integrand,s,xgrid);
area_1 = trapz(s,fS_1);

%% b
Nsim = 10000;
X1 = random(pd1,Nsim,1);
X2 = random(pd2,Nsim,1);
S = X1+X2;

fS_2 = ksdensity(S, s);
area_2 = trapz(s,fS_2);

%% c
integrand_c = @(s) @(t) cos(t*s).*exp(-t.^(alpha1)-t.^(alpha2)); 
tgrid = linspace(0,20,m);
fS_3 = num_int(integrand_c,s,tgrid)*(1/(pi));
area_3 = trapz(s,fS_3);

%% d

%%% From fftnoncentralchi2.m;
%%% Intermediate Propability ch2;
%%% https://www.marc-paolella.com/intermediate-probability, PROGRAMS in Matlab

function [x,p]=runthefft(n,h,alpha1,alpha2)
    n=2^n; x=(0:n-1)'*h-n*h/2; s=1/(h*n);
    t=2*pi*s*((0:n-1)'-n/2);
    sgn=ones(n,1); sgn(2:2:n)=-1*ones(n/2,1);
    CF = exp(-abs(t).^alpha1 - abs(t).^alpha2);
    % CF is the characteristic function in question; 
    phi=sgn.*CF; phi(n/2+1)=sgn(n/2+1); p=s.*abs(fft(phi)); 
end

function pdf = fft_stablesum(z, alpha1, alpha2)
    pmax = 18;
    step = 0.01;
    p = 14;
    maxz = round(max(abs(z))) + 5;
    
    while ((maxz/step + 1) > 2^(p-1))
        p = p + 1;
    end
    if p > pmax, p = pmax; end
    if maxz/step + 1 > 2^(p-1)
        step = (maxz + 1)*1.001 / (2^(p-1));
    end
    
    [xgrd, bigpdf] = runthefft(p, step, alpha1, alpha2);
    % pdf = wintp1(xgrd, bigpdf, z); 
    pdf = interp1(xgrd, bigpdf, z, 'linear');
end

fS_4 = fft_stablesum(s, alpha1, alpha2);
area_4 = trapz(s, fS_4);

%%

close all

function pl = pl_res(x_lim, s, fS_a, fS_b, fS_c, fS_d, ...
                     alpha1, alpha2, n, Nsim, ...
                     area_a, area_b, area_c, area_d)
    figure; 
    hold on; grid on; box on;
    set(gca, 'FontSize', 22);

    plt2 = plot(s, fS_b, '--b','LineWidth', 1.5);        
    plt3 = plot(s, fS_c, '-r', 'LineWidth', 1.8);         
    plt4 = plot(s, fS_d,'--g', 'LineWidth', 1.5);     
    plt1 = plot(s, fS_a,'.k','MarkerSize',10);        
    
    title(sprintf('PDF estimation of $S = X_1\\,(\\alpha_1 = %.1f) + X_2\\,(\\alpha_2 = %.1f)$, n = %d', ...
          alpha1, alpha2, n), ...
          'Interpreter','latex',fontsize=32); 
    xlabel('s',fontsize=24);
    ylabel('fS(s)',fontsize=24)
    
    
    lgd_text = { ...
        sprintf('$\\hat{f^1_S}(s)$ Convolution'), ...
        sprintf('$\\hat{f^2_S}(s)$ Kernel Density (n-sim = %d)', Nsim), ...
        sprintf('$\\hat{f^3_S}(s)$ Characteristic Function'), ...
        sprintf('$\\hat{f^4_S}(s)$ FFT' ) ...
    };
    
    legend([plt1, plt2, plt3, plt4], lgd_text, 'Location','northeast','Interpreter','latex',fontsize=24);
    
    txt = sprintf(['Area under the curves (on [-10,10]):\n' ...
                   'Convolution     = %.4f\n' ...
                   'Kernel Density  = %.4f\n' ...
                   'CF Method       = %.4f\n' ...
                   'FFT Method       = %.4f\n\n' ...
                   ] ...
                   ,area_a,area_b,area_c,area_d);
    
    xpos = x_lim(1) + 0.02*(x_lim(2) - x_lim(1));
    ypos = max([fS_a(:); fS_b(:); fS_c(:)]) * 0.95;
    text(xpos, ypos,txt,fontsize=22);
    xlim(x_lim)
    hold off
    pl = 1;
end
pl_res([-10,10], ...
s, fS_1, fS_2, fS_3, fS_4, ...
alpha1, alpha2, n, Nsim, ...
area_1, area_2, area_3, area_4);

pl_res([4,10], ...
s, fS_1, fS_2, fS_3, fS_4, ...
alpha1, alpha2, n, Nsim, ...
area_1, area_2, area_3, area_4);