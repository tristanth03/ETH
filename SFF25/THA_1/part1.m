%% Integrate function

clc;clear; close all
function fS_res = num_int(integrand,s,x)
    fS_res = zeros(size(s));
    for i = 1:length(s)
            f = integrand(s(i));
            fS_res(i) = trapz(x,f(x));
    end

end

function [x,p]=runthefft(n,h,alpha1,alpha2)
    % From fftnoncentralchi2.m (Intermediate Propability ch2)

    n=2^n; x=(0:n-1)'*h-n*h/2; s=1/(h*n);
    t=2*pi*s*((0:n-1)'-n/2);
    sgn=ones(n,1); sgn(2:2:n)=-1*ones(n/2,1);
    CF = exp(-abs(t).^alpha1 - abs(t).^alpha2);
    % CF is the characteristic function in question; 
    %   in this case it is for the noncentral chi^2.
    phi=sgn.*CF; phi(n/2+1)=sgn(n/2+1); p=s.*abs(fft(phi)); 
end

function pdf = fft_stablesum(z, alpha1, alpha2)
    % From fftnoncentralchi2.m (Intermediate Propability ch2)
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
    pdf = wintp1(xgrd, bigpdf, z); 
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
fS_a = num_int(integrand,s,xgrid);
area_a = trapz(s,fS_a);

%% b

Nsim = 10000;
X1 = random(pd1,Nsim,1);
X2 = random(pd2,Nsim,1);
S = X1+X2;

fS_b = ksdensity(S, s);
area_b = trapz(s,fS_b);

%% c

integrand_c = @(s) @(x) cos(x*s).*exp(-x.^(alpha1)-x.^(alpha2)); 
xgrid_c = linspace(0,20,m);
fS_c = num_int(integrand_c,s,xgrid_c)*(1/(pi));
area_c = trapz(s,fS_c);

%% d

fS_d = fft_stablesum(s, alpha1, alpha2);
area_d = trapz(s, fS_d);

%%

close all

function pl = pl_res(x_lim, s, fS_a, fS_b, fS_c, fS_d, ...
                     alpha1, alpha2, n, Nsim, ...
                     area_a, area_b, area_c, area_d)
    figure; 
    hold on; grid on; box on;
    plt2 = plot(s, fS_b, '--b', 'LineWidth', 1.5);        
    plt3 = plot(s, fS_c, '-r',  'LineWidth', 1.8);         
    plt4 = plot(s, fS_d, '--g', 'LineWidth', 1.5);     
    plt1 = plot(s, fS_a, '.k',  'MarkerSize', 10);        
    
    title(sprintf('PDF of $S = X_1\\,(\\alpha_1 = %.1f) + X_2\\,(\\alpha_2 = %.1f)$, n = %d', ...
          alpha1, alpha2, n), ...
          'Interpreter','latex'); 
    xlabel('s');
    ylabel('fS(s)')
    
    lgd_text = { ...
        sprintf('Convolution'), ...
        sprintf('Kernel Density (n-sim = %d)', Nsim), ...
        sprintf('Characteristic Function'), ...
        sprintf('FFT') ...
    };
    
    legend([plt1, plt2, plt3, plt4], lgd_text, 'Location','northeast');
    
    txt = sprintf(['Area under curves (on [-10,10]):\n' ...
                   'Convolution     = %.4f\n' ...
                   'Kernel Density  = %.4f\n' ...
                   'CF Method       = %.4f\n' ...
                   'FFT Method       = %.4f\n\n' ...
                   ] ...
                   ,area_a,area_b,area_c,area_d);
    
    xpos = x_lim(1) + 0.02*(x_lim(2) - x_lim(1));
    ypos = max([fS_a(:); fS_b(:); fS_c(:)]) * 0.95;
    text(xpos, ypos, txt);
    xlim(x_lim)
    hold off
end
pl_res([-8,8], ...
s, fS_a, fS_b, fS_c, fS_d, ...
alpha1, alpha2, n, Nsim, ...
area_a, area_b, area_c, area_d)

pl_res([4,10], ...
s, fS_a, fS_b, fS_c, fS_d, ...
alpha1, alpha2, n, Nsim, ...
area_a, area_b, area_c, area_d)