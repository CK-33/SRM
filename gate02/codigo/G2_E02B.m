
% ============================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 02B-R
% REVISAO GRAFICA DA PERIODICIDADE TRIFASICA
%
% GNU OCTAVE 11.3.0
% ============================================================

clear; close all; clc;

graphics_toolkit('gnuplot');
set(0, 'defaultfigurevisible', 'off');

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

pasta_dados = ['PASTA_DO_USUARIO\Documents\' ...
               'Pesquisa_SRM\TesteNo1_31OCT2026\' ...
               'TesteNo1_31OCT2026\Dados'];

arquivo = fullfile(pasta_dados, 'results_1.mat');

pasta_saida = fullfile( ...
    pasta_dados, 'Gate2_Exp02B_Revisado');

if ~exist(pasta_saida, 'dir')
    mkdir(pasta_saida);
end

t_inicio = 0.85;
t_fim = 0.98;

T_ref = 0.01308;
lag_max_s = 0.050;

% Tres ciclos para visualizacao
t_zoom_inicio = 0.870;
t_zoom_fim = t_zoom_inicio + 3*T_ref;

nomes = {'Fase A','Fase B','Fase C'};
cores = {'r','b','g'};

% ------------------------------------------------------------
% 2. CARREGAMENTO
% ------------------------------------------------------------

dados = load(arquivo);
r = dados.results;

t = r.time(:);
I = r.signals(3).values;

assert(size(I,1) == length(t));
assert(size(I,2) >= 3);

idx = find(t >= t_inicio & t <= t_fim);

tw = t(idx);
X = I(idx,1:3);

fs = 1/median(diff(t));
N = length(tw);

fprintf('\n=== G2/E02B-R ===\n');
fprintf('Fs = %.2f Hz\n',fs);
fprintf('Janela = %.6f a %.6f s\n',tw(1),tw(end));

% ------------------------------------------------------------
% 3. CONFIGURACAO VISUAL
% ------------------------------------------------------------

set(0, 'defaultaxesfontsize', 12);
set(0, 'defaulttextfontsize', 12);

% ------------------------------------------------------------
% 4. FIGURA 1 - VISAO GERAL
% ------------------------------------------------------------

figure(1, 'position',[100 100 1200 520]);

hold on;

for k = 1:3
    plot(tw,X(:,k),cores{k},'linewidth',0.8);
end

xlabel('Tempo (s)');
ylabel('Corrente (A)');
title('Gate 2 - Correntes trifasicas pre-evento');

legend(nomes{:},'location','northeastoutside');

grid on;
xlim([t_inicio t_fim]);

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E02B_R_Fig01_VisaoGeral.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 5. FIGURA 2 - TRES CICLOS
% ------------------------------------------------------------

idx_zoom = find( ...
    tw >= t_zoom_inicio & tw <= t_zoom_fim);

figure(2,'position',[100 100 1200 850]);

for k = 1:3

    subplot(3,1,k);

    plot(tw(idx_zoom),X(idx_zoom,k), ...
         cores{k},'linewidth',1.1);

    ylabel('Corrente (A)');
    title(nomes{k});

    grid on;
    xlim([t_zoom_inicio t_zoom_fim]);

    if k == 3
        xlabel('Tempo (s)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E02B_R_Fig02_TresCiclos.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 6. AUTOCORRELACAO
% ------------------------------------------------------------

Xc = X - repmat(mean(X,1),N,1);

nfft = 2^nextpow2(2*N-1);

lag_max = min(round(lag_max_s*fs),N-1);

lags = (0:lag_max)'/fs;

Rauto = zeros(lag_max+1,3);

for k = 1:3

    F = fft(Xc(:,k),nfft);

    R = real(ifft(abs(F).^2));

    R = R(1:lag_max+1);

    Rauto(:,k) = R/R(1);

end

figure(3,'position',[100 100 1200 850]);

for k = 1:3

    subplot(3,1,k);

    plot(lags*1000,Rauto(:,k), ...
         cores{k},'linewidth',1.4);

    hold on;

    yl = [-1 1];

    for m = 1:3

        Tm = m*T_ref*1000;

        plot([Tm Tm],yl,'k--', ...
             'linewidth',0.8);

    end

    ylabel('Autocorrelacao');
    title(nomes{k});

    xlim([0 50]);
    ylim(yl);
    grid on;

    if k == 3
        xlabel('Atraso (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E02B_R_Fig03_Autocorrelacao.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 7. CORRELACAO CRUZADA
% ------------------------------------------------------------

pares = [1 2; 2 3; 1 3];
nomes_pares = {'A-B','B-C','A-C'};

% Busca restrita a meio periodo para cada lado
lag_cruzado = round((T_ref/2)*fs);

lags_cross = (-lag_cruzado:lag_cruzado)'/fs;

atrasos = NaN(3,1);
coeficientes = NaN(3,1);

figure(4,'position',[100 100 1200 850]);

for p = 1:3

    a = pares(p,1);
    b = pares(p,2);

    xa = Xc(:,a);
    xb = Xc(:,b);

    Fa = fft(xa,nfft);
    Fb = fft(xb,nfft);

    % R_ab(lag) = soma xa(n)*xb(n+lag)
    C = real(ifft(conj(Fa).*Fb));

    Rneg = C(end-lag_cruzado+1:end);
    Rpos = C(1:lag_cruzado+1);

    R = [Rneg;Rpos];

    denom = sqrt(sum(xa.^2)*sum(xb.^2));
    R = R/denom;

    [valor,j] = max(R);

    atraso = lags_cross(j);

    atrasos(p) = atraso;
    coeficientes(p) = valor;

    subplot(3,1,p);

    plot(lags_cross*1000,R, ...
         'b','linewidth',1.3);

    hold on;

    plot(atraso*1000,valor, ...
         'ro','markersize',7, ...
         'markerfacecolor','r');

    plot([0 0],[-1 1],'k--');

    ylabel('Correlacao');
    title(['Par ' nomes_pares{p}]);

    xlim([-T_ref/2 T_ref/2]*1000);
    ylim([-1 1]);

    grid on;

    texto = sprintf( ...
        'Atraso = %.3f ms | R = %.4f', ...
        atraso*1000,valor);

    text(0.02,0.92,texto, ...
         'units','normalized', ...
         'fontsize',11, ...
         'backgroundcolor','w');

    if p == 3
        xlabel('Atraso (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E02B_R_Fig04_CorrelacaoCruzada.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 8. RESULTADOS
% ------------------------------------------------------------

fprintf('\n--- ATRASOS NO INTERVALO +/- T/2 ---\n');

for p = 1:3

    fprintf('%s: %.4f ms | R = %.6f\n', ...
        nomes_pares{p}, ...
        atrasos(p)*1000, ...
        coeficientes(p));

end

fprintf('\nArquivos salvos em:\n%s\n',pasta_saida);

fprintf('\nG2/E02B-R concluido.\n');
