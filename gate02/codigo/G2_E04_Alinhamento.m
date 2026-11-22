
% ============================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 04
% SENSIBILIDADE AO ALINHAMENTO TEMPORAL
%
% GNU OCTAVE 11.3.0
% ============================================================

clear; close all; clc;

graphics_toolkit('gnuplot');
set(0,'defaultfigurevisible','off');

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

pasta_dados = ['PASTA_DO_USUARIO\Documents\' ...
               'Pesquisa_SRM\TesteNo1_31OCT2026\' ...
               'TesteNo1_31OCT2026\Dados'];

arquivo = fullfile(pasta_dados,'results_1.mat');

pasta_saida = fullfile(pasta_dados,'Gate2_Exp04');

if ~exist(pasta_saida,'dir')
    mkdir(pasta_saida);
end

T_ref = 0.01308;
margem_evento = 0.010;
limiar_atividade = 1.0;

fracao_treino = 0.70;

max_deslocamento_s = 0.00025;

nomes = {'A','B','C'};
cores = {'r','b','g'};

% ------------------------------------------------------------
% 2. CARREGAMENTO
% ------------------------------------------------------------

dados = load(arquivo);
r = dados.results;

t = r.time(:);
I = r.signals(3).values;
evento = r.signals(6).values;

assert(size(I,1)==length(t));
assert(size(I,2)>=3);

idx_evento = find(evento(:,1)>0,1,'first');

if isempty(idx_evento)
    error('Evento nao encontrado.');
end

t_evento = t(idx_evento);
fs = 1/median(diff(t));

N_ciclo = round(T_ref*fs);
T_efetivo = N_ciclo/fs;

max_shift = round(max_deslocamento_s*fs);

fprintf('\n====================================\n');
fprintf('GATE 2 - EXPERIMENTO 04\n');
fprintf('====================================\n');

fprintf('Fs: %.2f Hz\n',fs);
fprintf('Periodo: %.6f ms\n',T_efetivo*1000);
fprintf('Maximo deslocamento: %d amostras\n',max_shift);

% ------------------------------------------------------------
% 3. REPRODUZIR OS CICLOS DO E03
% ------------------------------------------------------------

idx_ativo = find( ...
    any(abs(I(:,1:3))>limiar_atividade,2) & ...
    t<t_evento-margem_evento, ...
    1,'first');

idx_final = find( ...
    t<=t_evento-margem_evento, ...
    1,'last');

if isempty(idx_ativo) || isempty(idx_final)
    error('Intervalo pre-evento invalido.');
end

N_disponivel = idx_final-idx_ativo+1;

K = floor(N_disponivel/N_ciclo);

if K < 5
    error('Ciclos insuficientes.');
end

idx_usados = idx_ativo:(idx_ativo+K*N_ciclo-1);

X = I(idx_usados,1:3);

C = zeros(N_ciclo,K,3);

for fase = 1:3
    C(:,:,fase) = reshape(X(:,fase),N_ciclo,K);
end

K_treino = floor(fracao_treino*K);
K_teste = K-K_treino;

if K_treino < 3 || K_teste < 2
    error('Divisao treino/teste insuficiente.');
end

fprintf('Ciclos totais: %d\n',K);
fprintf('Treinamento: %d\n',K_treino);
fprintf('Validacao: %d\n',K_teste);

% ------------------------------------------------------------
% 4. REFERENCIA ORIGINAL - M0
% ------------------------------------------------------------

mu0 = zeros(N_ciclo,3);

for fase = 1:3
    mu0(:,fase) = mean(C(:,1:K_treino,fase),2);
end

% ------------------------------------------------------------
% 5. ALINHAMENTO TRIFASICO DE TREINAMENTO
% ------------------------------------------------------------

% Referencia inicial fixa: primeiro ciclo.
% Cada deslocamento e compartilhado pelas tres fases.
% Nao utilizamos deslocamento circular para evitar
% transportar amostras artificialmente entre as bordas.

referencia = squeeze(C(:,1,:));

C_alinhado = NaN(N_ciclo,K_treino,3);
shifts_treino = zeros(K_treino,1);

% Margem comum: evita que a funcao objetivo
% favoreca deslocamentos por usar menos amostras.
margem = max_shift + 1;
idx_centro = (margem+1):(N_ciclo-margem);

for k = 1:K_treino

    ciclo = squeeze(C(:,k,:));

    melhor_erro = Inf;
    melhor_shift = 0;

    for s = -max_shift:max_shift

        % Comparar referencia(n) com ciclo(n+s).
        idx_movido = idx_centro+s;

        erro = ciclo(idx_movido,:) - ...
               referencia(idx_centro,:);

        mse = mean(erro(:).^2);

        if mse < melhor_erro
            melhor_erro = mse;
            melhor_shift = s;
        end

    end

    shifts_treino(k) = melhor_shift;

    % Aplicar shift com NaN nas bordas.
    alinhado = NaN(N_ciclo,3);

    if melhor_shift >= 0

        destino = 1:(N_ciclo-melhor_shift);
        origem = (1+melhor_shift):N_ciclo;

    else

        destino = (1-melhor_shift):N_ciclo;
        origem = 1:(N_ciclo+melhor_shift);

    end

    alinhado(destino,:) = ciclo(origem,:);

    C_alinhado(:,k,:) = reshape( ...
        alinhado,N_ciclo,1,3);

end

% ------------------------------------------------------------
% 6. REFERENCIA ALINHADA - M1
% ------------------------------------------------------------

mu1 = NaN(N_ciclo,3);

for fase = 1:3

    for n = 1:N_ciclo

        valores = squeeze(C_alinhado(n,:,fase));
        valores = valores(~isnan(valores));

        if ~isempty(valores)
            mu1(n,fase) = mean(valores);
        end

    end

end

% Preencher apenas bordas sem cobertura.
% Para validacao justa, usaremos o suporte comum
% central, onde todas as referencias sao definidas.

idx_avaliacao = idx_centro;

assert(all(all(isfinite(mu1(idx_avaliacao,:)))));

% ------------------------------------------------------------
% 7. VALIDACAO SEM AJUSTE - M0 E M1
% ------------------------------------------------------------

RMSE0 = zeros(K_teste,3);
RMSE1 = zeros(K_teste,3);
RMSE2 = zeros(K_teste,3);

shifts_teste = zeros(K_teste,1);

for j = 1:K_teste

    k = K_treino+j;
    ciclo = squeeze(C(:,k,:));

    % M0: referencia fixa
    erro0 = ciclo(idx_avaliacao,:) - ...
            mu0(idx_avaliacao,:);

    % M1: referencia treinada com alinhamento,
    % mas ciclo de validacao sem ajuste
    erro1 = ciclo(idx_avaliacao,:) - ...
            mu1(idx_avaliacao,:);

    for fase = 1:3

        RMSE0(j,fase) = ...
            sqrt(mean(erro0(:,fase).^2));

        RMSE1(j,fase) = ...
            sqrt(mean(erro1(:,fase).^2));

    end

    % --------------------------------------------------------
    % M2: ALINHAMENTO RETROSPECTIVO DO CICLO DE TESTE
    % --------------------------------------------------------

    melhor_erro = Inf;
    melhor_shift = 0;

    for s = -max_shift:max_shift

        idx_movido = idx_avaliacao+s;

        erro = ciclo(idx_movido,:) - ...
               mu1(idx_avaliacao,:);

        mse = mean(erro(:).^2);

        if mse < melhor_erro

            melhor_erro = mse;
            melhor_shift = s;

        end

    end

    shifts_teste(j) = melhor_shift;

    erro2 = ciclo(idx_avaliacao+melhor_shift,:) - ...
            mu1(idx_avaliacao,:);

    for fase = 1:3

        RMSE2(j,fase) = ...
            sqrt(mean(erro2(:,fase).^2));

    end

end

% ------------------------------------------------------------
% 8. RESULTADOS
% ------------------------------------------------------------

fprintf('\n--- DESLOCAMENTOS TREINAMENTO ---\n');

for k = 1:K_treino

    fprintf('Ciclo %d: %+d amostras (%+.4f ms)\n', ...
        k,shifts_treino(k), ...
        1000*shifts_treino(k)/fs);

end

fprintf('\n--- DESLOCAMENTOS VALIDACAO (M2) ---\n');

for j = 1:K_teste

    fprintf('Ciclo %d: %+d amostras (%+.4f ms)\n', ...
        j,shifts_teste(j), ...
        1000*shifts_teste(j)/fs);

end

fprintf('\n--- RMSE MEDIO POR FASE ---\n');

fprintf('Fase | M0 (A) | M1 (A) | M2 (A)\n');

for fase = 1:3

    fprintf(' %s   | %.6f | %.6f | %.6f\n', ...
        nomes{fase}, ...
        mean(RMSE0(:,fase)), ...
        mean(RMSE1(:,fase)), ...
        mean(RMSE2(:,fase)));

end

fprintf('\n--- RMSE MEDIO GLOBAL ---\n');

fprintf('M0: %.6f A\n',mean(RMSE0(:)));
fprintf('M1: %.6f A\n',mean(RMSE1(:)));
fprintf('M2: %.6f A\n',mean(RMSE2(:)));

fprintf('\nOBSERVACAO: M2 usa alinhamento retrospectivo.\n');
fprintf('Nao equivale a previsao sem informacao do ciclo.\n');

% ------------------------------------------------------------
% 9. FIGURA 1 - REFERENCIAS M0 E M1
% ------------------------------------------------------------

tau = (0:N_ciclo-1)'/fs*1000;

figure(1,'position',[100 100 1200 850]);

for fase = 1:3

    subplot(3,1,fase);

    plot(tau,mu0(:,fase),'k--','linewidth',1.1);
    hold on;

    plot(tau,mu1(:,fase),cores{fase}, ...
         'linewidth',1.4);

    title(['Fase ' nomes{fase}]);
    ylabel('Corrente (A)');

    grid on;
    xlim([0 T_efetivo*1000]);

    if fase == 1
        legend('M0 - fixa','M1 - alinhada', ...
               'location','northeast');
    end

    if fase == 3
        xlabel('Tempo relativo (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E04_Fig01_Referencias.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 10. FIGURA 2 - RMSE POR FASE
% ------------------------------------------------------------

figure(2,'position',[100 100 1100 600]);

valores = [mean(RMSE0,1)', ...
           mean(RMSE1,1)', ...
           mean(RMSE2,1)'];

bar(valores);

set(gca,'xtick',1:3);
set(gca,'xticklabel',nomes);

ylabel('RMSE medio (A)');
title('Gate 2 - Comparacao dos metodos');

legend('M0','M1','M2','location','northeast');
grid on;

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E04_Fig02_RMSE.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 11. FIGURA 3 - DESLOCAMENTOS
% ------------------------------------------------------------

figure(3,'position',[100 100 1100 550]);

plot(1:K_treino, ...
     shifts_treino/fs*1000, ...
     'bo-','linewidth',1.4);

hold on;

plot(K_treino+(1:K_teste), ...
     shifts_teste/fs*1000, ...
     'rs-','linewidth',1.4);

xlabel('Indice do ciclo');
ylabel('Deslocamento estimado (ms)');

title('Gate 2 - Deslocamentos temporais');

legend('Treinamento','Validacao retrospectiva', ...
       'location','northeast');

grid on;

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E04_Fig03_Deslocamentos.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 12. EXPORTACAO NUMERICA
% ------------------------------------------------------------

fid = fopen(fullfile(pasta_saida, ...
    'G2_E04_Resultados.csv'),'w');

if fid < 0
    error('Falha ao criar CSV.');
end

fprintf(fid, ...
    'ciclo_validacao,fase,rmse_M0_A,rmse_M1_A,rmse_M2_A\n');

for j = 1:K_teste

    for fase = 1:3

        fprintf(fid,'%d,%s,%.8f,%.8f,%.8f\n', ...
            j,nomes{fase}, ...
            RMSE0(j,fase), ...
            RMSE1(j,fase), ...
            RMSE2(j,fase));

    end

end

fclose(fid);

fprintf('\nExperimento 04 concluido.\n');
fprintf('Resultados em:\n%s\n',pasta_saida);
