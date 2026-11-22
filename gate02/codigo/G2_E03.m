
% ============================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 03
% REFERENCIA PERIODICA TRIFASICA PRE-EVENTO
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

pasta_saida = fullfile(pasta_dados,'Gate2_Exp03');

if ~exist(pasta_saida,'dir')
    mkdir(pasta_saida);
end

T_ref = 0.01308;

margem_evento = 0.010;
limiar_atividade = 1.0;

fracao_treino = 0.70;

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

assert(size(I,1) == length(t));
assert(size(I,2) >= 3);

idx_evento = find(evento(:,1)>0,1,'first');

if isempty(idx_evento)
    error('Evento nao encontrado.');
end

t_evento = t(idx_evento);
fs = 1/median(diff(t));

N_ciclo = round(T_ref*fs);
T_efetivo = N_ciclo/fs;

fprintf('\n====================================\n');
fprintf('GATE 2 - EXPERIMENTO 03\n');
fprintf('====================================\n');

fprintf('Fs: %.2f Hz\n',fs);
fprintf('Evento: %.6f s\n',t_evento);
fprintf('Amostras por ciclo: %d\n',N_ciclo);
fprintf('Periodo efetivo: %.6f ms\n', ...
        T_efetivo*1000);

% ------------------------------------------------------------
% 3. INTERVALO PRE-EVENTO ATIVO
% ------------------------------------------------------------

idx_ativo = find( ...
    any(abs(I(:,1:3))>limiar_atividade,2) & ...
    t<t_evento-margem_evento, ...
    1,'first');

if isempty(idx_ativo)
    error('Regiao ativa nao encontrada.');
end

idx_final = find( ...
    t<=t_evento-margem_evento, ...
    1,'last');

N_disponivel = idx_final-idx_ativo+1;

K = floor(N_disponivel/N_ciclo);

if K < 5
    error('Menos de cinco ciclos completos disponiveis.');
end

% Descartar amostras finais incompletas
idx_usados = idx_ativo:(idx_ativo+K*N_ciclo-1);

t_usado = t(idx_usados);
X = I(idx_usados,1:3);

fprintf('Inicio utilizado: %.6f s\n',t_usado(1));
fprintf('Fim utilizado: %.6f s\n',t_usado(end));
fprintf('Ciclos completos: %d\n',K);

% ------------------------------------------------------------
% 4. ORGANIZAR CICLOS
% ------------------------------------------------------------

% Dimensoes:
% amostra dentro do ciclo x ciclo x fase

C = zeros(N_ciclo,K,3);

for fase = 1:3

    C(:,:,fase) = reshape( ...
        X(:,fase),N_ciclo,K);

end

tau = (0:N_ciclo-1)'/fs;

% ------------------------------------------------------------
% 5. SEPARACAO TEMPORAL
% ------------------------------------------------------------

K_treino = floor(fracao_treino*K);
K_teste = K-K_treino;

if K_treino < 3 || K_teste < 2
    error('Divisao treino/teste insuficiente.');
end

fprintf('Ciclos de referencia: %d\n',K_treino);
fprintf('Ciclos de validacao: %d\n',K_teste);

% ------------------------------------------------------------
% 6. REFERENCIA POR FASE
% ------------------------------------------------------------

mu = zeros(N_ciclo,3);
sigma = zeros(N_ciclo,3);

for fase = 1:3

    treino = C(:,1:K_treino,fase);

    mu(:,fase) = mean(treino,2);
    sigma(:,fase) = std(treino,0,2);

end

% ------------------------------------------------------------
% 7. VALIDACAO POR CICLO
% ------------------------------------------------------------

RMSE = zeros(K_teste,3);
MAE = zeros(K_teste,3);
RMS_real = zeros(K_teste,3);

for fase = 1:3

    for j = 1:K_teste

        k = K_treino+j;

        observado = C(:,k,fase);
        previsto = mu(:,fase);

        erro = observado-previsto;

        RMSE(j,fase) = sqrt(mean(erro.^2));
        MAE(j,fase) = mean(abs(erro));

        RMS_real(j,fase) = ...
            sqrt(mean(observado.^2));

    end
end

fprintf('\n--- VALIDACAO POR FASE ---\n');

fprintf('Fase | RMSE medio(A) | MAE medio(A)\n');

for fase = 1:3

    fprintf(' %s   | %12.6f | %11.6f\n', ...
        nomes{fase}, ...
        mean(RMSE(:,fase)), ...
        mean(MAE(:,fase)));

end

% ------------------------------------------------------------
% 8. FIGURA 1 - REFERENCIA TRIFASICA
% ------------------------------------------------------------

figure(1,'position',[100 100 1200 750]);

for fase = 1:3

    subplot(3,1,fase);

    plot(tau*1000,mu(:,fase), ...
         cores{fase},'linewidth',1.5);

    ylabel('Corrente (A)');
    title(['Referencia - Fase ' nomes{fase}]);

    grid on;

    xlim([0 T_efetivo*1000]);

    if fase == 3
        xlabel('Tempo relativo no ciclo (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E03_Fig01_Referencia.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 9. FIGURA 2 - MEDIA E DISPERSAO
% ------------------------------------------------------------

figure(2,'position',[100 100 1200 850]);

for fase = 1:3

    subplot(3,1,fase);

    superior = mu(:,fase)+2*sigma(:,fase);
    inferior = mu(:,fase)-2*sigma(:,fase);

    plot(tau*1000,superior,'k--', ...
         'linewidth',0.8);

    hold on;

    plot(tau*1000,inferior,'k--', ...
         'linewidth',0.8);

    plot(tau*1000,mu(:,fase), ...
         cores{fase},'linewidth',1.5);

    ylabel('Corrente (A)');
    title(['Dispersao - Fase ' nomes{fase}]);

    grid on;

    xlim([0 T_efetivo*1000]);

    if fase == 3
        xlabel('Tempo relativo no ciclo (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E03_Fig02_Dispersao.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 10. FIGURA 3 - CICLOS DE VALIDACAO
% ------------------------------------------------------------

figure(3,'position',[100 100 1200 850]);

for fase = 1:3

    subplot(3,1,fase);
    hold on;

    for j = 1:K_teste

        k = K_treino+j;

        plot(tau*1000,C(:,k,fase), ...
             'color',[0.65 0.65 0.65], ...
             'linewidth',0.7);

    end

    plot(tau*1000,mu(:,fase), ...
         cores{fase},'linewidth',1.7);

    ylabel('Corrente (A)');
    title(['Validacao - Fase ' nomes{fase}]);

    grid on;

    xlim([0 T_efetivo*1000]);

    if fase == 3
        xlabel('Tempo relativo no ciclo (ms)');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E03_Fig03_Validacao.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 11. FIGURA 4 - ERRO POR CICLO
% ------------------------------------------------------------

figure(4,'position',[100 100 1100 550]);

hold on;

for fase = 1:3

    plot(1:K_teste,RMSE(:,fase), ...
         [cores{fase} '-o'], ...
         'linewidth',1.4);

end

xlabel('Ciclo de validacao');
ylabel('RMSE (A)');

title('Gate 2 - Erro da referencia por ciclo');

legend('Fase A','Fase B','Fase C', ...
       'location','northeast');

grid on;

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E03_Fig04_RMSE.png'), ...
    '-dpng','-r250');

% ------------------------------------------------------------
% 12. EXPORTAR METRICAS
% ------------------------------------------------------------

fid = fopen(fullfile(pasta_saida, ...
    'G2_E03_Validacao.csv'),'w');

if fid < 0
    error('Falha ao criar CSV.');
end

fprintf(fid, ...
    'ciclo,fase,rmse_A,mae_A,rms_real_A\n');

for j = 1:K_teste

    for fase = 1:3

        fprintf(fid,'%d,%s,%.8f,%.8f,%.8f\n', ...
            j,nomes{fase}, ...
            RMSE(j,fase), ...
            MAE(j,fase), ...
            RMS_real(j,fase));

    end
end

fclose(fid);

fprintf('\nExperimento 03 concluido.\n');
fprintf('Resultados em:\n%s\n',pasta_saida);
