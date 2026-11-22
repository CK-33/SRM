
% ============================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 02A - ESTABILIDADE TEMPORAL DA PERIODICIDADE
% GNU OCTAVE 11.3.0
% ============================================================

clear; close all; clc;

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

pasta_dados = ['PASTA_DO_USUARIO\Documents\' ...
               'Pesquisa_SRM\TesteNo1_31OCT2026\' ...
               'TesteNo1_31OCT2026\Dados'];

arquivo = fullfile(pasta_dados, 'results_1.mat');

pasta_saida = fullfile(pasta_dados, 'Gate2_Exp02A');

if ~exist(pasta_saida, 'dir')
    mkdir(pasta_saida);
end

janela_s = 0.040;
passo_s  = 0.010;

periodo_min_s = 0.010;
periodo_max_s = 0.016;

margem_evento_s = 0.010;

% ------------------------------------------------------------
% 2. CARREGAMENTO
% ------------------------------------------------------------

dados = load(arquivo);
r = dados.results;

t = r.time(:);
I = r.signals(3).values;
evento = r.signals(6).values;

iB = I(:,2);

idx_evento = find(evento(:,1) > 0, 1, 'first');

if isempty(idx_evento)
    error('Evento nao encontrado.');
end

t_evento = t(idx_evento);

fs = 1/median(diff(t));

fprintf('\n=== GATE 2 - EXPERIMENTO 02A ===\n');
fprintf('Fs: %.2f Hz\n', fs);
fprintf('Evento: %.6f s\n', t_evento);

% ------------------------------------------------------------
% 3. IDENTIFICAR INTERVALO PRE-EVENTO UTIL
% ------------------------------------------------------------

% O sinal pode conter amostras nulas antes da energizacao.
% Este criterio e exploratorio e deve ser conferido no grafico.

limiar_atividade_A = 1.0;

idx_ativo = find(abs(iB) > limiar_atividade_A & ...
                 t < t_evento - margem_evento_s, ...
                 1, 'first');

if isempty(idx_ativo)
    error('Nao foi encontrada atividade pre-evento.');
end

t_inicio = t(idx_ativo);
t_fim = t_evento - margem_evento_s;

fprintf('Inicio ativo: %.6f s\n', t_inicio);
fprintf('Fim utilizado: %.6f s\n', t_fim);

% ------------------------------------------------------------
% 4. PARAMETROS EM AMOSTRAS
% ------------------------------------------------------------

Njanela = round(janela_s*fs);
Npasso = round(passo_s*fs);

idx_inicio = find(t >= t_inicio, 1, 'first');
idx_fim = find(t <= t_fim, 1, 'last');

centros = [];
periodos = [];
frequencias = [];
correlacoes = [];
rms_janelas = [];

% ------------------------------------------------------------
% 5. ANALISE POR JANELAS
% ------------------------------------------------------------

for ini = idx_inicio:Npasso:(idx_fim-Njanela+1)

    fim = ini + Njanela - 1;

    x_original = iB(ini:fim);
    x = x_original - mean(x_original);

    N = length(x);

    if sum(x.^2) <= eps
        continue;
    end

    % Autocorrelacao linear por FFT
    nfft = 2^nextpow2(2*N-1);

    X = fft(x, nfft);

    R = real(ifft(abs(X).^2));
    R = R(1:N);

    R = R/R(1);

    lags = (0:N-1)'/fs;

    % Busca limitada ao intervalo candidato
    idx_busca = find(lags >= periodo_min_s & ...
                     lags <= periodo_max_s);

    if isempty(idx_busca)
        continue;
    end

    % Identificar maximos locais
    candidatos = [];

    for k = 2:length(R)-1

        if R(k) > R(k-1) && R(k) >= R(k+1)

            if lags(k) >= periodo_min_s && ...
               lags(k) <= periodo_max_s

                candidatos(end+1) = k;
            end
        end
    end

    if isempty(candidatos)
        continue;
    end

    [corr_max, pos] = max(R(candidatos));

    idx_pico = candidatos(pos);

    T = lags(idx_pico);
    f = 1/T;

    centro = (t(ini)+t(fim))/2;

    centros(end+1,1) = centro;
    periodos(end+1,1) = T;
    frequencias(end+1,1) = f;
    correlacoes(end+1,1) = corr_max;

    rms_janelas(end+1,1) = ...
        sqrt(mean(x_original.^2));

end

% ------------------------------------------------------------
% 6. RESULTADOS NUMERICOS
% ------------------------------------------------------------

if isempty(periodos)
    error('Nenhum periodo candidato identificado.');
end

fprintf('\n--- RESULTADOS POR JANELA ---\n');

fprintf(['Centro(s) | Periodo(ms) | ' ...
         'Frequencia(Hz) | Correlacao | RMS(A)\n']);

for k = 1:length(periodos)

    fprintf('%.6f | %.4f | %.4f | %.5f | %.4f\n', ...
        centros(k), ...
        periodos(k)*1000, ...
        frequencias(k), ...
        correlacoes(k), ...
        rms_janelas(k));
end

fprintf('\n--- ESTATISTICAS ---\n');

fprintf('Periodo medio: %.6f ms\n', ...
        mean(periodos)*1000);

fprintf('Desvio padrao: %.6f ms\n', ...
        std(periodos)*1000);

fprintf('Frequencia media: %.6f Hz\n', ...
        mean(frequencias));

fprintf('Correlacao media: %.6f\n', ...
        mean(correlacoes));

% ------------------------------------------------------------
% 7. GRAFICO - PERIODICIDADE AO LONGO DO TEMPO
% ------------------------------------------------------------

figure(1);

plot(centros, periodos*1000, 'bo-', ...
     'LineWidth', 1.4, 'MarkerSize', 6);

hold on;

plot([centros(1) centros(end)], ...
     [13.08 13.08], 'r--', 'LineWidth', 1.2);

xlabel('Tempo central da janela (s)');
ylabel('Periodo estimado (ms)');

title('Gate 2 - Estabilidade temporal do periodo');

legend('Periodo estimado', ...
       'Referencia: 13.08 ms', ...
       'location', 'northeast');

grid on;

print(fullfile(pasta_saida, ...
      'G2_E02A_Fig01_Periodo.png'), '-dpng', '-r200');

% ------------------------------------------------------------
% 8. GRAFICO - CORRELACAO
% ------------------------------------------------------------

figure(2);

plot(centros, correlacoes, 'mo-', ...
     'LineWidth', 1.4, 'MarkerSize', 6);

xlabel('Tempo central da janela (s)');
ylabel('Autocorrelacao normalizada');

title('Gate 2 - Intensidade da periodicidade');

grid on;

print(fullfile(pasta_saida, ...
      'G2_E02A_Fig02_Correlacao.png'), '-dpng', '-r200');

% ------------------------------------------------------------
% 9. GRAFICO - FREQUENCIA E RMS
% ------------------------------------------------------------

figure(3);

subplot(2,1,1);

plot(centros, frequencias, 'b-o', ...
     'LineWidth', 1.3);

ylabel('Frequencia (Hz)');
title('Gate 2 - Frequencia de repeticao');
grid on;

subplot(2,1,2);

plot(centros, rms_janelas, 'k-o', ...
     'LineWidth', 1.3);

xlabel('Tempo central da janela (s)');
ylabel('RMS (A)');
title('RMS por janela');
grid on;

print(fullfile(pasta_saida, ...
      'G2_E02A_Fig03_Frequencia_RMS.png'), ...
      '-dpng', '-r200');

% ------------------------------------------------------------
% 10. EXPORTAR METRICAS
% ------------------------------------------------------------

fid = fopen(fullfile(pasta_saida, ...
             'G2_E02A_Resultados.csv'), 'w');

if fid < 0
    error('Falha ao criar arquivo CSV.');
end

fprintf(fid, ...
    'centro_s,periodo_ms,frequencia_hz,correlacao,rms_A\n');

for k = 1:length(periodos)

    fprintf(fid, '%.8f,%.8f,%.8f,%.8f,%.8f\n', ...
        centros(k), ...
        periodos(k)*1000, ...
        frequencias(k), ...
        correlacoes(k), ...
        rms_janelas(k));
end

fclose(fid);

fprintf('\nExperimento 02A concluido.\n');
fprintf('Resultados em: %s\n', pasta_saida);
