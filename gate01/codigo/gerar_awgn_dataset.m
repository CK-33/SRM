
% ============================================================
% GERACAO DE DATASET SINTETICO - SRM
% Metodo: Ruido Branco Gaussiano Aditivo (AWGN)
% Ambiente: GNU Octave 11.3.0
%
% Entrada:  results_1.mat ate results_12.mat
% Saida:    36 arquivos MAT (12 origens x 3 SNRs)
% Canais:   correntes das fases A, B e C
% ============================================================

clearvars;
clc;

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

pasta_dados = ...
'CAMINHO_DA_CAMPANHA\TesteNo1_31OCT2026\Dados';

pasta_saida = fullfile(pasta_dados, 'augmentation_awgn');

if ~exist(pasta_saida, 'dir')
    mkdir(pasta_saida);
end

SNR_lista = [30, 40, 50];

% Janela de referencia apos a falha
duracao_janela = 0.020;  % segundos

% Semente-base para reproducibilidade
seed_base = 42;

% Por seguranca, nao sobrescrever arquivos existentes
permitir_sobrescrita = false;

% Arquivo consolidado de resultados
arquivo_csv = fullfile(pasta_saida, ...
                       'metricas_awgn.csv');

if exist(arquivo_csv, 'file') && ~permitir_sobrescrita
    error('CSV ja existe. Renomeie ou remova antes de executar.');
end

fid = fopen(arquivo_csv, 'w');

if fid < 0
    error('Nao foi possivel criar o CSV.');
end

fprintf(fid, ['arquivo_origem,arquivo_sintetico,', ...
              'fase,snr_alvo_db,snr_real_db,', ...
              'rms_original,rms_sintetico,', ...
              'pico_original,pico_sintetico,', ...
              'seed,n_amostras_janela\n']);

fprintf('\n==========================================\n');
fprintf(' GERADOR AWGN TRIFASICO - SRM\n');
fprintf('==========================================\n');

total_gerados = 0;

% ------------------------------------------------------------
% 2. PROCESSAMENTO DOS 12 ARQUIVOS
% ------------------------------------------------------------

for numero = 1:12

    nome_original = sprintf('results_%d.mat', numero);
    caminho_original = fullfile(pasta_dados, nome_original);

    fprintf('\nProcessando: %s\n', nome_original);

    if ~exist(caminho_original, 'file')
        warning('Arquivo nao encontrado: %s', nome_original);
        continue;
    end

    dados = load(caminho_original);

    if ~isfield(dados, 'results')
        warning('Variavel results ausente: %s', nome_original);
        continue;
    end

    results_original = dados.results;

    tempo = results_original.time(:);
    correntes = results_original.signals(3).values;
    evento = results_original.signals(6).values;

    if size(correntes, 2) ~= 3
        warning('Esperadas tres fases em %s', nome_original);
        continue;
    end

    if size(correntes, 1) ~= length(tempo)
        warning('Dimensoes inconsistentes em %s', nome_original);
        continue;
    end

    % Frequencia obtida do vetor temporal
    dt = median(diff(tempo));
    Fs = 1 / dt;

    % Identificacao do primeiro evento
    idx_evento = find(evento(:) == 1, 1, 'first');

    if isempty(idx_evento)
        warning('Evento nao encontrado em %s', nome_original);
        continue;
    end

    % Janela com numero FIXO de amostras
    N = round(duracao_janela * Fs);

    idx_final = idx_evento + N - 1;

    if idx_final > size(correntes, 1)
        warning('Janela excede registro em %s', nome_original);
        continue;
    end

    indices = idx_evento:idx_final;

    fprintf('Fs: %.2f Hz | Evento: %.6f s\n', ...
            Fs, tempo(idx_evento));

    fprintf('Janela: %d amostras\n', N);

    % --------------------------------------------------------
    % 3. GERACAO POR NIVEL DE SNR
    % --------------------------------------------------------

    for k = 1:length(SNR_lista)

        SNR_dB = SNR_lista(k);

        % Semente unica por arquivo e SNR
        seed = seed_base + 1000 * numero + k;

        nome_saida = sprintf( ...
            'results_%d_awgn_%ddB_seed%d.mat', ...
            numero, SNR_dB, seed);

        caminho_saida = fullfile(pasta_saida, nome_saida);

        if exist(caminho_saida, 'file') && ...
           ~permitir_sobrescrita

            warning('Saida ja existe: %s', nome_saida);
            continue;
        end

        % Copia integral da estrutura original
        results = results_original;

        correntes_sinteticas = correntes;

        % Semente reproduzivel
        randn('seed', seed);

        % Metricas temporarias
        metricas = zeros(3, 5);

        for fase = 1:3

            x = correntes(:, fase);

            % Potencia de referencia: 20 ms pos-falha
            x_ref = x(indices);

            Ps = mean(x_ref.^2);

            if Ps <= 0 || ~isfinite(Ps)
                error('Potencia invalida: arquivo %d fase %d', ...
                      numero, fase);
            end

            % Potencia de ruido desejada
            Pn = Ps / (10^(SNR_dB / 10));

            % Ruido para TODO o registro
            ruido = sqrt(Pn) * randn(size(x));

            % Corrente sintetica completa
            y = x + ruido;

            correntes_sinteticas(:, fase) = y;

            % Metricas na janela de referencia
            y_ref = y(indices);
            n_ref = ruido(indices);

            rms_x = sqrt(mean(x_ref.^2));
            rms_y = sqrt(mean(y_ref.^2));

            pico_x = max(abs(x_ref));
            pico_y = max(abs(y_ref));

            snr_real = 10 * log10( ...
                mean(x_ref.^2) / mean(n_ref.^2));

            metricas(fase, :) = ...
                [snr_real, rms_x, rms_y, pico_x, pico_y];

        end

        % Substitui SOMENTE o sinal de corrente trifasico
        results.signals(3).values = correntes_sinteticas;

        % ----------------------------------------------------
        % 4. METADADOS DA GERACAO
        % ----------------------------------------------------

        augmentation = struct();

        augmentation.metodo = 'AWGN trifasico';
        augmentation.arquivo_origem = nome_original;
        augmentation.snr_alvo_db = SNR_dB;
        augmentation.seed = seed;

        augmentation.frequencia_amostragem_hz = Fs;
        augmentation.janela_referencia_s = duracao_janela;
        augmentation.amostras_referencia = N;
        augmentation.indice_evento = idx_evento;
        augmentation.tempo_evento_s = tempo(idx_evento);

        augmentation.canais_modificados = ...
            'results.signals(3).values - fases A B C';

        augmentation.snr_real_db = metricas(:, 1);

        % ----------------------------------------------------
        % 5. SALVAMENTO E VERIFICACAO
        % ----------------------------------------------------

        save('-mat7-binary', caminho_saida, ...
             'results', 'augmentation');

        verificacao = load(caminho_saida);

        assert(isequal(size( ...
            verificacao.results.signals(3).values), ...
            size(correntes)));

        assert(isequal( ...
            verificacao.results.time, ...
            results_original.time));

        assert(isequal( ...
            verificacao.results.signals(6).values, ...
            results_original.signals(6).values));

        % Verifica todos os outros sinais
        for s = 1:length(results_original.signals)

            if s == 3
                continue;
            end

            assert(isequaln( ...
                verificacao.results.signals(s), ...
                results_original.signals(s)));

        end

        % ----------------------------------------------------
        % 6. REGISTRO DAS METRICAS
        % ----------------------------------------------------

        fprintf('\n  SNR alvo: %d dB\n', SNR_dB);

        fprintf('  Fase | SNR real | RMS orig | RMS sint\n');

        for fase = 1:3

            fprintf('    %d  | %8.3f | %8.3f | %8.3f\n', ...
                fase, ...
                metricas(fase, 1), ...
                metricas(fase, 2), ...
                metricas(fase, 3));

            fprintf(fid, ...
                '%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n', ...
                nome_original, nome_saida, ...
                fase, SNR_dB, ...
                metricas(fase, 1), ...
                metricas(fase, 2), ...
                metricas(fase, 3), ...
                metricas(fase, 4), ...
                metricas(fase, 5), ...
                seed, N);

        end

        fprintf('  Salvo: %s\n', nome_saida);

        total_gerados = total_gerados + 1;

    end

end

fclose(fid);

fprintf('\n==========================================\n');
fprintf(' PROCESSAMENTO FINALIZADO\n');
fprintf(' Arquivos gerados: %d\n', total_gerados);
fprintf(' Pasta: %s\n', pasta_saida);
fprintf(' Metricas: %s\n', arquivo_csv);
fprintf('==========================================\n');
