
1;

% ============================================================
% PESQUISA SRM
% GATE 4 - EXPERIMENTO 01
% Expansao controlada por AWGN
% Versao 2 - Localizacao automatica dos dados
% GNU Octave 11.3.0
% ============================================================

% ============================================================
% FUNCOES AUXILIARES
% ============================================================

function ok = verificar_pasta(pasta, registros)

    ok = true;

    for r = registros

        arquivo = fullfile(pasta, ...
            sprintf('results_%d.mat', r));

        if exist(arquivo, 'file') ~= 2
            ok = false;
            return;
        end

    end

end


function encontrada = localizar_dados(pasta, registros, nivel)

    encontrada = '';

    if nivel < 0
        return;
    end

    if exist(pasta, 'dir') ~= 7
        return;
    end

    if verificar_pasta(pasta, registros)
        encontrada = pasta;
        return;
    end

    if nivel == 0
        return;
    end

    itens = dir(pasta);

    for k = 1:numel(itens)

        if ~itens(k).isdir
            continue;
        end

        nome = itens(k).name;

        if strcmp(nome, '.') || strcmp(nome, '..')
            continue;
        end

        if nome(1) == '.'
            continue;
        end

        % Nao procurar em pastas de experimentos anteriores

        if strncmpi(nome, 'Gate', 4)
            continue;
        end

        subpasta = fullfile(pasta, nome);

        encontrada = localizar_dados( ...
            subpasta, registros, nivel - 1);

        if ~isempty(encontrada)
            return;
        end

    end

end


function valor = calcular_rms(x)

    valor = sqrt(mean(x.^2));

end


function salvar_figura(h, destino)

    set(h, 'paperpositionmode', 'auto');

    print(h, destino, '-dpng', '-r150');

    close(h);

    fprintf('Figura exportada: %s\n', destino);

end


function acf = calcular_acf(x, maxlag)

    x = x(:) - mean(x);

    energia = sum(x.^2);

    acf = zeros(maxlag + 1, 1);

    for lag = 0:maxlag

        acf(lag+1) = sum( ...
            x(1:end-lag) .* ...
            x(1+lag:end)) / energia;

    end

end


% ============================================================
% PROGRAMA PRINCIPAL
% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('==============================================\n');
fprintf(' PESQUISA SRM\n');
fprintf(' GATE 4 - EXPERIMENTO 01\n');
fprintf(' Expansao controlada por AWGN\n');
fprintf(' Versao 2\n');
fprintf('==============================================\n');


% ------------------------------------------------------------
% 1. CONFIGURACOES
% ------------------------------------------------------------

PROJECT_ROOT = [
    'PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
    'TesteNo1_31OCT2026'
];

ROOT = fullfile(PROJECT_ROOT, 'Data and Scripts');

% Se desejar informar manualmente o caminho dos dados,
% insira-o entre as aspas abaixo.
% Caso contrario, deixe vazio.

DATA_ROOT = '';

% Particionamento herdado do Gate 3

REG_TREINO = [1 2 4 5 7 8 10 11];

REG_VALID = [3 9];

REG_TESTE = [6 12];

REGISTROS = sort([REG_TREINO REG_VALID]);

SNR_LEVELS = [50 40 30];

FASES = {'A', 'B', 'C'};

SEED_BASE = 42;

FS_REF = 200000;

T_REF = 0.01308;

TOL_SNR = 0.5;

T_EVENTOS = [
    1.0017;
    1.0019;
    1.0021;
    1.0024;
    1.0026;
    1.0030;
    1.0040;
    1.0050;
    1.0060;
    1.0070;
    1.0075;
    1.0080
];

assert(isempty(intersect(REGISTROS, REG_TESTE)));

assert(numel(REGISTROS) == 10);


% ------------------------------------------------------------
% 2. LOCALIZACAO AUTOMATICA DOS DADOS
% ------------------------------------------------------------

fprintf('\nLocalizando os dados originais...\n');

if isempty(DATA_ROOT)

    % Verificar primeiro o diretorio de trabalho

    if verificar_pasta(ROOT, REGISTROS)

        DATA_ROOT = ROOT;

    % Verificar o diretorio atual do Octave

    elseif verificar_pasta(pwd, REGISTROS)

        DATA_ROOT = pwd;

    else

        % Procurar dentro da pasta do projeto

        DATA_ROOT = localizar_dados( ...
            PROJECT_ROOT, REGISTROS, 5);

    end

end

% Caso a busca automatica nao encontre todos os arquivos

if isempty(DATA_ROOT) || ...
   ~verificar_pasta(DATA_ROOT, REGISTROS)

    fprintf('\nBusca automatica sem sucesso.\n');

    fprintf(['Informe a pasta que contem os arquivos ' ...
             'results_1.mat, results_2.mat, etc.\n']);

    DATA_ROOT = input('Caminho da pasta: ', 's');

    DATA_ROOT = strtrim(DATA_ROOT);

    DATA_ROOT = strrep(DATA_ROOT, '"', '');

end

if ~verificar_pasta(DATA_ROOT, REGISTROS)

    fprintf('\nArquivos ausentes:\n');

    for r = REGISTROS

        arquivo = fullfile(DATA_ROOT, ...
            sprintf('results_%d.mat', r));

        if exist(arquivo, 'file') ~= 2
            fprintf('  %s\n', arquivo);
        end

    end

    error('Nao foi possivel localizar os dados autorizados.');

end

fprintf('\nDados localizados com sucesso!\n');

fprintf('Diretorio dos dados:\n%s\n', DATA_ROOT);

fprintf('Diretorio de trabalho:\n%s\n', ROOT);

fprintf('Registros autorizados: %d\n', numel(REGISTROS));

fprintf('Registros 06 e 12: RESERVADOS\n');


% ------------------------------------------------------------
% 3. PREPARACAO DOS DIRETORIOS
% ------------------------------------------------------------

if exist(ROOT, 'dir') ~= 7
    mkdir(ROOT);
end

OUT = fullfile(ROOT, 'Gate4_Exp01');

% Se a pasta anterior estiver vazia, reutiliza-la.
% Se houver resultados, criar nova execucao numerada.

OUT_BASE = OUT;

execucao = 1;

while exist(OUT, 'dir') == 7

    itens = dir(OUT);

    nomes = {itens.name};

    nomes = nomes(~ismember(nomes, {'.', '..'}));

    if isempty(nomes)
        break;
    end

    execucao = execucao + 1;

    OUT = sprintf('%s_Run%02d', OUT_BASE, execucao);

end

if exist(OUT, 'dir') ~= 7
    mkdir(OUT);
end

DATA_OUT = fullfile(OUT, 'Dados_AWGN');

FIG_OUT = fullfile(OUT, 'Figuras');

mkdir(DATA_OUT);

mkdir(FIG_OUT);

fprintf('\nDiretorio de saida:\n%s\n', OUT);


% ------------------------------------------------------------
% 4. ARQUIVOS DE RASTREABILIDADE
% ------------------------------------------------------------

fid_manifesto = fopen( ...
    fullfile(OUT, 'G4_E01_Manifesto.csv'), 'w');

fid_metricas = fopen( ...
    fullfile(OUT, 'G4_E01_Metricas.csv'), 'w');

assert(fid_manifesto >= 0);

assert(fid_metricas >= 0);

fprintf(fid_manifesto, ...
    ['arquivo,registro_origem,grupo,metodo,' ...
     'snr_nominal_dB,fs_Hz,n_amostras,t_evento_s,' ...
     'seed_A,seed_B,seed_C\n']);

fprintf(fid_metricas, ...
    ['registro,grupo,fase,snr_nominal_dB,seed,' ...
     'snr_pre_dB,snr_global_dB,' ...
     'rms_original_pre_A,rms_awgn_pre_A,' ...
     'rms_original_pos_A,rms_awgn_pos_A,' ...
     'rms_ruido_A,media_ruido_A,' ...
     'erro_rms_pre_pct,erro_rms_pos_pct,' ...
     'pico_original_A,pico_awgn_A,' ...
     'correlacao_original_awgn,' ...
     'autocorrelacao_ruido_lag1,' ...
     'assimetria_ruido,curtose_ruido,' ...
     'correlacao_ruido_AB,' ...
     'correlacao_ruido_AC,' ...
     'correlacao_ruido_BC,' ...
     'snr_aprovado\n']);

AUDIT = [];

EXEMPLO = struct();

N_GERADOS = 0;


% ------------------------------------------------------------
% 5. PROCESSAMENTO DOS REGISTROS
% ------------------------------------------------------------

try

for r = REGISTROS

    if any(REG_TREINO == r)

        grupo = 'treino';

    elseif any(REG_VALID == r)

        grupo = 'validacao';

    else

        error('Registro nao autorizado.');

    end

    fprintf('\n');
    fprintf('----------------------------------------------\n');
    fprintf('Registro %02d | Grupo: %s\n', r, grupo);
    fprintf('----------------------------------------------\n');

    % --------------------------------------------------------
    % 5.1. Carregamento local
    % --------------------------------------------------------

    arquivo = fullfile(DATA_ROOT, ...
        sprintf('results_%d.mat', r));

    S = load(arquivo, 'results');

    if ~isfield(S, 'results')
        error('Variavel results ausente no registro %d.', r);
    end

    original = S.results;

    clear S;

    assert(isfield(original, 'time'));

    assert(isfield(original, 'signals'));

    assert(numel(original.signals) >= 6);


    % --------------------------------------------------------
    % 5.2. Extracao das correntes
    % --------------------------------------------------------

    t = double(original.time(:));

    I_original = double(original.signals(3).values);

    evento = double(original.signals(6).values);

    N = numel(t);

    transposto = false;

    if size(I_original, 1) ~= N && ...
       size(I_original, 2) == N

        I_original = I_original.';

        transposto = true;

    end

    assert(size(I_original, 1) == N);

    assert(size(I_original, 2) == 3);

    assert(all(isfinite(I_original(:))));

    assert(all(isfinite(t)));

    evento = evento(:);

    assert(numel(evento) == N);

    assert(all(isfinite(evento)));



    % --------------------------------------------------------
    % 5.3. Auditoria temporal
    % Versao 3 - Irregularidades pontuais documentadas
    % --------------------------------------------------------

    dt = diff(t);

    assert(all(isfinite(dt)));

    if any(dt <= 0)

        error(['Base temporal nao estritamente crescente ' ...
               'no registro %d.'], r);

    end

    dt_med = median(dt);

    Fs = 1 / dt_med;

    erro_fs = abs(Fs - FS_REF) / FS_REF;

    if erro_fs > 0.001

        error('Frequencia nominal inesperada.');

    end

    % Caracterizacao das irregularidades

    erro_dt = abs(dt - dt_med) / dt_med;

    idx_irregulares = find(erro_dt > 0.001);

    n_irregulares = numel(idx_irregulares);

    frac_irregulares = n_irregulares / numel(dt);

    pct_irregulares = 100 * frac_irregulares;

    n_quase_nulos = sum(dt < 0.001 * dt_med);

    n_intervalos_longos = sum(dt > 1.001 * dt_med);

    N_uniforme = round((t(end)-t(1)) / dt_med) + 1;

    n_excedentes = N - N_uniforme;

    % Auditoria no terminal

    fprintf('\nAuditoria da base temporal:\n');

    fprintf('  Fs nominal: %.6f Hz\n', Fs);

    fprintf('  dt mediano: %.9f us\n', dt_med*1e6);

    fprintf('  dt minimo: %.12e s\n', min(dt));

    fprintf('  dt maximo: %.9f us\n', max(dt)*1e6);

    fprintf('  Intervalos irregulares: %d\n', ...
        n_irregulares);

    fprintf('  Percentual irregular: %.8f %%\n', ...
        pct_irregulares);

    fprintf('  Intervalos quase nulos: %d\n', ...
        n_quase_nulos);

    fprintf('  Intervalos longos: %d\n', ...
        n_intervalos_longos);

    fprintf('  Amostras observadas: %d\n', N);

    fprintf('  Amostras uniformes esperadas: %d\n', ...
        N_uniforme);

    fprintf('  Excedente de amostras: %d\n', ...
        n_excedentes);

    % Limite operacional provisório:
    % interromper se mais de 0.1% dos intervalos
    % forem irregulares.
    %
    % Este limite nao certifica adequacao para FFT/STFT.

    if frac_irregulares > 0.001

        error(['Irregularidade temporal acima do limite ' ...
               'operacional no registro %d.'], r);

    end

    if n_intervalos_longos > 0

        error(['Intervalos superiores ao passo nominal ' ...
               'detectados no registro %d.'], r);

    end

    if n_irregulares > 0

        fprintf(['  AVISO: irregularidades pontuais ' ...
                 'identificadas.\n']);

        fprintf(['  Base temporal original preservada ' ...
                 'para a geracao AWGN.\n']);

        fprintf(['  Regularizacao temporal sera ' ...
                 'avaliada no E02.\n']);

    end

    % Registro permanente da auditoria temporal

    arquivo_auditoria = fullfile(OUT, ...
        'G4_E01_Auditoria_Temporal.csv');

    novo_csv = exist(arquivo_auditoria, 'file') ~= 2;

    fid_t = fopen(arquivo_auditoria, 'a');

    assert(fid_t >= 0);

    if novo_csv

        fprintf(fid_t, ...
            ['registro,n_amostras,n_uniforme,' ...
             'n_excedentes,fs_nominal_Hz,' ...
             'dt_min_s,dt_med_s,dt_max_s,' ...
             'n_irregulares,pct_irregulares,' ...
             'n_quase_nulos,n_intervalos_longos\n']);

    end

    fprintf(fid_t, ...
        ['%d,%d,%d,%d,%.12g,%.12g,%.12g,' ...
         '%.12g,%d,%.12g,%d,%d\n'], ...
        r, N, N_uniforme, n_excedentes, Fs, ...
        min(dt), dt_med, max(dt), ...
        n_irregulares, pct_irregulares, ...
        n_quase_nulos, n_intervalos_longos);

    fclose(fid_t);



    % --------------------------------------------------------
    % 5.4. Identificacao do evento
    % --------------------------------------------------------

    idx_evento = find(evento > 0.5, 1, 'first');

    if isempty(idx_evento)

        error('Evento nao encontrado no registro %d.', r);

    end

    t_evento = t(idx_evento);

    % Conferir com os instantes conhecidos dos Gates anteriores

    erro_evento = abs(t_evento - T_EVENTOS(r));

    if erro_evento > 2 * dt_med

        warning(['Instante de evento diferente da referencia ' ...
                 'historica no registro %d.'], r);

    end

    idx_pre = t < t_evento;

    idx_pos = t >= t_evento;

    assert(sum(idx_pre) >= round(T_REF * Fs));

    assert(sum(idx_pos) > 0);

    fprintf('Amostras: %d\n', N);

    fprintf('Fs: %.6f Hz\n', Fs);

    fprintf('Evento: %.8f s\n', t_evento);

    fprintf('Amostras pre: %d\n', sum(idx_pre));

    fprintf('Amostras pos: %d\n', sum(idx_pos));


    % --------------------------------------------------------
    % 5.5. Potencia de referencia
    % --------------------------------------------------------

    P_pre = mean(I_original(idx_pre,:).^2, 1);

    assert(all(P_pre > 0));

    fprintf('RMS pre A: %.6f A\n', sqrt(P_pre(1)));

    fprintf('RMS pre B: %.6f A\n', sqrt(P_pre(2)));

    fprintf('RMS pre C: %.6f A\n', sqrt(P_pre(3)));


    % --------------------------------------------------------
    % 5.6. Geracao das variantes AWGN
    % --------------------------------------------------------

    for snr_db = SNR_LEVELS

        I_awgn = zeros(N, 3);

        RU = zeros(N, 3);

        seeds = zeros(1, 3);

        for f = 1:3

            seed = SEED_BASE + ...
                   10000*r + ...
                   10*snr_db + f;

            seeds(f) = seed;

            rng(seed, 'twister');

            P_ruido = P_pre(f) / ...
                      10^(snr_db/10);

            sigma = sqrt(P_ruido);

            ruido = sigma * randn(N, 1);

            RU(:,f) = ruido;

            I_awgn(:,f) = I_original(:,f) + ruido;

        end

        assert(all(isfinite(I_awgn(:))));

        CR = corrcoef(RU);

        snr_pre_fases = zeros(1,3);

        snr_global_fases = zeros(1,3);


        % ----------------------------------------------------
        % 5.7. Auditoria das fases
        % ----------------------------------------------------

        for f = 1:3

            x = I_original(:,f);

            y = I_awgn(:,f);

            ruido = y - x;

            P_ruido_pre = mean(ruido(idx_pre).^2);

            P_ruido_global = mean(ruido.^2);

            SNR_pre = 10*log10( ...
                mean(x(idx_pre).^2) / P_ruido_pre);

            SNR_global = 10*log10( ...
                mean(x.^2) / P_ruido_global);

            rms_x_pre = calcular_rms(x(idx_pre));

            rms_y_pre = calcular_rms(y(idx_pre));

            rms_x_pos = calcular_rms(x(idx_pos));

            rms_y_pos = calcular_rms(y(idx_pos));

            erro_pre = 100 * ...
                (rms_y_pre-rms_x_pre) / ...
                max(rms_x_pre, eps);

            erro_pos = 100 * ...
                (rms_y_pos-rms_x_pos) / ...
                max(rms_x_pos, eps);

            pico_x = max(abs(x));

            pico_y = max(abs(y));

            C = corrcoef(x,y);

            correlacao = C(1,2);

            mu_ruido = mean(ruido);

            ruido_c = ruido - mu_ruido;

            m2 = mean(ruido_c.^2);

            m3 = mean(ruido_c.^3);

            m4 = mean(ruido_c.^4);

            assimetria = m3 / m2^(3/2);

            curtose = m4 / m2^2;

            acf1 = sum( ...
                ruido_c(1:end-1) .* ...
                ruido_c(2:end)) / ...
                sum(ruido_c.^2);

            aprovado = abs(SNR_pre-snr_db) <= TOL_SNR;

            snr_pre_fases(f) = SNR_pre;

            snr_global_fases(f) = SNR_global;

            AUDIT(end+1,:) = [
                r, snr_db, f, ...
                SNR_pre, SNR_global, ...
                erro_pre, erro_pos, ...
                correlacao, acf1, ...
                assimetria, curtose, ...
                aprovado
            ];

            fprintf(fid_metricas, ...
                '%d,%s,%s,%d,%d', ...
                r, grupo, FASES{f}, ...
                snr_db, seeds(f));

            valores = [
                SNR_pre, SNR_global, ...
                rms_x_pre, rms_y_pre, ...
                rms_x_pos, rms_y_pos, ...
                sqrt(P_ruido_global), ...
                mu_ruido, ...
                erro_pre, erro_pos, ...
                pico_x, pico_y, ...
                correlacao, acf1, ...
                assimetria, curtose, ...
                CR(1,2), CR(1,3), CR(2,3)
            ];

            fprintf(fid_metricas, ',%.12g', valores);

            fprintf(fid_metricas, ',%d\n', aprovado);

            if ~aprovado

                error(['SNR fora da tolerancia: ' ...
                       'registro %d, fase %s, nivel %d dB.'], ...
                       r, FASES{f}, snr_db);

            end

        end


        % ----------------------------------------------------
        % 5.8. Preservacao da estrutura original
        % ----------------------------------------------------

        results = original;

        if transposto

            results.signals(3).values = I_awgn.';

        else

            results.signals(3).values = I_awgn;

        end

        assert(isequaln(results.time, original.time));

        for canal = 1:numel(original.signals)

            if canal ~= 3

                assert(isequaln( ...
                    results.signals(canal), ...
                    original.signals(canal)));

            end

        end


        % ----------------------------------------------------
        % 5.9. Metadados
        % ----------------------------------------------------

        metadata = struct();

        metadata.gate = 4;

        metadata.experimento = 'E01';

        metadata.versao = 2;

        metadata.registro_origem = r;

        metadata.grupo = grupo;

        metadata.metodo = 'AWGN';

        metadata.snr_nominal_dB = snr_db;

        metadata.snr_pre_dB = snr_pre_fases;

        metadata.snr_global_dB = snr_global_fases;

        metadata.snr_referencia = ...
            'Potencia media pre-evento por fase';

        metadata.seeds = seeds;

        metadata.fs_Hz = Fs;

        metadata.n_amostras = N;

        metadata.t_evento_s = t_evento;

        metadata.canal_modificado = ...
            'signals(3).values';

        metadata.nova_simulacao_fisica = false;

        metadata.nota = [ ...
            'Variante estatistica das correntes medidas. ' ...
            'Demais canais mantidos como no original. ' ...
            'Nao constitui simulacao fisica independente.'];


        % ----------------------------------------------------
        % 5.10. Salvamento
        % ----------------------------------------------------

        nome = sprintf( ...
            'results_%02d_AWGN_%ddB.mat', ...
            r, snr_db);

        destino = fullfile(DATA_OUT, nome);

        if exist(destino, 'file') == 2

            error('Arquivo derivado ja existente.');

        end

        save('-mat7-binary', destino, ...
            'results', 'metadata');

        assert(exist(destino, 'file') == 2);

        N_GERADOS = N_GERADOS + 1;

        fprintf(fid_manifesto, ...
            ['%s,%d,%s,AWGN,%d,%.9f,%d,' ...
             '%.9f,%d,%d,%d\n'], ...
             nome, r, grupo, snr_db, Fs, N, ...
             t_evento, seeds(1), seeds(2), seeds(3));

        fflush(fid_manifesto);

        fflush(fid_metricas);

        fprintf('\nAWGN nominal: %d dB\n', snr_db);

        fprintf('  SNR pre A/B/C: %.4f / %.4f / %.4f\n', ...
            snr_pre_fases(1), ...
            snr_pre_fases(2), ...
            snr_pre_fases(3));

        fprintf('  SNR global A/B/C: %.4f / %.4f / %.4f\n', ...
            snr_global_fases(1), ...
            snr_global_fases(2), ...
            snr_global_fases(3));

        fprintf('  Arquivo salvo: %s\n', nome);


        % ----------------------------------------------------
        % 5.11. Exemplo destinado as figuras
        % ----------------------------------------------------

        if r == 1

            EXEMPLO.t = t;

            EXEMPLO.t_evento = t_evento;

            EXEMPLO.original = I_original;

            campo = sprintf('snr%d', snr_db);

            EXEMPLO.(campo) = I_awgn;

            campo_ruido = sprintf('ruido%d', snr_db);

            EXEMPLO.(campo_ruido) = RU;

        end

        clear results metadata I_awgn RU;

    end

    clear original I_original evento;

end

catch erro

    fclose(fid_manifesto);

    fclose(fid_metricas);

    rethrow(erro);

end

fclose(fid_manifesto);

fclose(fid_metricas);


% ------------------------------------------------------------
% 6. AUDITORIA GLOBAL
% ------------------------------------------------------------

assert(N_GERADOS == 30);

assert(size(AUDIT,1) == 90);

assert(all(AUDIT(:,12) == 1));

fprintf('\n');
fprintf('==============================================\n');
fprintf(' AUDITORIA GLOBAL\n');
fprintf('==============================================\n');

fprintf('Variantes geradas: %d\n', N_GERADOS);

fprintf('Realizacoes de ruido: %d\n', size(AUDIT,1));

fprintf('Realizacoes aprovadas: %d/%d\n', ...
    sum(AUDIT(:,12)), size(AUDIT,1));

fid_resumo = fopen( ...
    fullfile(OUT, 'G4_E01_Resumo.csv'), 'w');

assert(fid_resumo >= 0);

fprintf(fid_resumo, ...
    ['snr_nominal_dB,n_realizacoes,' ...
     'snr_pre_medio_dB,snr_global_medio_dB,' ...
     'erro_maximo_snr_dB,' ...
     'erro_rms_pre_medio_abs_pct,' ...
     'erro_rms_pos_medio_abs_pct,' ...
     'correlacao_media,' ...
     'acf1_media_abs,' ...
     'assimetria_media_abs,' ...
     'curtose_media\n']);

for snr_db = SNR_LEVELS

    m = AUDIT(:,2) == snr_db;

    snr_pre_medio = mean(AUDIT(m,4));

    snr_global_medio = mean(AUDIT(m,5));

    erro_snr = max(abs(AUDIT(m,4)-snr_db));

    erro_rms_pre = mean(abs(AUDIT(m,6)));

    erro_rms_pos = mean(abs(AUDIT(m,7)));

    corr_media = mean(AUDIT(m,8));

    acf_media = mean(abs(AUDIT(m,9)));

    assim_media = mean(abs(AUDIT(m,10)));

    curt_media = mean(AUDIT(m,11));

    fprintf('\nSNR nominal: %d dB\n', snr_db);

    fprintf('  SNR pre medio: %.6f dB\n', ...
        snr_pre_medio);

    fprintf('  SNR global medio: %.6f dB\n', ...
        snr_global_medio);

    fprintf('  Erro maximo SNR: %.6f dB\n', erro_snr);

    fprintf('  Erro RMS pre medio absoluto: %.6f %%\n', ...
        erro_rms_pre);

    fprintf('  Erro RMS pos medio absoluto: %.6f %%\n', ...
        erro_rms_pos);

    fprintf('  Correlacao media: %.8f\n', corr_media);

    fprintf('  ACF lag 1 media absoluta: %.8f\n', ...
        acf_media);

    fprintf('  Assimetria media absoluta: %.8f\n', ...
        assim_media);

    fprintf('  Curtose media: %.8f\n', curt_media);

    fprintf(fid_resumo, '%d,%d', snr_db, sum(m));

    fprintf(fid_resumo, ',%.12g', [
        snr_pre_medio, ...
        snr_global_medio, ...
        erro_snr, ...
        erro_rms_pre, ...
        erro_rms_pos, ...
        corr_media, ...
        acf_media, ...
        assim_media, ...
        curt_media
    ]);

    fprintf(fid_resumo, '\n');

end

fclose(fid_resumo);


% ------------------------------------------------------------
% 7. CONFIGURACOES GRAFICAS
% ------------------------------------------------------------

graphics_toolkit('gnuplot');

set(0, 'defaultfigurevisible', 'off');

set(0, 'defaultaxesfontsize', 11);

cores = [
    0.000 0.447 0.741;
    0.850 0.325 0.098;
    0.466 0.674 0.188
];


% ------------------------------------------------------------
% FIGURA 1 - FORMAS DE ONDA
% ------------------------------------------------------------

h = figure('position', [100 100 1200 900]);

te = EXEMPLO.t_evento;

t_ms = 1000*(EXEMPLO.t-te);

m = t_ms >= -13.08 & t_ms <= 20;

for f = 1:3

    subplot(3,1,f);

    hold on;

    plot(t_ms(m), EXEMPLO.original(m,f), ...
        'k', 'linewidth', 1.5);

    for k = 1:3

        campo = sprintf('snr%d', SNR_LEVELS(k));

        Y = EXEMPLO.(campo);

        plot(t_ms(m), Y(m,f), ...
            'color', cores(k,:), ...
            'linewidth', 0.6);

    end

    yl = ylim();

    plot([0 0], yl, 'k--');

    ylim(yl);

    grid on;

    ylabel('Corrente (A)');

    title(sprintf('Registro 01 - Fase %s', FASES{f}));

    if f == 1

        legend('Original', '50 dB', '40 dB', '30 dB', ...
            'location', 'northeast');

    end

    if f == 3

        xlabel('Tempo relativo ao evento (ms)');

    end

end

salvar_figura(h, fullfile(FIG_OUT, ...
    'G4_E01_01_Formas_de_Onda.png'));


% ------------------------------------------------------------
% FIGURA 2 - DISTRIBUICAO DO RUIDO
% ------------------------------------------------------------

h = figure('position', [100 100 1200 850]);

for k = 1:3

    subplot(3,1,k);

    campo = sprintf('ruido%d', SNR_LEVELS(k));

    R = EXEMPLO.(campo);

    ruido = R(:,2);

    [contagem, centros] = hist(ruido,60);

    largura = centros(2)-centros(1);

    densidade = contagem / ...
        (sum(contagem)*largura);

    bar(centros,densidade,1);

    hold on;

    mu = mean(ruido);

    sigma = std(ruido);

    xx = linspace(min(ruido),max(ruido),300);

    gauss = exp(-(xx-mu).^2/(2*sigma^2)) ...
        / (sigma*sqrt(2*pi));

    plot(xx,gauss,'r','linewidth',2);

    grid on;

    title(sprintf('Fase B - AWGN %d dB', ...
        SNR_LEVELS(k)));

    xlabel('Ruido (A)');

    ylabel('Densidade');

    legend('Empirica','Gaussiana', ...
        'location','northeast');

end

salvar_figura(h, fullfile(FIG_OUT, ...
    'G4_E01_02_Distribuicao_Ruido.png'));


% ------------------------------------------------------------
% FIGURA 3 - ERROS DE SNR
% ------------------------------------------------------------

h = figure('position', [100 100 1100 800]);

subplot(2,1,1);

plot([25 55],[25 55],'k--','linewidth',1.5);

hold on;

for f = 1:3

    m = AUDIT(:,3) == f;

    scatter(AUDIT(m,2),AUDIT(m,4), ...
        35,cores(f,:),'filled');

end

grid on;

xlabel('SNR nominal (dB)');

ylabel('SNR efetivo pre-evento (dB)');

title('SNR nominal versus efetivo');

legend('Identidade','Fase A','Fase B','Fase C', ...
    'location','northwest');

subplot(2,1,2);

hold on;

for f = 1:3

    m = AUDIT(:,3) == f;

    erro = AUDIT(m,4)-AUDIT(m,2);

    scatter(AUDIT(m,2)+(f-2)*0.3,erro, ...
        35,cores(f,:),'filled');

end

plot([25 55],[0 0],'k--');

grid on;

xlabel('SNR nominal (dB)');

ylabel('Erro de SNR (dB)');

title('Desvio em relacao ao valor nominal');

legend('Fase A','Fase B','Fase C', ...
    'location','best');

salvar_figura(h, fullfile(FIG_OUT, ...
    'G4_E01_03_SNR_Nominal_Efetivo.png'));


% ------------------------------------------------------------
% FIGURA 4 - PRESERVACAO DO RMS
% ------------------------------------------------------------

h = figure('position', [100 100 1100 650]);

ERROS = zeros(3,2);

for k = 1:3

    m = AUDIT(:,2) == SNR_LEVELS(k);

    ERROS(k,1) = mean(abs(AUDIT(m,6)));

    ERROS(k,2) = mean(abs(AUDIT(m,7)));

end

bar(ERROS);

set(gca,'xtick',1:3);

set(gca,'xticklabel',{'50 dB','40 dB','30 dB'});

grid on;

xlabel('SNR nominal');

ylabel('Erro absoluto medio RMS (%)');

title('Preservacao das amplitudes RMS');

legend('Pre-evento','Pos-evento', ...
    'location','northwest');

salvar_figura(h, fullfile(FIG_OUT, ...
    'G4_E01_04_Preservacao_RMS.png'));


% ------------------------------------------------------------
% FIGURA 5 - AUTOCORRELACAO DO RUIDO
% ------------------------------------------------------------

h = figure('position', [100 100 1100 850]);

MAXLAG = 100;

for k = 1:3

    subplot(3,1,k);

    campo = sprintf('ruido%d', SNR_LEVELS(k));

    R = EXEMPLO.(campo);

    ruido = R(:,2);

    acf = calcular_acf(ruido,MAXLAG);

    N_ruido = numel(ruido);

    limite = 1.96/sqrt(N_ruido);

    % O atraso zero e omitido para facilitar a
    % visualizacao dos atrasos nao nulos.

    stem(1:MAXLAG,acf(2:end),'filled', ...
        'markersize',2);

    hold on;

    plot([1 MAXLAG],[limite limite],'r--');

    plot([1 MAXLAG],[-limite -limite],'r--');

    plot([1 MAXLAG],[0 0],'k:');

    grid on;

    xlabel('Atraso (amostras)');

    ylabel('Autocorrelacao');

    title(sprintf('AWGN %d dB - Fase B', ...
        SNR_LEVELS(k)));

    legend('ACF','Limites pontuais de 95%', ...
        'location','northeast');

end

salvar_figura(h, fullfile(FIG_OUT, ...
    'G4_E01_05_Autocorrelacao_Ruido.png'));


% ------------------------------------------------------------
% 8. PROTOCOLO FINAL
% ------------------------------------------------------------

fid = fopen(fullfile(OUT, ...
    'G4_E01_Protocolo.txt'),'w');

assert(fid >= 0);

fprintf(fid,'GATE 4 - EXPERIMENTO 01\n');

fprintf(fid,'Versao 2\n\n');

fprintf(fid,'Versao GNU Octave: %s\n', version());

fprintf(fid,'Data: %s\n\n', datestr(now()));

fprintf(fid,'OBJETIVO\n');

fprintf(fid,'Expansao estatistica por AWGN.\n\n');

fprintf(fid,'METODO\n');

fprintf(fid,'Niveis de SNR: 50, 40 e 30 dB.\n');

fprintf(fid, ...
    'Referencia: potencia pre-evento por fase.\n');

fprintf(fid, ...
    'Intensidade constante durante todo o registro.\n');

fprintf(fid, ...
    'Realizacoes independentes entre fases.\n');

fprintf(fid, ...
    'Sementes deterministicas por variante e fase.\n');

fprintf(fid, ...
    'Semente base: %d.\n',SEED_BASE);

fprintf(fid, ...
    'Tolerancia de SNR: %.3f dB.\n\n',TOL_SNR);

fprintf(fid,'PARTICIONAMENTO\n');

fprintf(fid,'Treino: ');

fprintf(fid,'%02d ',REG_TREINO);

fprintf(fid,'\nValidacao: ');

fprintf(fid,'%02d ',REG_VALID);

fprintf(fid,'\nTeste reservado: ');

fprintf(fid,'%02d ',REG_TESTE);

fprintf(fid,'\n\n');

fprintf(fid,'RESULTADOS\n');

fprintf(fid,'Registros processados: %d\n', ...
    numel(REGISTROS));

fprintf(fid,'Variantes geradas: %d\n',N_GERADOS);

fprintf(fid,'Realizacoes de ruido: %d\n',size(AUDIT,1));

fprintf(fid,'Realizacoes aprovadas: %d\n',sum(AUDIT(:,12)));

fprintf(fid,'Figuras produzidas: 5\n\n');

fprintf(fid,'RESTRICOES CIENTIFICAS\n');

fprintf(fid,'Arquivos originais preservados.\n');

fprintf(fid,'Somente o canal de correntes modificado.\n');

fprintf(fid,'Marcadores temporais preservados.\n');

fprintf(fid,'Registros 06 e 12 nao carregados.\n');

fprintf(fid,'Seis variantes de teste adiadas.\n');

fprintf(fid, ...
    'Variantes nao constituem simulacoes independentes.\n');

fprintf(fid, ...
    'Nenhum classificador treinado neste experimento.\n');

fprintf(fid, ...
    'SNR global pode diferir do SNR pre-evento.\n');

fclose(fid);


% ------------------------------------------------------------
% 9. ENCERRAMENTO
% ------------------------------------------------------------

set(0,'defaultfigurevisible','on');

fprintf('\n');
fprintf('==============================================\n');
fprintf(' G4/E01 - EXECUCAO CONCLUIDA\n');
fprintf('==============================================\n');

fprintf('Registros processados: %d\n',numel(REGISTROS));

fprintf('Arquivos AWGN gerados: %d\n',N_GERADOS);

fprintf('Realizacoes aprovadas: %d/%d\n', ...
    sum(AUDIT(:,12)),size(AUDIT,1));

fprintf('Figuras exportadas: 5\n');

fprintf('Registros 06 e 12: PRESERVADOS\n');

fprintf('\nDiretorio dos resultados:\n%s\n',OUT);

fprintf('\nArquivos produzidos:\n');

fprintf('  G4_E01_Manifesto.csv\n');

fprintf('  G4_E01_Metricas.csv\n');

fprintf('  G4_E01_Resumo.csv\n');

fprintf('  G4_E01_Protocolo.txt\n');

fprintf('  Dados_AWGN/\n');

fprintf('  Figuras/\n');

fprintf('\nAguardando interpretacao cientifica.\n');

fprintf('==============================================\n');
