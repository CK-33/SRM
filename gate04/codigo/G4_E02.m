
1;

% ============================================================
% PESQUISA SRM
% GATE 4 - EXPERIMENTO 02.1
% Auditoria e regularizacao temporal
% GNU Octave 11.3.0
% ============================================================

clear;
clc;
close all;

pkg load signal;

% ============================================================
% 1. FUNCOES AUXILIARES
% ============================================================

function v = rms_local(x)
    v = sqrt(mean(x.^2));
end

function v = erro_pct(a, b)
    v = 100 * abs(a-b) / max(abs(a), eps);
end

function P = psd_bandas(x, Fs)

    L = 2048;

    x = x(:);

    [pxx, f] = pwelch( ...
        x, hamming(L), 0.5, ...
        4096, Fs);

    limites = [0 5000;
               5000 20000;
               20000 50000;
               50000 100000];

    P = zeros(1,4);

    df = f(2)-f(1);

    for b = 1:4

        if b < 4
            idx = f >= limites(b,1) & ...
                  f < limites(b,2);
        else
            idx = f >= limites(b,1) & ...
                  f <= limites(b,2);
        end

        P(b) = sum(pxx(idx))*df;

    end

end

function exportar(fig, pasta, nome)

    arquivo = fullfile(pasta, nome);

    set(fig, 'paperpositionmode', 'auto');

    print(fig, arquivo, '-dpng', '-r150');

    close(fig);

    fprintf('Figura exportada: %s\n', nome);

end

% ============================================================
% 2. CONFIGURACAO
% ============================================================

ROOT = [
    'PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
    'TesteNo1_31OCT2026/Data and Scripts'
];

DATA_ROOT = fullfile(ROOT, 'Results_1_to_12');

REG_TREINO = [1 2 4 5 7 8 10 11];
REG_VALID  = [3 9];

REGISTROS = sort([REG_TREINO REG_VALID]);

FS = 200000;
TS = 1/FS;

% Tolerancia para coincidencia com grade nominal.
% 0.001 do passo temporal = 5 ns.

TOL_GRADE = 0.001 * TS;

FASES = {'A', 'B', 'C'};

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

assert(exist(DATA_ROOT, 'dir') == 7);

assert(~any(ismember(REGISTROS, [6 12])));

% Evitar sobrescrita de execucoes anteriores.

run_id = 1;

while true

    OUT = fullfile(ROOT, sprintf( ...
        'Gate4_Exp02_1_Run%02d', run_id));

    if exist(OUT, 'dir') ~= 7
        break;
    end

    run_id = run_id + 1;

end

mkdir(OUT);

FIG = fullfile(OUT, 'Figuras');

mkdir(FIG);

fprintf('\n');
fprintf('==============================================\n');
fprintf(' PESQUISA SRM\n');
fprintf(' GATE 4 - EXPERIMENTO 02.1\n');
fprintf(' Auditoria e regularizacao temporal\n');
fprintf('==============================================\n');

fprintf('Fs nominal: %.0f Hz\n', FS);
fprintf('Passo nominal: %.6f us\n', TS*1e6);
fprintf('Tolerancia da grade: %.3f ns\n', ...
        TOL_GRADE*1e9);

fprintf('Registros autorizados: %d\n', ...
        numel(REGISTROS));

fprintf('Registros 06 e 12: RESERVADOS\n');
fprintf('Saida: %s\n\n', OUT);

% ============================================================
% 3. ABERTURA DOS CSV
% ============================================================

fid_grade = fopen(fullfile(OUT, ...
    'G4_E02_1_Grade.csv'), 'w');

fid_impacto = fopen(fullfile(OUT, ...
    'G4_E02_1_Impacto.csv'), 'w');

fid_psd = fopen(fullfile(OUT, ...
    'G4_E02_1_PSD.csv'), 'w');

fid_eventos = fopen(fullfile(OUT, ...
    'G4_E02_1_Eventos_Agregados.csv'), 'w');

assert(all([fid_grade fid_impacto ...
            fid_psd fid_eventos] >= 0));

fprintf(fid_grade, ...
    ['registro,grupo,n_original,n_grade,' ...
     'n_fora_grade,n_ausentes,n_multiplos,' ...
     'n_excedentes,n_colisoes_extras,' ...
     'cobertura_pct,erro_max_grade_ns,' ...
     'erro_max_grade_us,' ...
     'status\n']);

fprintf(fid_impacto, ...
    ['registro,grupo,fase,regiao,' ...
     'rms_original,rms_primeira,rms_ultima,' ...
     'erro_rms_primeira_pct,' ...
     'erro_rms_ultima_pct,' ...
     'pico_original,pico_primeira,pico_ultima,' ...
     'erro_pico_primeira_pct,' ...
     'erro_pico_ultima_pct,' ...
     'rmse_primeira_ultima_A,' ...
     'dif_max_primeira_ultima_A,' ...
     'rms_derivada_original_passo_fixo,' ...
     'rms_derivada_primeira,' ...
     'rms_derivada_ultima\n']);

fprintf(fid_psd, ...
    ['registro,grupo,fase,regiao,banda,' ...
     'pot_original_indexada,' ...
     'pot_primeira,pot_ultima,' ...
     'erro_primeira_pct,erro_ultima_pct,' ...
     'dif_primeira_ultima_dB\n']);

fprintf(fid_eventos, ...
    ['registro,grupo,n_colisoes_pre,' ...
     'n_colisoes_pos,n_colisoes_evento,' ...
     'dif_max_A,dif_max_B,dif_max_C\n']);

% Armazenamento exclusivo de resultados agregados.

GRADE = [];

IMPACTO = [];

ESPECTRAL = [];

EVENTOS = [];

DEMO = struct();

% ============================================================
% 4. PROCESSAMENTO DOS REGISTROS
% ============================================================

for r = REGISTROS

    if any(REG_TREINO == r)
        grupo = 'treino';
    else
        grupo = 'validacao';
    end

    fprintf('\n----------------------------------------------\n');
    fprintf('Registro %02d | %s\n', r, grupo);
    fprintf('----------------------------------------------\n');

    arquivo = fullfile(DATA_ROOT, ...
        sprintf('results_%d.mat', r));

    assert(exist(arquivo, 'file') == 2);

    S = load(arquivo, 'results');

    t = double(S.results.time(:));

    I = double(S.results.signals(3).values);

    E = double(S.results.signals(6).values);

    N = numel(t);

    if size(I,1) ~= N && size(I,2) == N
        I = I.';
    end

    assert(size(I,1) == N);
    assert(size(I,2) == 3);

    E = E(:);

    assert(numel(E) == N);

    assert(all(isfinite(t)));
    assert(all(isfinite(I(:))));
    assert(all(isfinite(E)));

    assert(all(diff(t) > 0));

    te = T_EVENTOS(r);

    % --------------------------------------------------------
    % 4.1. Grade temporal nominal
    % --------------------------------------------------------

    Nu = round((t(end)-t(1))/TS)+1;

    tu = t(1)+(0:Nu-1)'*TS;

    % Indice nominal mais proximo.

    k = round((t-t(1))/TS)+1;

    dentro = k >= 1 & k <= Nu;

    erro_grade = inf(N,1);

    erro_grade(dentro) = abs( ...
        t(dentro)-tu(k(dentro)));

    candidato = dentro & ...
                erro_grade <= TOL_GRADE;

    idx = find(candidato);

    % Quantidade de amostras candidatas
    % em cada ponto nominal.

    contagem = accumarray( ...
        k(idx), 1, [Nu 1], @sum, 0);

    ausentes = find(contagem == 0);

    multiplos = find(contagem > 1);

    n_fora = sum(~candidato);

    n_ausentes = numel(ausentes);

    n_multiplos = numel(multiplos);

    n_colisoes_extras = sum( ...
        max(contagem-1,0));

    cobertura = 100*sum(contagem>0)/Nu;

    max_erro_grade = max(erro_grade(idx));

    % Balanco de amostras.

    assert(N == ...
        n_fora + sum(contagem));

    assert(N-Nu == ...
        n_fora+n_colisoes_extras-n_ausentes);

    fprintf('Amostras originais: %d\n', N);
    fprintf('Amostras na grade: %d\n', Nu);

    fprintf('Fora da grade: %d\n', n_fora);

    fprintf('Pontos nominais ausentes: %d\n', ...
        n_ausentes);

    fprintf('Pontos com multiplas amostras: %d\n', ...
        n_multiplos);

    fprintf('Amostras extras por colisao: %d\n', ...
        n_colisoes_extras);

    fprintf('Cobertura da grade: %.9f %%\n', ...
        cobertura);

    fprintf('Erro maximo de coincidencia: %.9f ns\n', ...
        max_erro_grade*1e9);

    % --------------------------------------------------------
    % 4.2. Primeira e ultima amostra
    % --------------------------------------------------------

    primeiro = accumarray( ...
        k(idx), idx, [Nu 1], @min, 0);

    ultimo = accumarray( ...
        k(idx), idx, [Nu 1], @max, 0);

    status = 'GRADE_COMPLETA';

    if n_ausentes > 0

        status = 'GRADE_INCOMPLETA';

    end

    fprintf(fid_grade, ...
        ['%d,%s,%d,%d,%d,%d,%d,%d,%d,' ...
         '%.12g,%.12g,%.12g,%s\n'], ...
        r, grupo, N, Nu, ...
        n_fora, n_ausentes, n_multiplos, ...
        N-Nu, n_colisoes_extras, ...
        cobertura, max_erro_grade*1e9, ...
        max_erro_grade*1e6, status);

    GRADE = [
        GRADE;
        r N Nu n_fora n_ausentes ...
        n_multiplos N-Nu cobertura
    ];

    % Nao interpolar automaticamente.
    % Se faltarem amostras, registrar e continuar.

    if n_ausentes > 0

        fprintf(['AVISO: grade incompleta. ' ...
                 'Interpolacao nao autorizada.\n']);

        clear S t I E;

        continue;

    end

    I1 = I(primeiro,:);
    I2 = I(ultimo,:);

    E1 = E(primeiro);
    E2 = E(ultimo);

    assert(size(I1,1) == Nu);
    assert(size(I2,1) == Nu);

    assert(all(isfinite(I1(:))));
    assert(all(isfinite(I2(:))));

    assert(max(abs(diff(tu)-TS)) < 1e-10);

    D = I2-I1;

    assert(all(all(D(contagem==1,:) == 0)));

    % --------------------------------------------------------
    % 4.3. Auditoria dos eventos discretos
    % --------------------------------------------------------

    % Usar a grade nominal para rotulos temporais.
    % Tolerancia apenas contra arredondamento numerico.

    tol_evento = 1e-10;

    pre_u = tu < te-tol_evento;

    pos_u = ~pre_u;

    % Mesma convencao no sinal original.

    pre_o = t < te-tol_evento;

    pos_o = ~pre_o;

    col_pre = sum(tu(multiplos) < ...
                  te-tol_evento);

    col_pos = n_multiplos-col_pre;

    col_evento = sum(abs( ...
        tu(multiplos)-te) <= TOL_GRADE);

    dif_fases = max(abs(D),[],1);

    fprintf('Colisoes pre-evento: %d\n', col_pre);

    fprintf('Colisoes pos-evento: %d\n', col_pos);

    fprintf('Colisoes no evento principal: %d\n', ...
        col_evento);

    fprintf(['Diferenca maxima entre convencoes ' ...
             'A/B/C: %.9g / %.9g / %.9g A\n'], ...
        dif_fases);

    fprintf('Marcadores divergentes: %d\n', ...
        sum(E1 ~= E2));

    fprintf(fid_eventos, ...
        '%d,%s,%d,%d,%d,%.12g,%.12g,%.12g\n', ...
        r, grupo, col_pre, col_pos, col_evento, ...
        dif_fases);

    EVENTOS = [
        EVENTOS;
        r col_pre col_pos col_evento dif_fases
    ];

    % --------------------------------------------------------
    % 4.4. Separacao temporal
    % --------------------------------------------------------

    % Regioes:
    % 1 - pre-evento
    % 2 - primeiros 200 ms apos evento
    % 3 - pos-evento tardio
    % 4 - registro completo

    MASK_O = {
        pre_o;
        pos_o & t < te+0.200;
        t >= te+0.200;
        true(N,1)
    };

    MASK_U = {
        pre_u;
        pos_u & tu < te+0.200;
        tu >= te+0.200;
        true(Nu,1)
    };

    REGIOES = {
        'pre';
        'transitorio_200ms';
        'tardio';
        'global'
    };

    % --------------------------------------------------------
    % 4.5. Impacto nas caracteristicas
    % --------------------------------------------------------

    for p = 1:3

        for q = 1:4

            xo = I(MASK_O{q},p);

            x1 = I1(MASK_U{q},p);

            x2 = I2(MASK_U{q},p);

            Ro = rms_local(xo);

            R1 = rms_local(x1);

            R2 = rms_local(x2);

            eR1 = erro_pct(Ro,R1);

            eR2 = erro_pct(Ro,R2);

            Po = max(abs(xo));

            P1 = max(abs(x1));

            P2 = max(abs(x2));

            eP1 = erro_pct(Po,P1);

            eP2 = erro_pct(Po,P2);

            rmse12 = rms_local(x1-x2);

            dif12 = max(abs(x1-x2));

            % Derivada original por passo nominal.
            % Nao dividir pelos dt quase nulos.

            Do = rms_local(diff(xo)/TS);

            D1 = rms_local(diff(x1)/TS);

            D2 = rms_local(diff(x2)/TS);

            fprintf(fid_impacto, ...
                ['%d,%s,%s,%s,' ...
                 '%.12g,%.12g,%.12g,' ...
                 '%.12g,%.12g,' ...
                 '%.12g,%.12g,%.12g,' ...
                 '%.12g,%.12g,' ...
                 '%.12g,%.12g,' ...
                 '%.12g,%.12g,%.12g\n'], ...
                r,grupo,FASES{p},REGIOES{q}, ...
                Ro,R1,R2,eR1,eR2, ...
                Po,P1,P2,eP1,eP2, ...
                rmse12,dif12,Do,D1,D2);

            IMPACTO = [
                IMPACTO;
                r p q eR1 eR2 eP1 eP2 ...
                rmse12 dif12 ...
                erro_pct(Do,D1) ...
                erro_pct(Do,D2)
            ];

            % ------------------------------------------------
            % 4.6. Impacto espectral
            % ------------------------------------------------
            %
            % A PSD original indexada assume Fs nominal.
            % Ela NAO e uma PSD de referencia absoluta
            % para amostragem nao uniforme.

            Bo = psd_bandas(xo,FS);

            B1 = psd_bandas(x1,FS);

            B2 = psd_bandas(x2,FS);

            for b = 1:4

                eb1 = erro_pct(Bo(b),B1(b));

                eb2 = erro_pct(Bo(b),B2(b));

                db12 = 10*log10( ...
                    max(B2(b),realmin)/ ...
                    max(B1(b),realmin));

                fprintf(fid_psd, ...
                    ['%d,%s,%s,%s,%d,' ...
                     '%.12g,%.12g,%.12g,' ...
                     '%.12g,%.12g,%.12g\n'], ...
                    r,grupo,FASES{p},REGIOES{q},b, ...
                    Bo(b),B1(b),B2(b), ...
                    eb1,eb2,db12);

                ESPECTRAL = [
                    ESPECTRAL;
                    r p q b eb1 eb2 db12
                ];

            end

        end

    end

    % --------------------------------------------------------
    % 4.7. Exemplo para figuras
    % --------------------------------------------------------

    if r == 1

        % Dados permanecem localmente no Octave.

        DEMO.t = t;
        DEMO.tu = tu;

        DEMO.I = I;
        DEMO.I1 = I1;
        DEMO.I2 = I2;

        DEMO.te = te;

        DEMO.candidato = candidato;

        DEMO.multiplos = multiplos;

    end

    fprintf('Auditoria do registro concluida.\n');

    clear S I I1 I2 E E1 E2 D;

end

fclose(fid_grade);
fclose(fid_impacto);
fclose(fid_psd);
fclose(fid_eventos);

% ============================================================
% 5. RESUMO GLOBAL
% ============================================================

fprintf('\n==============================================\n');
fprintf(' AUDITORIA GLOBAL\n');
fprintf('==============================================\n');

fprintf('Registros processados: %d\n', ...
    size(GRADE,1));

fprintf('Registros com grade completa: %d/%d\n', ...
    sum(GRADE(:,5)==0), size(GRADE,1));

fprintf('Amostras originais: %d\n', ...
    sum(GRADE(:,2)));

fprintf('Amostras nominais esperadas: %d\n', ...
    sum(GRADE(:,3)));

fprintf('Total fora da grade: %d\n', ...
    sum(GRADE(:,4)));

fprintf('Total de pontos ausentes: %d\n', ...
    sum(GRADE(:,5)));

fprintf('Total de pontos multiplos: %d\n', ...
    sum(GRADE(:,6)));

fprintf('Total de amostras excedentes: %d\n', ...
    sum(GRADE(:,7)));

fprintf('Cobertura minima: %.9f %%\n', ...
    min(GRADE(:,8)));

if ~isempty(IMPACTO)

    global_idx = IMPACTO(:,3)==4;

    G = IMPACTO(global_idx,:);

    fprintf('\nIMPACTO NO RMS GLOBAL\n');

    fprintf('Erro medio - primeira: %.9g %%\n', ...
        mean(G(:,4)));

    fprintf('Erro medio - ultima: %.9g %%\n', ...
        mean(G(:,5)));

    fprintf('Erro maximo - primeira: %.9g %%\n', ...
        max(G(:,4)));

    fprintf('Erro maximo - ultima: %.9g %%\n', ...
        max(G(:,5)));

    fprintf('\nIMPACTO NO PICO GLOBAL\n');

    fprintf('Erro maximo - primeira: %.9g %%\n', ...
        max(G(:,6)));

    fprintf('Erro maximo - ultima: %.9g %%\n', ...
        max(G(:,7)));

    fprintf('\nCOMPARACAO ENTRE CONVENCOES\n');

    fprintf('RMSE medio: %.12g A\n', ...
        mean(G(:,8)));

    fprintf('Diferenca absoluta maxima: %.12g A\n', ...
        max(G(:,9)));

    fprintf('\nIMPACTO NAS DERIVADAS\n');

    fprintf('Erro medio - primeira: %.9g %%\n', ...
        mean(G(:,10)));

    fprintf('Erro medio - ultima: %.9g %%\n', ...
        mean(G(:,11)));

    fprintf('Erro maximo - primeira: %.9g %%\n', ...
        max(G(:,10)));

    fprintf('Erro maximo - ultima: %.9g %%\n', ...
        max(G(:,11)));

    fprintf('\nIMPACTO ESPECTRAL\n');

    BANDAS = {
        '0-5 kHz';
        '5-20 kHz';
        '20-50 kHz';
        '50-100 kHz'
    };

    for b = 1:4

        z = ESPECTRAL( ...
            ESPECTRAL(:,4)==b & ...
            ESPECTRAL(:,3)==4,:);

        fprintf('\nBanda %s\n', BANDAS{b});

        fprintf('Erro medio primeira: %.9g %%\n', ...
            mean(z(:,5)));

        fprintf('Erro medio ultima: %.9g %%\n', ...
            mean(z(:,6)));

        fprintf('Diferenca max entre convencoes: %.9g dB\n', ...
            max(abs(z(:,7))));

    end

end

% ============================================================
% 6. FIGURAS
% ============================================================

graphics_toolkit('gnuplot');

set(0,'defaultfigurevisible','off');

set(0,'defaultaxesfontsize',12);
set(0,'defaulttextfontsize',12);

% ------------------------------------------------------------
% Figura 01 - Contagem de irregularidades
% ------------------------------------------------------------

fig = figure('position',[100 100 1100 650]);

bar(GRADE(:,1), ...
    GRADE(:,[4 5 6 7]),'grouped');

grid on;

xlabel('Registro');

ylabel('Quantidade');

title('Auditoria da grade temporal');

legend({'Fora da grade', ...
        'Ausentes', ...
        'Pontos multiplos', ...
        'Amostras excedentes'}, ...
        'location','northwest');

set(gca,'xtick',REGISTROS);

exportar(fig,FIG, ...
    'G4_E02_1_01_Auditoria_Grade.png');

if ~isempty(IMPACTO)

    % --------------------------------------------------------
    % Figura 02 - Erro relativo no RMS
    % --------------------------------------------------------

    fig = figure('position',[100 100 1100 650]);

    M = zeros(3,2);

    for p = 1:3

        z = IMPACTO( ...
            IMPACTO(:,2)==p & ...
            IMPACTO(:,3)==4,:);

        M(p,:) = mean(z(:,4:5),1);

    end

    bar(M,'grouped');

    grid on;

    set(gca,'xtick',1:3, ...
        'xticklabel',FASES);

    ylabel('Erro absoluto medio RMS (%)');

    xlabel('Fase');

    title('Impacto da regularizacao no RMS');

    legend({'Primeira amostra', ...
            'Ultima amostra'}, ...
            'location','northwest');

    exportar(fig,FIG, ...
        'G4_E02_1_02_Impacto_RMS.png');

    % --------------------------------------------------------
    % Figura 03 - Impacto nas bandas espectrais
    % --------------------------------------------------------

    fig = figure('position',[100 100 1100 650]);

    M = zeros(4,2);

    for b = 1:4

        z = ESPECTRAL( ...
            ESPECTRAL(:,4)==b & ...
            ESPECTRAL(:,3)==4,:);

        M(b,:) = mean(z(:,5:6),1);

    end

    bar(M,'grouped');

    grid on;

    set(gca,'xtick',1:4, ...
        'xticklabel',BANDAS);

    ylabel('Diferenca absoluta media (%)');

    title('Impacto espectral da regularizacao');

    legend({'Primeira amostra', ...
            'Ultima amostra'}, ...
            'location','northwest');

    exportar(fig,FIG, ...
        'G4_E02_1_03_Impacto_Espectral.png');

    % --------------------------------------------------------
    % Figura 04 - Impacto nas derivadas
    % --------------------------------------------------------

    fig = figure('position',[100 100 1100 650]);

    M = zeros(3,2);

    for p = 1:3

        z = IMPACTO( ...
            IMPACTO(:,2)==p & ...
            IMPACTO(:,3)==4,:);

        M(p,:) = mean(z(:,10:11),1);

    end

    bar(M,'grouped');

    grid on;

    set(gca,'xtick',1:3, ...
        'xticklabel',FASES);

    ylabel('Diferenca absoluta media (%)');

    title('Impacto no RMS da derivada discreta');

    legend({'Primeira amostra', ...
            'Ultima amostra'}, ...
            'location','northwest');

    exportar(fig,FIG, ...
        'G4_E02_1_04_Impacto_Derivadas.png');

end

if isfield(DEMO,'t')

    % --------------------------------------------------------
    % Figura 05 - Proximidade do evento principal
    % --------------------------------------------------------

    fig = figure('position',[100 100 1200 850]);

    mo = abs(DEMO.t-DEMO.te) <= 100e-6;

    mu = abs(DEMO.tu-DEMO.te) <= 100e-6;

    for p = 1:3

        subplot(3,1,p);

        plot((DEMO.t(mo)-DEMO.te)*1e6, ...
             DEMO.I(mo,p),'ko-', ...
             'markersize',3);

        hold on;

        plot((DEMO.tu(mu)-DEMO.te)*1e6, ...
             DEMO.I1(mu,p),'b.-');

        plot((DEMO.tu(mu)-DEMO.te)*1e6, ...
             DEMO.I2(mu,p),'r--');

        grid on;

        ylabel('Corrente (A)');

        title(sprintf('Registro 01 - Fase %s', ...
            FASES{p}));

        if p == 1

            legend({'Original','Primeira','Ultima'}, ...
                'location','northwest');

        end

        if p == 3

            xlabel('Tempo relativo ao evento (us)');

        end

    end

    exportar(fig,FIG, ...
        'G4_E02_1_05_Detalhe_Evento.png');

    % --------------------------------------------------------
    % Figura 06 - Distribuicao dos passos temporais
    % --------------------------------------------------------

    fig = figure('position',[100 100 1100 650]);

    dt = diff(DEMO.t);

    desvio = abs(dt-TS)/TS;

    irregular = desvio>0.001;

    semilogy(find(irregular), ...
        dt(irregular)*1e6,'ro', ...
        'markersize',5);

    hold on;

    plot([1 numel(dt)],[TS TS]*1e6, ...
        'k--','linewidth',1.5);

    grid on;

    xlabel('Indice da amostra');

    ylabel('Intervalo temporal (us)');

    title('Registro 01 - Intervalos excepcionais');

    legend({'Intervalos irregulares', ...
            'Passo nominal'}, ...
            'location','northeast');

    exportar(fig,FIG, ...
        'G4_E02_1_06_Intervalos_Irregulares.png');

    % --------------------------------------------------------
    % Figura 07 - Comparacao das PSDs
    % --------------------------------------------------------

    fig = figure('position',[100 100 1200 850]);

    for p = 1:3

        subplot(3,1,p);

        [P0,f] = pwelch( ...
            DEMO.I(:,p), hamming(2048), ...
            0.5,4096,FS);

        [P1,~] = pwelch( ...
            DEMO.I1(:,p), hamming(2048), ...
            0.5,4096,FS);

        [P2,~] = pwelch( ...
            DEMO.I2(:,p), hamming(2048), ...
            0.5,4096,FS);

        plot(f/1000, ...
            10*log10(max(P0,realmin)), ...
            'k','linewidth',1);

        hold on;

        plot(f/1000, ...
            10*log10(max(P1,realmin)), ...
            'b');

        plot(f/1000, ...
            10*log10(max(P2,realmin)), ...
            'r--');

        grid on;

        ylabel('PSD (dB A^2/Hz)');

        title(sprintf('Registro 01 - Fase %s', ...
            FASES{p}));

        if p == 1

            legend({'Original indexado', ...
                    'Primeira','Ultima'}, ...
                    'location','northeast');

        end

        if p == 3

            xlabel('Frequencia (kHz)');

        end

        xlim([0 100]);

    end

    exportar(fig,FIG, ...
        'G4_E02_1_07_Comparacao_PSD.png');

end

% ============================================================
% 7. PROTOCOLO
% ============================================================

fid = fopen(fullfile(OUT, ...
    'G4_E02_1_Protocolo.txt'),'w');

assert(fid >= 0);

fprintf(fid,'GATE 4 - EXPERIMENTO 02.1\n\n');

fprintf(fid,'Fs nominal: %.0f Hz\n',FS);

fprintf(fid,'Passo nominal: %.12g s\n',TS);

fprintf(fid,'Tolerancia da grade: %.12g s\n', ...
    TOL_GRADE);

fprintf(fid,'Registros: ');

fprintf(fid,'%02d ',REGISTROS);

fprintf(fid,'\n');

fprintf(fid,'Registros 06 e 12 reservados.\n');

fprintf(fid,['Selecao trifasica conjunta ' ...
             'por grade nominal.\n']);

fprintf(fid,['Comparacao entre primeira e ' ...
             'ultima amostra coincidente.\n']);

fprintf(fid,['Nenhuma interpolacao automatica ' ...
             'foi aplicada.\n']);

fprintf(fid,['Nenhum arquivo MAT derivado ' ...
             'foi salvo.\n']);

fprintf(fid,['A PSD original indexada constitui ' ...
             'referencia comparativa, nao uma ' ...
             'verdade espectral absoluta.\n']);

fprintf(fid,['Derivadas originais calculadas ' ...
             'com passo nominal fixo.\n']);

fprintf(fid,['Regiao transitoria definida ' ...
             'operacionalmente como 200 ms.\n']);

fprintf(fid,['Resultados de validacao nao devem ' ...
             'determinar os parametros da ' ...
             'expansao futura.\n']);

fprintf(fid,['Nenhum arquivo original ' ...
             'foi modificado.\n']);

fclose(fid);

clear DEMO;

set(0,'defaultfigurevisible','on');

% ============================================================
% 8. ENCERRAMENTO
% ============================================================

fprintf('\n==============================================\n');
fprintf(' G4/E02.1 - EXECUCAO CONCLUIDA\n');
fprintf('==============================================\n');

fprintf('Registros autorizados: %d\n', ...
    numel(REGISTROS));

fprintf('Registros auditados: %d\n', ...
    size(GRADE,1));

fprintf('Grade completa: %d/%d\n', ...
    sum(GRADE(:,5)==0),size(GRADE,1));

fprintf('Registros 06 e 12: PRESERVADOS\n');

fprintf('Arquivos originais: PRESERVADOS\n');

fprintf('Interpolacao automatica: NAO\n');

fprintf('Novos arquivos MAT: NAO\n');

fprintf('\nDiretorio dos resultados:\n%s\n',OUT);

fprintf('\nAguardando interpretacao cientifica.\n');

fprintf('==============================================\n');
