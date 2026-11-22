
% ============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 03
% VERSAO CORRIGIDA
%
% Detectores determinísticos:
% D1 - RMS por fase
% D2 - Desequilibrio trifasico
% D3 - Fator de crista por fase
%
% GNU Octave 11.3.0
% Sem pacotes adicionais
%
% Entradas:
%   Gate2_Exp05/G2_E05_Caracteristicas.csv
%   Gate3_Exp02/G3_E02_Manifesto.csv
%
% Saidas:
%   Gate3_Exp03/G3_E03_Metricas.csv
%   Gate3_Exp03/G3_E03_Limiares.csv
%   Gate3_Exp03/G3_E03_Decisoes.csv
%   Gate3_Exp03/G3_E03_Fig01_F1.png
%   Gate3_Exp03/G3_E03_Fig02_FPR.png
%
% IMPORTANTE:
%   O teste nao sera avaliado nesta etapa.
% ============================================================

clear;
clc;
close all;

% ============================================================
% 0. CONFIGURACAO
% ============================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

entrada = fullfile(base, ...
    'Gate2_Exp05', ...
    'G2_E05_Caracteristicas.csv');

manifesto = fullfile(base, ...
    'Gate3_Exp02', ...
    'G3_E02_Manifesto.csv');

saida = fullfile(base, 'Gate3_Exp03');

if ~exist(saida, 'dir')
    mkdir(saida);
end

if ~exist(entrada, 'file')
    error('CSV de caracteristicas nao encontrado.');
end

if ~exist(manifesto, 'file')
    error('Manifesto do G3/E02 nao encontrado.');
end

fprintf('\n========================================\n');
fprintf('GATE 3 - EXPERIMENTO 03\n');
fprintf('DETECTORES DETERMINISTICOS\n');
fprintf('========================================\n');

% ============================================================
% 1. LEITURA DAS CARACTERISTICAS
% ============================================================

fid = fopen(entrada, 'rt');

if fid < 0
    error('Nao foi possivel abrir o CSV de entrada.');
end

cab = fgetl(fid);

if ~ischar(cab)
    fclose(fid);
    error('CSV de caracteristicas vazio.');
end

cab = strrep(cab, char([239 187 191]), '');

nomes = strtrim(strsplit(cab, ','));

fmt = repmat('%s', 1, numel(nomes));

C = textscan(fid, fmt, ...
    'Delimiter', ',', ...
    'ReturnOnError', false);

fclose(fid);

N = numel(C{1});

for j = 1:numel(C)

    if numel(C{j}) ~= N
        error('Colunas inconsistentes no CSV.');
    end

end

fprintf('\nLinhas de caracteristicas: %d\n', N);

% ============================================================
% 2. LEITURA DO MANIFESTO G3/E02
% ============================================================

fid = fopen(manifesto, 'rt');

if fid < 0
    error('Nao foi possivel abrir o manifesto.');
end

cab2 = fgetl(fid);

if ~ischar(cab2)
    fclose(fid);
    error('Manifesto vazio.');
end

cab2 = strrep(cab2, char([239 187 191]), '');

nomes2 = strtrim(strsplit(cab2, ','));

fmt2 = repmat('%s', 1, numel(nomes2));

M = textscan(fid, fmt2, ...
    'Delimiter', ',', ...
    'ReturnOnError', false);

fclose(fid);

if numel(M{1}) ~= N
    error('Manifesto e caracteristicas possuem tamanhos diferentes.');
end

% ============================================================
% 3. IDENTIFICACAO DAS COLUNAS
% ============================================================

cr = find(strcmp(nomes, 'registro'));
cy = find(strcmp(nomes, 'regiao'));
cf = find(strcmp(nomes, 'fase'));
cw = find(strcmp(nomes, 'janela'));

c_rms = find(strcmp(nomes, 'rms_A'));

c_imb = find(strcmp(nomes, ...
    'desequilibrio_rms_pct'));

c_fc = find(strcmp(nomes, ...
    'fator_crista'));

cm_linha = find(strcmp(nomes2, 'linha_csv'));
cm_reg = find(strcmp(nomes2, 'registro'));
cm_fase = find(strcmp(nomes2, 'fase'));
cm_regiao = find(strcmp(nomes2, 'regiao'));
cm_grupo = find(strcmp(nomes2, 'grupo'));

indices = [ ...
    cr cy cf cw ...
    c_rms c_imb c_fc ...
    cm_linha cm_reg cm_fase cm_regiao cm_grupo ...
];

if numel(indices) ~= 12
    error('Colunas obrigatorias ausentes ou duplicadas.');
end

registro = str2double(C{cr});
janela = str2double(C{cw});

fase_txt = C{cf};
regiao = C{cy};

rms = str2double(C{c_rms});
imb = str2double(C{c_imb});
fc = str2double(C{c_fc});

linha_manifesto = str2double(M{cm_linha});
reg_manifesto = str2double(M{cm_reg});

grupo = M{cm_grupo};

% ============================================================
% 4. VALIDACAO DOS DADOS E ROTULOS
% ============================================================

numericos = [registro janela rms imb fc];

if any(~isfinite(numericos(:)))
    error('Existem valores numericos invalidos.');
end

if any(registro ~= floor(registro)) || ...
   any(janela ~= floor(janela))
    error('Registro ou janela com identificador nao inteiro.');
end

% O numero da linha inclui o cabecalho.
linhas_esperadas = (2:N+1)';

if any(linha_manifesto ~= linhas_esperadas)
    error('Manifesto fora de ordem.');
end

if any(registro ~= reg_manifesto)
    error('Registro diferente entre CSV e manifesto.');
end

y = NaN(N, 1);
fase = NaN(N, 1);

for i = 1:N

    s = lower(strtrim(regiao{i}));
    p = upper(strtrim(fase_txt{i}));

    if strcmp(s, 'pre')
        y(i) = 0;
    elseif strcmp(s, 'pos')
        y(i) = 1;
    end

    if strcmp(p, 'A')
        fase(i) = 1;
    elseif strcmp(p, 'B')
        fase(i) = 2;
    elseif strcmp(p, 'C')
        fase(i) = 3;
    end

    % Conferir manifesto
    if ~strcmpi(strtrim(fase_txt{i}), ...
                strtrim(M{cm_fase}{i}))
        error('Fase inconsistente na linha %d.', i);
    end

    if ~strcmpi(strtrim(regiao{i}), ...
                strtrim(M{cm_regiao}{i}))
        error('Regiao inconsistente na linha %d.', i);
    end

end

if any(isnan(y))
    error('Rotulos pre/pos invalidos.');
end

if any(isnan(fase))
    error('Fases invalidas.');
end

treino = strcmp(grupo, 'treino');
valid = strcmp(grupo, 'validacao');
teste = strcmp(grupo, 'teste');

if any(~(treino | valid | teste))
    error('Grupo desconhecido no manifesto.');
end

if sum(treino) ~= 948 || ...
   sum(valid) ~= 237 || ...
   sum(teste) ~= 237

    error('Contagens diferentes do G3/E02.');

end

fprintf('Treinamento: %d linhas\n', sum(treino));
fprintf('Validacao: %d linhas\n', sum(valid));
fprintf('Teste reservado: %d linhas\n', sum(teste));

% ============================================================
% 5. VERIFICACAO DAS JANELAS TRIFASICAS
% ============================================================

% CORRECAO PRINCIPAL:
%
% A numeracao das janelas pode reiniciar nas regioes
% pre e pos.
%
% Portanto, a identificacao unica utiliza:
%
% [registro, rotulo_pre_pos, janela]

pares = unique([registro y janela], 'rows');

J = rows(pares);

% Armazenar indices de cada janela para reutilizacao.
indices_janelas = zeros(J, 3);

for k = 1:J

    idx = find( ...
        registro == pares(k, 1) & ...
        y == pares(k, 2) & ...
        janela == pares(k, 3));

    if numel(idx) ~= 3

        error(['Janela trifasica incompleta: ' ...
               'registro=%d, regiao=%d, ' ...
               'janela=%d, linhas=%d'], ...
               pares(k, 1), ...
               pares(k, 2), ...
               pares(k, 3), ...
               numel(idx));

    end

    % Garantir exatamente uma linha de cada fase.
    [fases_ordenadas, ordem] = sort(fase(idx));

    if ~isequal(fases_ordenadas(:)', [1 2 3])

        error(['Fases incorretas: ' ...
               'registro=%d, regiao=%d, janela=%d'], ...
               pares(k, 1), ...
               pares(k, 2), ...
               pares(k, 3));

    end

    idx = idx(ordem);

    % Conferir pertencimento ao mesmo conjunto.
    if numel(unique(grupo(idx))) ~= 1
        error('Janela com grupos inconsistentes.');
    end

    % Desequilibrio e uma caracteristica global.
    tolerancia = 1e-7 * ...
        max(1, max(abs(imb(idx))));

    if max(imb(idx)) - min(imb(idx)) > tolerancia
        error('Desequilibrio global inconsistente.');
    end

    indices_janelas(k, :) = idx(:)';

end

if J ~= 474
    error('Esperadas 474 janelas trifasicas.');
end

fprintf('\nJanelas trifasicas verificadas: %d\n', J);

% ============================================================
% 6. APRENDIZAGEM DOS LIMIARES
% ============================================================

% Percentis empiricos fixos:
%
% RMS:
%   cauda superior, percentil 99
%
% Desequilibrio:
%   cauda superior, percentil 99
%
% Fator de crista:
%   cauda inferior, percentil 1
%
% Utiliza exclusivamente observacoes PRE de TREINO.
%
% Estes percentis sao configuracoes preliminares,
% nao limites fisicos certificados.

alpha = 0.01;

lim_rms = zeros(3, 1);
lim_fc = zeros(3, 1);

for ph = 1:3

    idx = treino & y == 0 & fase == ph;

    valores_rms = sort(rms(idx));
    valores_fc = sort(fc(idx));

    nr = numel(valores_rms);
    nf = numel(valores_fc);

    if nr < 2 || nf < 2
        error('Dados pre-evento insuficientes.');
    end

    ir = max(1, ceil((1-alpha)*nr));
    ic = max(1, ceil(alpha*nf));

    lim_rms(ph) = valores_rms(ir);
    lim_fc(ph) = valores_fc(ic);

end

% Desequilibrio global: usar apenas a fase A
% para nao triplicar observacoes identicas.

idx_global = treino & y == 0 & fase == 1;

valores_imb = sort(imb(idx_global));

if numel(valores_imb) < 2
    error('Dados insuficientes para desequilibrio.');
end

ig = max(1, ceil((1-alpha)*numel(valores_imb)));

lim_imb = valores_imb(ig);

fprintf('\n--- LIMIARES APRENDIDOS NO TREINO ---\n');

for ph = 1:3

    fprintf('Fase %s: RMS > %.6f A; FC < %.6f\n', ...
        char(64+ph), ...
        lim_rms(ph), ...
        lim_fc(ph));

end

fprintf('Desequilibrio > %.6f %%\n', lim_imb);

% ============================================================
% 7. APLICACAO DOS DETECTORES
% ============================================================

pred_rms = zeros(N, 1);
pred_fc = zeros(N, 1);
pred_imb = zeros(N, 1);

for ph = 1:3

    idx = fase == ph;

    pred_rms(idx) = rms(idx) > lim_rms(ph);

    pred_fc(idx) = fc(idx) < lim_fc(ph);

end

pred_imb = imb > lim_imb;

% ============================================================
% 8. AGREGACAO POR JANELA TRIFASICA
% ============================================================

Y = zeros(J, 1);
G = cell(J, 1);
R = zeros(J, 1);

PR = zeros(J, 1);
PI = zeros(J, 1);
PF = zeros(J, 1);

for k = 1:J

    % Indices previamente verificados na secao 5.
    idx = indices_janelas(k, :);

    Y(k) = y(idx(1));
    G{k} = grupo{idx(1)};
    R(k) = registro(idx(1));

    % D1: alguma fase excedeu seu RMS?
    PR(k) = any(pred_rms(idx));

    % D2: decisao global de desequilibrio.
    PI(k) = pred_imb(idx(1));

    % D3: alguma fase possui fator de crista baixo?
    PF(k) = any(pred_fc(idx));

end

P = [PR PI PF];

nomes_det = { ...
    'D1_RMS', ...
    'D2_Desequilibrio', ...
    'D3_FatorCrista' ...
};

fprintf('\nJanelas agregadas: %d\n', J);

% ============================================================
% 9. METRICAS - TREINO E VALIDACAO
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E03 - RESULTADOS\n');
fprintf('========================================\n');

arquivo_metricas = fullfile( ...
    saida, 'G3_E03_Metricas.csv');

fm = fopen(arquivo_metricas, 'wt');

if fm < 0
    error('Nao foi possivel criar CSV de metricas.');
end

fprintf(fm, ...
    ['conjunto,detector,TP,FP,TN,FN,' ...
     'precisao,recall,F1,FPR\n']);

F1_val = zeros(3, 1);
FPR_val = zeros(3, 1);

conjuntos = {'treino', 'validacao'};

for g = 1:2

    idxg = strcmp(G, conjuntos{g});

    fprintf('\n%s\n', upper(conjuntos{g}));

    for d = 1:3

        yp = P(idxg, d);
        yt = Y(idxg);

        TP = sum(yp == 1 & yt == 1);
        FP = sum(yp == 1 & yt == 0);
        TN = sum(yp == 0 & yt == 0);
        FN = sum(yp == 0 & yt == 1);

        precisao = TP / max(1, TP+FP);
        recall = TP / max(1, TP+FN);

        f1 = 2*precisao*recall / ...
            max(eps, precisao+recall);

        fpr = FP / max(1, FP+TN);

        fprintf( ...
            '%s: F1=%.4f Recall=%.4f FPR=%.4f\n', ...
            nomes_det{d}, ...
            f1, recall, fpr);

        fprintf(fm, ...
            ['%s,%s,%d,%d,%d,%d,' ...
             '%.8f,%.8f,%.8f,%.8f\n'], ...
            conjuntos{g}, ...
            nomes_det{d}, ...
            TP, FP, TN, FN, ...
            precisao, recall, f1, fpr);

        if g == 2
            F1_val(d) = f1;
            FPR_val(d) = fpr;
        end

    end

end

fclose(fm);

% ============================================================
% 10. EXPORTAR LIMIARES
% ============================================================

arquivo_limiares = fullfile( ...
    saida, 'G3_E03_Limiares.csv');

fl = fopen(arquivo_limiares, 'wt');

if fl < 0
    error('Nao foi possivel criar CSV de limiares.');
end

fprintf(fl, 'detector,fase,operador,limiar\n');

for ph = 1:3

    fprintf(fl, 'RMS,%s,maior,%.12g\n', ...
        char(64+ph), lim_rms(ph));

    fprintf(fl, 'FatorCrista,%s,menor,%.12g\n', ...
        char(64+ph), lim_fc(ph));

end

fprintf(fl, ...
    'Desequilibrio,ABC,maior,%.12g\n', lim_imb);

fclose(fl);

% ============================================================
% 11. EXPORTAR DECISOES
% ============================================================

% Apenas treinamento e validacao.
% O teste permanece reservado.

arquivo_decisoes = fullfile( ...
    saida, 'G3_E03_Decisoes.csv');

fd = fopen(arquivo_decisoes, 'wt');

if fd < 0
    error('Nao foi possivel criar CSV de decisoes.');
end

fprintf(fd, ...
    ['registro,regiao,janela,grupo,rotulo,' ...
     'D1_RMS,D2_Imb,D3_Crista\n']);

for k = 1:J

    if strcmp(G{k}, 'teste')
        continue;
    end

    if Y(k) == 0
        nome_regiao = 'pre';
    else
        nome_regiao = 'pos';
    end

    fprintf(fd, ...
        '%d,%s,%d,%s,%d,%d,%d,%d\n', ...
        R(k), ...
        nome_regiao, ...
        pares(k, 3), ...
        G{k}, ...
        Y(k), ...
        P(k, 1), ...
        P(k, 2), ...
        P(k, 3));

end

fclose(fd);

% ============================================================
% 12. CONFIGURACAO GRAFICA
% ============================================================

motores = available_graphics_toolkits();

graficos_disponiveis = true;

if any(strcmp(motores, 'qt'))

    graphics_toolkit('qt');
    fprintf('\nMotor grafico: Qt\n');

elseif any(strcmp(motores, 'gnuplot'))

    graphics_toolkit('gnuplot');
    fprintf('\nMotor grafico: gnuplot\n');

else

    graficos_disponiveis = false;
    warning('Nenhum motor grafico disponivel.');

end

% ============================================================
% 13. FIGURA 1 - F1 NA VALIDACAO
% ============================================================

if graficos_disponiveis

    fig = figure( ...
        'visible', 'off', ...
        'color', 'w', ...
        'position', [100 100 1050 470]);

    barh(1:3, F1_val, 0.55);

    set(gca, ...
        'ytick', 1:3, ...
        'yticklabel', { ...
            'RMS', ...
            'Desequilibrio', ...
            'Fator de crista'}, ...
        'ydir', 'reverse', ...
        'fontsize', 12);

    xlim([0 1.12]);

    xlabel('F1 na validacao');
    title('G3/E03 - Comparacao dos detectores');

    grid on;

    for d = 1:3

        text( ...
            F1_val(d)+0.025, ...
            d, ...
            sprintf('%.3f', F1_val(d)), ...
            'fontsize', 11);

    end

    set(gcf, 'paperpositionmode', 'auto');

    print(fig, ...
        fullfile(saida, 'G3_E03_Fig01_F1.png'), ...
        '-dpng', '-r180');

    close(fig);

    % ========================================================
    % 14. FIGURA 2 - TAXA DE FALSOS ALARMES
    % ========================================================

    fig = figure( ...
        'visible', 'off', ...
        'color', 'w', ...
        'position', [100 100 1050 470]);

    valores_fpr = 100 * FPR_val;

    barh(1:3, valores_fpr, 0.55);

    set(gca, ...
        'ytick', 1:3, ...
        'yticklabel', { ...
            'RMS', ...
            'Desequilibrio', ...
            'Fator de crista'}, ...
        'ydir', 'reverse', ...
        'fontsize', 12);

    limite = max(5, 1.25*max(valores_fpr)+1);

    xlim([0 limite]);

    xlabel('Falsos positivos no pre-evento (%)');
    title('G3/E03 - Taxa de falsos alarmes');

    grid on;

    for d = 1:3

        text( ...
            valores_fpr(d)+0.02*limite, ...
            d, ...
            sprintf('%.2f%%', valores_fpr(d)), ...
            'fontsize', 11);

    end

    set(gcf, 'paperpositionmode', 'auto');

    print(fig, ...
        fullfile(saida, 'G3_E03_Fig02_FPR.png'), ...
        '-dpng', '-r180');

    close(fig);

end

% ============================================================
% 15. FINALIZACAO
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E03 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Linhas analisadas: %d\n', N);
fprintf('Janelas trifasicas: %d\n', J);
fprintf('Detectores: %d\n', columns(P));

fprintf('Treino: %d janelas\n', ...
    sum(strcmp(G, 'treino')));

fprintf('Validacao: %d janelas\n', ...
    sum(strcmp(G, 'validacao')));

fprintf('Teste reservado: %d janelas\n', ...
    sum(strcmp(G, 'teste')));

fprintf('\nArquivos salvos em:\n%s\n', saida);

fprintf('\nOBSERVACAO CIENTIFICA:\n');
fprintf('Rotulos pre/pos sao proxies temporais.\n');
fprintf('Nao existe referencia saudavel independente.\n');
fprintf('Nenhuma generalizacao fisica foi demonstrada.\n');
