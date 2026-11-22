
% ================================================================
% PESQUISA SRM - GATE 3
% EXPERIMENTO 01 - AUDITORIA ESTATISTICA E SEPARABILIDADE
%
% Versao revisada
% GNU Octave 11.3.0
%
% Entrada:
%   Gate2_Exp05/G2_E05_Caracteristicas.csv
%
% Saidas:
%   G3_E01_Qualidade.csv
%   G3_E01_Efeitos.csv
%   G3_E01_Consistencia.csv
%   G3_E01_Correlacao.csv
%   G3_E01_Fig01_Efeitos.png
%   G3_E01_Fig02_Correlacao.png
%   G3_E01_Fig03_Consistencia.png
%
% Nenhum arquivo MAT e aberto.
% Nenhum dado original e modificado.
% ================================================================

clear; clc; close all;

graphics_toolkit('gnuplot');
set(0, 'defaultfigurevisible', 'off');

% ================================================================
% 1. DIRETORIOS
% ================================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

arquivo = fullfile(base, ...
    'Gate2_Exp05', ...
    'G2_E05_Caracteristicas.csv');

saida = fullfile(base, 'Gate3_Exp01');

if ~exist(saida, 'dir')
    mkdir(saida);
end

if ~exist(arquivo, 'file')
    error('CSV nao encontrado: %s', arquivo);
end

fprintf('\n========================================\n');
fprintf('GATE 3 - EXPERIMENTO 01\n');
fprintf('AUDITORIA ESTATISTICA E SEPARABILIDADE\n');
fprintf('========================================\n');
fprintf('\nArquivo: %s\n', arquivo);

% ================================================================
% 2. LEITURA DO CSV
% ================================================================

fid = fopen(arquivo, 'r');

if fid < 0
    error('Nao foi possivel abrir o CSV.');
end

cabecalho = fgetl(fid);

if ~ischar(cabecalho)
    fclose(fid);
    error('CSV vazio.');
end

nomes = strsplit(cabecalho, ',');

for j = 1:numel(nomes)
    nomes{j} = strtrim(nomes{j});
end

ncol = numel(nomes);

formato = repmat('%s', 1, ncol);

C = textscan(fid, formato, ...
    'Delimiter', ',', ...
    'ReturnOnError', false);

fclose(fid);

N = numel(C{1});

for j = 1:ncol
    if numel(C{j}) ~= N
        error('CSV com colunas de comprimentos diferentes.');
    end
end

fprintf('Linhas lidas: %d\n', N);
fprintf('Colunas encontradas: %d\n', ncol);

% ================================================================
% 3. IDENTIFICACAO DOS METADADOS
% ================================================================

col_reg = find(strcmp(nomes, 'registro'), 1);
col_regiao = find(strcmp(nomes, 'regiao'), 1);
col_fase = find(strcmp(nomes, 'fase'), 1);

if isempty(col_reg) || ...
   isempty(col_regiao) || ...
   isempty(col_fase)

    fprintf('\nColunas encontradas:\n');

    for j = 1:ncol
        fprintf('%2d: %s\n', j, nomes{j});
    end

    error('Metadados obrigatorios nao encontrados.');
end

registro = str2double(C{col_reg});

regiao = C{col_regiao};
fase_txt = C{col_fase};

if any(~isfinite(registro))
    error('Identificadores de registro invalidos.');
end

% 0 = pre-evento
% 1 = pos-evento

y = NaN(N,1);
fase = NaN(N,1);

for i = 1:N

    s = lower(strtrim(regiao{i}));

    if strcmp(s, 'pre')
        y(i) = 0;
    elseif strcmp(s, 'pos')
        y(i) = 1;
    end

    p = upper(strtrim(fase_txt{i}));

    if strcmp(p, 'A')
        fase(i) = 1;
    elseif strcmp(p, 'B')
        fase(i) = 2;
    elseif strcmp(p, 'C')
        fase(i) = 3;
    end

end

if any(isnan(y))
    error('Valores invalidos na coluna regiao.');
end

if any(isnan(fase))
    error('Valores invalidos na coluna fase.');
end

regs = unique(registro);

fprintf('\n--- ESTRUTURA DA BASE ---\n');
fprintf('Registros: %d\n', numel(regs));
fprintf('Linhas pre-evento: %d\n', sum(y==0));
fprintf('Linhas pos-evento: %d\n', sum(y==1));

% ================================================================
% 4. SELECAO DAS CARACTERISTICAS
% ================================================================

% Metadados e identificadores nao entram na analise.
% Isso evita usar tempo do evento ou numero do registro
% como preditores artificiais.

excluir = { ...
    'registro', ...
    'regiao', ...
    'janela', ...
    't_inicio_s', ...
    't_fim_s', ...
    't_evento_s', ...
    'fs_Hz', ...
    'fase' ...
};

X = [];
features = {};

for j = 1:ncol

    if any(strcmp(nomes{j}, excluir))
        continue;
    end

    valores = str2double(C{j});

    % Validar que a coluna e numerica.
    coluna_numerica = true;

    for i = 1:N

        texto = lower(strtrim(C{j}{i}));

        if isnan(valores(i)) && ...
           ~any(strcmp(texto, {'nan','na',''}))

            coluna_numerica = false;
            break;

        end

    end

    if coluna_numerica

        X(:,end+1) = valores;
        features{end+1} = nomes{j};

    end

end

P = columns(X);

if P == 0
    error('Nenhuma caracteristica numerica encontrada.');
end

fprintf('Caracteristicas analisadas: %d\n', P);

% ================================================================
% 5. AUDITORIA DE QUALIDADE
% ================================================================

arquivo_qualidade = fullfile(saida, ...
    'G3_E01_Qualidade.csv');

fq = fopen(arquivo_qualidade, 'w');

fprintf(fq, ...
    'caracteristica,n_nan,n_inf,n_finitos,media,desvio,variancia_zero\n');

fprintf('\n--- AUDITORIA DE QUALIDADE ---\n');

for j = 1:P

    v = X(:,j);

    vf = v(isfinite(v));

    n_nan = sum(isnan(v));
    n_inf = sum(isinf(v));
    n_fin = numel(vf);

    if isempty(vf)

        media = NaN;
        desvio = NaN;
        constante = 1;

    else

        media = mean(vf);
        desvio = std(vf);
        constante = double(desvio == 0);

    end

    fprintf(fq, ...
        '%s,%d,%d,%d,%.10g,%.10g,%d\n', ...
        features{j},n_nan,n_inf,n_fin, ...
        media,desvio,constante);

    fprintf('%-28s NaN=%d Inf=%d DP=%.6g\n', ...
        features{j},n_nan,n_inf,desvio);

end

fclose(fq);

% ================================================================
% 6. TAMANHO DE EFEITO POR FASE
% ================================================================

% Cohen d descritivo com desvio padrao combinado.
% As janelas nao sao tratadas como independentes
% para inferencia estatistica.

D = NaN(P,3);
Delta = NaN(P,3);

arquivo_efeitos = fullfile(saida, ...
    'G3_E01_Efeitos.csv');

fe = fopen(arquivo_efeitos,'w');

fprintf(fe, ...
    'fase,caracteristica,media_pre,media_pos,delta,cohen_d\n');

fprintf('\n--- TAMANHO DE EFEITO POR FASE ---\n');

for ph = 1:3

    for j = 1:P

        a = X(y==0 & fase==ph,j);
        b = X(y==1 & fase==ph,j);

        a = a(isfinite(a));
        b = b(isfinite(b));

        media_a = NaN;
        media_b = NaN;
        delta = NaN;
        d = NaN;

        if numel(a)>=2 && numel(b)>=2

            media_a = mean(a);
            media_b = mean(b);

            delta = media_b-media_a;

            sp = sqrt( ...
                ((numel(a)-1)*var(a) + ...
                 (numel(b)-1)*var(b)) / ...
                (numel(a)+numel(b)-2));

            if sp > 0
                d = delta/sp;
            end

        end

        D(j,ph) = d;
        Delta(j,ph) = delta;

        fprintf(fe, ...
            '%s,%s,%.10g,%.10g,%.10g,%.10g\n', ...
            char(64+ph),features{j}, ...
            media_a,media_b,delta,d);

    end

end

fclose(fe);

% ================================================================
% 7. CONSISTENCIA ENTRE REGISTROS
% ================================================================

Cons = NaN(P,3);
Npos = zeros(P,3);
Nneg = zeros(P,3);
Nvalid = zeros(P,3);

arquivo_cons = fullfile(saida, ...
    'G3_E01_Consistencia.csv');

fc = fopen(arquivo_cons,'w');

fprintf(fc, ...
    'fase,caracteristica,n_registros,delta_positivo,delta_negativo,consistencia_pct\n');

for ph = 1:3

    for j = 1:P

        deltas = [];

        for rr = 1:numel(regs)

            r = regs(rr);

            a = X(registro==r & y==0 & fase==ph,j);
            b = X(registro==r & y==1 & fase==ph,j);

            a = a(isfinite(a));
            b = b(isfinite(b));

            if ~isempty(a) && ~isempty(b)

                deltas(end+1) = mean(b)-mean(a);

            end

        end

        np = sum(deltas>0);
        nn = sum(deltas<0);
        nv = numel(deltas);

        if nv > 0
            consistencia = 100*max(np,nn)/nv;
        else
            consistencia = NaN;
        end

        Cons(j,ph) = consistencia;
        Npos(j,ph) = np;
        Nneg(j,ph) = nn;
        Nvalid(j,ph) = nv;

        fprintf(fc, ...
            '%s,%s,%d,%d,%d,%.6f\n', ...
            char(64+ph),features{j}, ...
            nv,np,nn,consistencia);

    end

end

fclose(fc);

% ================================================================
% 8. MATRIZ DE CORRELACAO
% ================================================================

% Correlacao de Pearson por pares finitos.
% Descritiva: mistura de fases e regioes pode
% aumentar correlacoes artificialmente.

R = NaN(P,P);

for a = 1:P

    for b = 1:P

        ok = isfinite(X(:,a)) & isfinite(X(:,b));

        if sum(ok)>=3 && ...
           std(X(ok,a))>0 && ...
           std(X(ok,b))>0

            tmp = corrcoef(X(ok,a),X(ok,b));
            R(a,b) = tmp(1,2);

        end

    end

end

% Exportar matriz com cabecalhos
fr = fopen(fullfile(saida, ...
    'G3_E01_Correlacao.csv'),'w');

fprintf(fr,'caracteristica');

for j = 1:P
    fprintf(fr,',%s',features{j});
end

fprintf(fr,'\n');

for i = 1:P

    fprintf(fr,'%s',features{i});

    for j = 1:P
        fprintf(fr,',%.10g',R(i,j));
    end

    fprintf(fr,'\n');

end

fclose(fr);

% ================================================================
% 9. FIGURA 1 - TAMANHO DE EFEITO
% ================================================================

fig = figure('visible','off', ...
    'position',[100 100 1500 750]);

bar(D);

grid on;
ylabel('Cohen d (descritivo)');
xlabel('Caracteristica');
title('G3/E01 - Tamanho de efeito por fase');

legend('Fase A','Fase B','Fase C', ...
    'location','northeast');

set(gca,'xtick',1:P);
set(gca,'xticklabel',features);
set(gca,'xticklabelrotation',45);

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E01_Fig01_Efeitos.png'), ...
    '-dpng','-r200');

close(fig);

% ================================================================
% 10. FIGURA 2 - CORRELACAO
% ================================================================

fig = figure('visible','off', ...
    'position',[100 100 1200 1000]);

imagesc(R,[-1 1]);
axis square;

colorbar;
title('G3/E01 - Correlacao de Pearson');

set(gca,'xtick',1:P);
set(gca,'xticklabel',features);

set(gca,'ytick',1:P);
set(gca,'yticklabel',features);

set(gca,'xticklabelrotation',45);

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E01_Fig02_Correlacao.png'), ...
    '-dpng','-r200');

close(fig);

% ================================================================
% 11. FIGURA 3 - CONSISTENCIA
% ================================================================

fig = figure('visible','off', ...
    'position',[100 100 1500 750]);

bar(Cons);

grid on;
ylim([0 105]);

ylabel('Consistencia de direcao (%)');
xlabel('Caracteristica');

title('G3/E01 - Consistencia entre registros');

legend('Fase A','Fase B','Fase C', ...
    'location','northeast');

set(gca,'xtick',1:P);
set(gca,'xticklabel',features);
set(gca,'xticklabelrotation',45);

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E01_Fig03_Consistencia.png'), ...
    '-dpng','-r200');

close(fig);

% ================================================================
% 12. RESUMO DE CARACTERISTICAS PRIORITARIAS
% ================================================================

fprintf('\n--- CARACTERISTICAS MAIS SEPARADORAS ---\n');

for ph = 1:3

    valores = abs(D(:,ph));
    valores(~isfinite(valores)) = -Inf;

    [ordenado,ordem] = sort(valores,'descend');

    fprintf('\nFASE %s\n',char(64+ph));

    nmostrar = min(8,P);

    for k = 1:nmostrar

        j = ordem(k);

        if ~isfinite(ordenado(k))
            break;
        end

        fprintf('%2d. %-28s d=%+.4f | Cons=%.1f%%\n', ...
            k,features{j}, ...
            D(j,ph),Cons(j,ph));

    end

end

% ================================================================
% 13. FINALIZACAO
% ================================================================

fprintf('\n========================================\n');
fprintf('G3/E01 - PROCESSAMENTO CONCLUIDO\n');
fprintf('========================================\n');

fprintf('Registros analisados: %d\n',numel(regs));
fprintf('Observacoes: %d\n',N);
fprintf('Caracteristicas: %d\n',P);

fprintf('\nResultados em:\n%s\n',saida);

fprintf('\nNenhum modelo foi treinado nesta etapa.\n');
fprintf('Aguardando avaliacao dos resultados.\n');
