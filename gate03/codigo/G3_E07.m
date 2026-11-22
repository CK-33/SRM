
% ================================================================
% G3/E07 - ANALISE TEMPORAL E PERSISTENCIA
% GNU Octave 11.3.0
% Versao revisada
% ================================================================
% Utiliza exclusivamente os CSVs produzidos anteriormente.
% Nao acessa arquivos MAT.
% Particoes recuperadas do manifesto G3/E02.
% Teste reservado nao avaliado.
% Motor grafico: FLTK -> GNUPLOT.
% ================================================================

clear;
clc;
close all;

% ================================================================
% 1. CONFIGURACOES
% ================================================================

ROOT = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

ENTRADA = fullfile(ROOT, 'Gate2_Exp05', ...
                  'G2_E05_Caracteristicas_Corrigido.csv');

MANIFESTO = fullfile(ROOT, 'Gate3_Exp02', ...
                    'G3_E02_Manifesto.csv');

SAIDA = fullfile(ROOT, 'Gate3_Exp07');

if exist(SAIDA, 'dir') ~= 7
    mkdir(SAIDA);
end

HISTORICOS = [1 2 3 5];
PERSISTENCIAS = [1 2 3 4 5];

T_JANELA = 0.01308;
LAMBDA = 0.01;
TOLERANCIA = 0.05;
MIN_ESTAVEIS = 5;

rng(42);

fprintf('\n');
fprintf('========================================\n');
fprintf('G3/E07 - ANALISE TEMPORAL\n');
fprintf('========================================\n');

% ================================================================
% 2. FUNCOES AUXILIARES
% ================================================================

% ------------------------------------------------
% 2.1 Interpretador robusto de linha CSV
% ------------------------------------------------

function campos = separar_csv(linha, delimitador)

    campos = {};
    atual = '';
    aspas = false;

    i = 1;

    while i <= length(linha)

        c = linha(i);

        if c == '"'

            if aspas && i < length(linha) && ...
                    linha(i+1) == '"'

                atual(end+1) = '"';
                i = i + 1;

            else

                aspas = ~aspas;

            end

        elseif c == delimitador && ~aspas

            campos{end+1} = strtrim(atual);
            atual = '';

        else

            atual(end+1) = c;

        end

        i = i + 1;

    end

    if aspas
        error('Campo CSV com aspas nao fechadas.');
    end

    campos{end+1} = strtrim(atual);

end


% ------------------------------------------------
% 2.2 Leitura robusta de arquivo CSV
% ------------------------------------------------

function [cab, dados] = ler_csv(arquivo)

    fid = fopen(arquivo, 'r');

    if fid < 0
        error('Nao foi possivel abrir: %s', arquivo);
    end

    unwind_protect

        linha = fgetl(fid);

        if ~ischar(linha)
            error('Arquivo CSV vazio: %s', arquivo);
        end

        % Remocao de BOM UTF-8.
        if numel(linha) >= 3

            if isequal(double(linha(1:3)), ...
                       [239 187 191])

                linha = linha(4:end);

            end

        end

        % Identificacao automatica do delimitador.
        delimitadores = {',', ';', char(9)};

        contagens = zeros(1,3);

        for k = 1:3

            p = separar_csv(linha, delimitadores{k});
            contagens(k) = numel(p);

        end

        [~,k] = max(contagens);

        delimitador = delimitadores{k};

        cab = separar_csv(linha, delimitador);

        % Remover colunas vazias excedentes no final.
        while numel(cab)>1 && isempty(cab{end})
            cab(end) = [];
        end

        ncol = numel(cab);

        if any(cellfun(@isempty,cab))
            error('Cabecalho possui colunas sem nome.');
        end

        dados = cell(0,ncol);

        nlinha = 1;

        while true

            linha = fgetl(fid);

            if ~ischar(linha)
                break;
            end

            nlinha = nlinha + 1;

            if isempty(strtrim(linha))
                continue;
            end

            try

                p = separar_csv(linha,delimitador);

            catch err

                error('CSV %s, linha %d: %s', ...
                      arquivo,nlinha,err.message);

            end

            while numel(p)>ncol && isempty(p{end})
                p(end) = [];
            end

            if numel(p) ~= ncol

                fprintf('\n--- ERRO DE ESTRUTURA CSV ---\n');

                fprintf('Arquivo: %s\n',arquivo);
                fprintf('Linha: %d\n',nlinha);
                fprintf('Colunas esperadas: %d\n',ncol);
                fprintf('Colunas encontradas: %d\n',numel(p));

                fprintf('\nCabecalho:\n');
                disp(cab');

                fprintf('\nLinha original:\n');
                disp(linha);

                error('Estrutura CSV inconsistente.');

            end

            dados(end+1,:) = p;

        end

    unwind_protect_cleanup

        fclose(fid);

    end_unwind_protect

    fprintf('\nArquivo importado:\n%s\n',arquivo);

    fprintf('Linhas: %d | Colunas: %d\n', ...
            rows(dados),columns(dados));

end


% ------------------------------------------------
% 2.3 Identificacao de colunas
% ------------------------------------------------

function idx = achar_coluna(cab,alternativas)

    for k = 1:numel(alternativas)

        idx = find(strcmpi(cab,alternativas{k}),1);

        if ~isempty(idx)
            return;
        end

    end

    fprintf('\nColunas disponiveis:\n');
    disp(cab');

    error('Coluna nao encontrada: %s', ...
          strjoin(alternativas,' / '));

end


% ------------------------------------------------
% 2.4 Conversao de registros
% ------------------------------------------------

function v = converter_registro(c)

    v = zeros(numel(c),1);

    for i = 1:numel(c)

        v(i) = str2double(c{i});

        if isnan(v(i))

            tok = regexp(c{i},'\d+','match');

            if isempty(tok)
                error('Registro invalido: %s',c{i});
            end

            v(i) = str2double(tok{end});

        end

    end

    if any(~isfinite(v)) || ...
       any(v~=fix(v)) || ...
       any(v<1 | v>12)

        error('Identificadores de registro invalidos.');

    end

end


% ------------------------------------------------
% 2.5 Conversao de classes
% ------------------------------------------------

function y = converter_classe(c)

    y = zeros(numel(c),1);

    for i = 1:numel(c)

        s = lower(strtrim(c{i}));

        if strcmp(s,'0') || strncmp(s,'pre',3)

            y(i) = 0;

        elseif strcmp(s,'1') || ...
               strncmp(s,'pos',3) || ...
               strncmp(s,'post',4)

            y(i) = 1;

        else

            error('Classe desconhecida: %s',c{i});

        end

    end

end


% ------------------------------------------------
% 2.6 Conversao de fases
% ------------------------------------------------

function f = converter_fase(s)

    s = upper(strtrim(s));

    if any(strcmp(s, ...
           {'A','1','FASEA','FASE A','PHASE A'}))

        f = 1;

    elseif any(strcmp(s, ...
           {'B','2','FASEB','FASE B','PHASE B'}))

        f = 2;

    elseif any(strcmp(s, ...
           {'C','3','FASEC','FASE C','PHASE C'}))

        f = 3;

    else

        error('Fase desconhecida: %s',s);

    end

end


% ------------------------------------------------
% 2.7 Funcao sigmoide
% ------------------------------------------------

function p = sigmoide(z)

    z = max(min(z,40),-40);

    p = 1./(1+exp(-z));

end


% ------------------------------------------------
% 2.8 Regressao logistica L2
% ------------------------------------------------

function w = treinar_logistica(X,y,lambda)

    X = [ones(rows(X),1),X];

    n = rows(X);
    d = columns(X);

    w = zeros(d,1);

    R = eye(d);
    R(1,1) = 0;

    for it = 1:200

        p = sigmoide(X*w);

        g = X'*(p-y)/n + lambda*R*w;

        q = max(p.*(1-p),1e-8);

        H = X'*bsxfun(@times,X,q)/n ...
            + lambda*R + 1e-8*eye(d);

        passo = H\g;

        wnovo = w-passo;

        if norm(wnovo-w)<1e-7

            w = wnovo;
            break;

        end

        w = wnovo;

    end

end


% ------------------------------------------------
% 2.9 Metricas de classificacao
% ------------------------------------------------

function [acc,prec,rec,fpr,f1] = metricas(y,pred)

    TP = sum(y==1 & pred==1);
    TN = sum(y==0 & pred==0);
    FP = sum(y==0 & pred==1);
    FN = sum(y==1 & pred==0);

    acc = (TP+TN)/max(TP+TN+FP+FN,1);

    prec = TP/max(TP+FP,1);
    rec = TP/max(TP+FN,1);
    fpr = FP/max(FP+TN,1);

    f1 = 2*prec*rec/max(prec+rec,eps);

end


% ------------------------------------------------
% 2.10 Detector com persistencia
% ------------------------------------------------

function a = alarme_persistente(p,m)

    a = zeros(numel(p),1);

    contagem = 0;

    for i = 1:numel(p)

        if p(i)

            contagem = contagem+1;

        else

            contagem = 0;

        end

        a(i) = contagem>=m;

    end

end


% ------------------------------------------------
% 2.11 Construcao do historico temporal
% ------------------------------------------------

function [Xt,idx] = construir_historico( ...
                            X,reg,tempo,H,T)

    n = rows(X);
    d = columns(X);

    Xt = zeros(n,d*H);

    valido = false(n,1);

    for i = H:n

        jj = i-H+1:i;

        % Proibido atravessar registros.
        if any(reg(jj)~=reg(i))
            continue;
        end

        % Proibido atravessar lacunas temporais.
        if H>1

            dt = diff(tempo(jj));

            if any(dt<=0) || ...
               any(abs(dt-T)>0.05*T)

                continue;

            end

        end

        % Somente janela atual e historico passado.
        for k = 1:H

            cols = (k-1)*d+1:k*d;

            Xt(i,cols) = X(i-k+1,:);

        end

        valido(i) = true;

    end

    idx = find(valido);

    Xt = Xt(idx,:);

end


% ------------------------------------------------
% 2.12 Atraso de deteccao
% ------------------------------------------------

function [primeiro,atraso] = primeiro_alarme(a,y,t,T)

    ipos = find(y==1,1);

    primeiro = NaN;
    atraso = NaN;

    if isempty(ipos)
        return;
    end

    k = find(a(ipos:end)==1,1);

    if ~isempty(k)

        primeiro = k;

        j = ipos+k-1;

        % Tempo medido desde o inicio da primeira
        % janela pos-evento ate o final da janela
        % em que o alarme foi confirmado.

        atraso = t(j)-t(ipos)+T;

    end

end


% ------------------------------------------------
% 2.13 Estabilizacao retrospectiva
% ------------------------------------------------

function [idx,vf] = estabilizacao(x,t,tol,M,T)

    x = x(:);
    t = t(:);

    idx = NaN;
    vf = NaN;

    n = numel(x);

    if n<max(5,M)
        return;
    end

    vf = median(x(end-4:end));

    faixa = tol*max(abs(vf),eps);

    for i = 1:n-M+1

        % Exigir continuidade das janelas.
        dt = diff(t(i:end));

        if any(abs(dt-T)>0.05*T)
            continue;
        end

        % Exigir permanencia na faixa ate o fim.
        if all(abs(x(i:end)-vf)<=faixa)

            idx = i;
            return;

        end

    end

end


% ------------------------------------------------
% 2.14 Estilo dos graficos
% ------------------------------------------------

function estilo_eixos(ax)

    set(ax, ...
        'fontsize',12, ...
        'fontname','Arial', ...
        'linewidth',0.8, ...
        'box','on', ...
        'layer','top', ...
        'xgrid','off', ...
        'ygrid','on', ...
        'gridlinestyle',':');

    try

        set(ax,'gridalpha',0.18);

    catch
    end

end


function fig = nova_figura()

    fig = figure( ...
        'visible','off', ...
        'color',[1 1 1], ...
        'position',[100 100 1250 760]);

    set(fig,'paperpositionmode','auto');
    set(fig,'inverthardcopy','off');

end


% ------------------------------------------------
% 2.15 Exportacao e auditoria de PNG
% ------------------------------------------------

function salvar_figura(fig,destino)

    drawnow();

    if exist(destino,'file')==2
        delete(destino);
    end

    print(fig,destino,'-dpng','-r220');

    info = dir(destino);

    if isempty(info) || info.bytes<1000

        error('Falha na exportacao: %s',destino);

    end

    % Verificacao de imagem vazia ou uniforme.
    img = imread(destino);

    if isempty(img) || ...
       max(double(img(:)))-min(double(img(:)))<5

        error('Imagem vazia ou uniforme: %s',destino);

    end

    close(fig);

end


% ================================================================
% 3. LEITURA DOS DADOS
% ================================================================

fprintf('\n--- IMPORTACAO DOS DADOS ---\n');

[cab,D] = ler_csv(ENTRADA);

[cm,M] = ler_csv(MANIFESTO);

assert(rows(D)==1422, ...
       'Numero inesperado de linhas no CSV.');

assert(columns(D)==24, ...
       'Numero inesperado de colunas no CSV.');

assert(rows(D)==rows(M), ...
       'Manifesto e CSV possuem tamanhos diferentes.');

% ------------------------------------------------
% Identificar campos do CSV de caracteristicas.
% ------------------------------------------------

c_reg = achar_coluna(cab, ...
    {'registro','record','id_registro','arquivo'});

c_fase = achar_coluna(cab, ...
    {'fase','phase'});

c_classe = achar_coluna(cab, ...
    {'regiao','classe','label','estado','periodo'});

c_tempo = achar_coluna(cab, ...
    {'t_inicio_s','t_inicio','tempo_inicio_s', ...
     'tempo_inicio','inicio_s','t0_s','t0', ...
     't_ini_s','window_start_s','t_ini', ...
     'inicio','inicio_janela_s','t_start_s'});

% ------------------------------------------------
% Identificar campos do manifesto.
% ------------------------------------------------

m_linha = achar_coluna(cm,{'linha_csv'});
m_reg = achar_coluna(cm,{'registro'});
m_fase = achar_coluna(cm,{'fase'});
m_classe = achar_coluna(cm,{'regiao'});
m_grupo = achar_coluna(cm,{'grupo'});

% ------------------------------------------------
% Recuperar e ordenar manifesto.
% ------------------------------------------------

linhas_manifesto = str2double(M(:,m_linha));

[linhas_manifesto,ordm] = sort(linhas_manifesto);

M = M(ordm,:);

esperado = (2:rows(D)+1)';

assert(isequal(linhas_manifesto,esperado), ...
       'Indices do manifesto nao correspondem ao CSV.');

reg_original = converter_registro(D(:,c_reg));

classe_original = converter_classe(D(:,c_classe));

reg_manifesto = converter_registro(M(:,m_reg));

classe_manifesto = converter_classe(M(:,m_classe));

assert(isequal(reg_original,reg_manifesto), ...
       'Registros inconsistentes entre CSV e manifesto.');

assert(isequal(classe_original,classe_manifesto), ...
       'Classes inconsistentes entre CSV e manifesto.');

% ------------------------------------------------
% Verificar fases.
% ------------------------------------------------

for i = 1:rows(D)

    f1 = converter_fase(D{i,c_fase});
    f2 = converter_fase(M{i,m_fase});

    assert(f1==f2, ...
           'Inconsistencia de fase no manifesto.');

end

% ------------------------------------------------
% Recuperar particoes.
% ------------------------------------------------

grupos = cellfun(@lower,M(:,m_grupo), ...
                 'uniformoutput',false);

grupo_treino = strcmp(grupos,'treino');

grupo_valid = strcmp(grupos,'validacao');

grupo_teste = strcmp(grupos,'teste');

assert(all(grupo_treino | grupo_valid | grupo_teste), ...
       'Particao desconhecida no manifesto.');

REG_TREINO = unique(reg_original(grupo_treino))';

REG_VALID = unique(reg_original(grupo_valid))';

REG_TESTE = unique(reg_original(grupo_teste))';

assert(isempty(intersect(REG_TREINO,REG_VALID)));
assert(isempty(intersect(REG_TREINO,REG_TESTE)));
assert(isempty(intersect(REG_VALID,REG_TESTE)));

assert(isequal(REG_TESTE,[6 12]), ...
       'Registros reservados diferentes do protocolo.');

assert(sum(grupo_treino)==948);
assert(sum(grupo_valid)==237);
assert(sum(grupo_teste)==237);

fprintf('\n--- PARTICOES AUDITADAS ---\n');

fprintf('Treinamento: ');
fprintf('%02d ',REG_TREINO);
fprintf('\n');

fprintf('Validacao: ');
fprintf('%02d ',REG_VALID);
fprintf('\n');

fprintf('Teste reservado: ');
fprintf('%02d ',REG_TESTE);
fprintf('\n');

% ------------------------------------------------
% Excluir teste antes de processar caracteristicas.
% ------------------------------------------------

n_teste = sum(grupo_teste)/3;

desenvolvimento = ~grupo_teste;

D = D(desenvolvimento,:);

reg_original = reg_original(desenvolvimento);

classe_original = classe_original(desenvolvimento);

tempo_original = str2double(D(:,c_tempo));

if any(~isfinite(tempo_original))

    error('Tempos invalidos no desenvolvimento.');

end

% ================================================================
% 4. CARACTERISTICAS
% ================================================================

features = { ...
    'rms_A', ...
    'media_A', ...
    'desvio_A', ...
    'pico_abs_A', ...
    'pico_a_pico_A', ...
    'fator_crista', ...
    'mae_derivada_A_s', ...
    'rms_derivada_A_s', ...
    'energia_A2_s', ...
    'potencia_0_5k', ...
    'potencia_5_20k', ...
    'potencia_20_50k', ...
    'potencia_50_100k', ...
    'centroide_Hz', ...
    'rms_trifasico_A', ...
    'desequilibrio_rms_pct'};

F = zeros(rows(D),16);

for j = 1:16

    c = achar_coluna(cab,{features{j}});

    F(:,j) = str2double(D(:,c));

end

if any(~isfinite(F(:)))

    error('NaN ou Inf nas caracteristicas.');

end

% ================================================================
% 5. RECONSTRUCAO TRIFASICA
% ================================================================

% 14 caracteristicas por fase.
% 2 caracteristicas globais.
% Total: 44.

chaves = unique( ...
    [reg_original tempo_original],'rows');

N = rows(chaves);

X = zeros(N,44);

reg = chaves(:,1);
tempo = chaves(:,2);

y = zeros(N,1);

for i = 1:N

    jj = find( ...
        reg_original==reg(i) & ...
        tempo_original==tempo(i));

    assert(numel(jj)==3, ...
           'Janela sem exatamente tres fases.');

    assert(numel(unique(classe_original(jj)))==1, ...
           'Rotulos inconsistentes na janela.');

    y(i) = classe_original(jj(1));

    fases_encontradas = zeros(1,3);

    for k = 1:numel(jj)

        f = converter_fase(D{jj(k),c_fase});

        fases_encontradas(f) = ...
            fases_encontradas(f)+1;

        cols = (f-1)*14+1:f*14;

        X(i,cols) = F(jj(k),1:14);

    end

    assert(all(fases_encontradas==1), ...
           'Fases duplicadas ou ausentes.');

    globais = F(jj,15:16);

    assert(all(max(globais)-min(globais)<=1e-6), ...
           'Caracteristicas globais inconsistentes.');

    X(i,43:44) = globais(1,:);

end

[~,ord] = sortrows([reg tempo],[1 2]);

X = X(ord,:);
y = y(ord);

reg = reg(ord);
tempo = tempo(ord);

treino = ismember(reg,REG_TREINO);
valid = ismember(reg,REG_VALID);

assert(all(treino | valid));
assert(~any(treino & valid));

assert(N==395);
assert(sum(treino)==316);
assert(sum(valid)==79);
assert(n_teste==79);

assert(~any(ismember(reg,REG_TESTE)));

fprintf('\n--- JANELAS RECONSTRUIDAS ---\n');

fprintf('Treinamento: %d\n',sum(treino));
fprintf('Validacao: %d\n',sum(valid));
fprintf('Teste reservado: %d\n',n_teste);

fprintf('Caracteristicas por janela: %d\n', ...
        columns(X));

S = X(:,[1 15 29 44]);

nomes_S = {'RMS_A','RMS_B','RMS_C','Desequilibrio'};

registros = unique(reg);

% ================================================================
% 6. AUDITORIA TEMPORAL
% ================================================================

fprintf('\n--- AUDITORIA TEMPORAL ---\n');

for r = registros'

    ii = find(reg==r);

    dt = diff(tempo(ii));

    assert(all(dt>0), ...
           'Tempos repetidos ou fora de ordem.');

    assert(any(y(ii)==0) && any(y(ii)==1), ...
           'Registro sem as duas regioes.');

    assert(all(diff(y(ii))>=0), ...
           'Ordem temporal pre/pos inconsistente.');

    lacunas = sum(abs(dt-T_JANELA)>0.05*T_JANELA);

    fprintf(['Registro %02d | Janelas=%d | ' ...
             'Pre=%d | Pos=%d | Lacunas=%d\n'], ...
            r,numel(ii), ...
            sum(y(ii)==0), ...
            sum(y(ii)==1),lacunas);

end

% ================================================================
% 7. EVOLUCAO TEMPORAL
% ================================================================

fprintf('\n--- EVOLUCAO TEMPORAL ---\n');

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Evolucao_Temporal.csv'),'w');

assert(fid>=0);

fprintf(fid,['registro,classe,t_inicio_s,t_relativo_ms,' ...
    'rms_A,rms_B,rms_C,desequilibrio,' ...
    'derivada_rms_A,derivada_rms_B,' ...
    'derivada_rms_C,derivada_desequilibrio\n']);

for r = registros'

    ii = find(reg==r);

    ipos = find(y(ii)==1,1);

    t0 = tempo(ii(ipos));

    dS = NaN(numel(ii),4);

    for k = 2:numel(ii)

        dt = tempo(ii(k))-tempo(ii(k-1));

        if abs(dt-T_JANELA)<=0.05*T_JANELA

            dS(k,:) = ...
                (S(ii(k),:)-S(ii(k-1),:))/dt;

        end

    end

    for k = 1:numel(ii)

        i = ii(k);

        fprintf(fid,'%d,%d,%.10f,%.8f', ...
            r,y(i),tempo(i), ...
            1000*(tempo(i)-t0));

        fprintf(fid,',%.10g',S(i,:));
        fprintf(fid,',%.10g',dS(k,:));

        fprintf(fid,'\n');

    end

end

fclose(fid);

fprintf('Evolucao temporal exportada.\n');

% ================================================================
% 8. CLASSIFICACAO TEMPORAL
% ================================================================

fprintf('\n--- CLASSIFICACAO TEMPORAL ---\n');

% Todos os historicos serao comparados usando
% exatamente as mesmas janelas elegiveis.

[~,idx_comum] = construir_historico( ...
    X,reg,tempo,max(HISTORICOS),T_JANELA);

assert(~isempty(idx_comum), ...
       'Nenhuma janela elegivel para H5.');

met = zeros(numel(HISTORICOS),5);

probs = NaN(N,numel(HISTORICOS));

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Classificacao_Temporal.csv'),'w');

assert(fid>=0);

fprintf(fid,['historico,n_treino,n_validacao,' ...
    'n_features,acuracia,precisao,recall,fpr,f1\n']);

for k = 1:numel(HISTORICOS)

    H = HISTORICOS(k);

    [Xt,idx] = construir_historico( ...
        X,reg,tempo,H,T_JANELA);

    [ok,local] = ismember(idx_comum,idx);

    assert(all(ok));

    Xt = Xt(local,:);

    idx = idx_comum;

    it = treino(idx);
    iv = valid(idx);

    Xtr = Xt(it,:);
    Xva = Xt(iv,:);

    ytr = y(idx(it));
    yva = y(idx(iv));

    assert(numel(unique(ytr))==2, ...
           'Treinamento sem as duas classes.');

    assert(numel(unique(yva))==2, ...
           'Validacao sem as duas classes.');

    % Normalizacao exclusivamente pelo treino.
    mu = mean(Xtr,1);

    sd = std(Xtr,0,1);

    ativos = sd>1e-12;

    Xtr = bsxfun(@rdivide, ...
        bsxfun(@minus,Xtr(:,ativos),mu(ativos)), ...
        sd(ativos));

    Xva = bsxfun(@rdivide, ...
        bsxfun(@minus,Xva(:,ativos),mu(ativos)), ...
        sd(ativos));

    w = treinar_logistica(Xtr,ytr,LAMBDA);

    p = sigmoide([ones(rows(Xva),1) Xva]*w);

    pred = p>=0.5;

    [acc,prec,rec,fpr,f1] = metricas(yva,pred);

    met(k,:) = [acc prec rec fpr f1];

    probs(idx(iv),k) = p;

    fprintf(['H%d | Treino=%d | Validacao=%d | ' ...
             'Features=%d\n'], ...
             H,sum(it),sum(iv),sum(ativos));

    fprintf(['     ACC=%.4f F1=%.4f ' ...
             'Recall=%.4f FPR=%.4f\n'], ...
             acc,f1,rec,fpr);

    fprintf(fid, ...
        '%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f\n', ...
        H,sum(it),sum(iv),sum(ativos), ...
        acc,prec,rec,fpr,f1);

end

fclose(fid);

% ================================================================
% 9. PERSISTENCIA DOS ALARMES
% ================================================================

fprintf('\n--- PERSISTENCIA DOS ALARMES ---\n');

% Limiar fixado na versao anterior do G3/E07.
% Nao reajustado pela validacao.

LIMIAR_RMS_B = 9.174299;

fprintf('Limiar RMS B: %.6f A\n',LIMIAR_RMS_B);

pers = zeros(numel(PERSISTENCIAS),3);

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Persistencia.csv'),'w');

assert(fid>=0);

fprintf(fid,['persistencia,registro,recall,fpr,f1,' ...
    'primeira_janela,atraso_relativo_ms\n']);

for k = 1:numel(PERSISTENCIAS)

    P = PERSISTENCIAS(k);

    todos_y = [];
    todos_a = [];
    atrasos = [];

    for r = REG_VALID

        ii = find(reg==r);

        raw = S(ii,2)>LIMIAR_RMS_B;

        % Reiniciar persistencia nas lacunas.
        limites = [1; ...
            find(abs(diff(tempo(ii))-T_JANELA) ...
                 >0.05*T_JANELA)+1; ...
            numel(ii)+1];

        a = zeros(numel(ii),1);

        for j = 1:numel(limites)-1

            seg = limites(j):limites(j+1)-1;

            a(seg) = alarme_persistente(raw(seg),P);

        end

        [~,~,rec,fpr,f1] = metricas(y(ii),a);

        [primeiro,atraso] = primeiro_alarme( ...
            a,y(ii),tempo(ii),T_JANELA);

        todos_y = [todos_y; y(ii)];

        todos_a = [todos_a; a];

        atrasos(end+1) = atraso;

        fprintf(fid, ...
            '%d,%d,%.8f,%.8f,%.8f,%.8g,%.8g\n', ...
            P,r,rec,fpr,f1,primeiro,atraso*1000);

    end

    [~,~,rec,fpr,f1] = metricas(todos_y,todos_a);

    detectados = atrasos(isfinite(atrasos));

    if isempty(detectados)

        atraso_medio = NaN;

    else

        atraso_medio = mean(detectados);

    end

    pers(k,:) = [rec fpr atraso_medio];

    fprintf(['M=%d | Recall=%.4f | FPR=%.4f | ' ...
             'F1=%.4f\n'],P,rec,fpr,f1);

    fprintf(['      Atraso=%.3f ms | ' ...
             'Detectados=%d/%d\n'], ...
             atraso_medio*1000, ...
             numel(detectados),numel(REG_VALID));

end

fclose(fid);

% ================================================================
% 10. ESTABILIZACAO RETROSPECTIVA
% ================================================================

fprintf('\n--- ESTABILIZACAO POS-EVENTO ---\n');

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Estabilizacao.csv'),'w');

assert(fid>=0);

fprintf(fid,['registro,particao,caracteristica,' ...
    'referencia_final,janelas_pos_evento,' ...
    'primeira_janela_estavel,tempo_relativo_ms\n']);

for r = registros'

    ii = find(reg==r & y==1);

    if ismember(r,REG_TREINO)

        particao = 'treino';

    else

        particao = 'validacao';

    end

    for j = 1:4

        [k,vf] = estabilizacao( ...
            S(ii,j),tempo(ii), ...
            TOLERANCIA,MIN_ESTAVEIS,T_JANELA);

        ts = NaN;

        if isfinite(k)

            ts = 1000*(tempo(ii(k))-tempo(ii(1)));

        end

        fprintf(fid, ...
            '%d,%s,%s,%.8g,%d,%.8g,%.8g\n', ...
            r,particao,nomes_S{j},vf, ...
            numel(ii),k,ts);

        if j==2 || j==4

            fprintf(['Registro %02d | %-15s | ' ...
                     'Janela=%g | Tempo=%.3f ms\n'], ...
                     r,nomes_S{j},k,ts);

        end

    end

end

fclose(fid);

% ================================================================
% 11. EXPORTAR PREDICOES
% ================================================================

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Predicoes.csv'),'w');

assert(fid>=0);

fprintf(fid,'registro,t_inicio_s,classe');

for H = HISTORICOS

    fprintf(fid,',H%d',H);

end

fprintf(fid,'\n');

for i = find(valid)'

    fprintf(fid,'%d,%.10f,%d', ...
            reg(i),tempo(i),y(i));

    fprintf(fid,',%.10g',probs(i,:));

    fprintf(fid,'\n');

end

fclose(fid);

% ================================================================
% 12. FUNCAO DE GERACAO DOS GRAFICOS
% ================================================================

function gerar_figuras(S,reg,tempo,y,valid, ...
                       Hs,met,probs,Ps,pers,saida,T)

    azul = [0.00 0.35 0.65];

    laranja = [0.85 0.32 0.10];

    verde = [0.12 0.55 0.38];

    cores = [azul; laranja];

    registros = unique(reg(valid));

    nomes = {'RMS fase A','RMS fase B', ...
             'RMS fase C','Desequilibrio'};

    unidades = {'Corrente RMS (A)', ...
                'Corrente RMS (A)', ...
                'Corrente RMS (A)', ...
                'Desequilibrio (%)'};

    % ============================================================
    % FIGURAS 01 A 04 - EVOLUCAO TEMPORAL
    % ============================================================

    for j = 1:4

        fig = nova_figura();

        hold on;

        hh = [];

        labels = {};

        for k = 1:numel(registros)

            ir = find(reg==registros(k));

            primeiro_pos = find(y(ir)==1,1);

            t0 = tempo(ir(primeiro_pos));

            tx = (tempo(ir)-t0)*1000;

            cor = cores(mod(k-1,2)+1,:);

            % Marcadores individuais evitam
            % representar lacunas como dados observados.

            hh(k) = plot(tx,S(ir,j),'o', ...
                'color',cor, ...
                'linewidth',1.4, ...
                'markersize',4);

            % Linhas apenas em intervalos contiguos.
            for l = 2:numel(ir)

                dt = tempo(ir(l))-tempo(ir(l-1));

                if abs(dt-T)<=0.05*T

                    plot(tx(l-1:l), ...
                         S(ir(l-1:l),j),'-', ...
                         'color',cor, ...
                         'linewidth',1.5, ...
                         'handlevisibility','off');

                end

            end

            labels{k} = sprintf( ...
                'Registro %02d',registros(k));

        end

        yl = ylim();

        hh(end+1) = plot([0 0],yl,'--k', ...
                         'linewidth',1.3);

        labels{end+1} = 'Primeira janela pos-evento';

        legend(hh,labels, ...
               'location','northwest');

        xlabel('Tempo relativo (ms)');

        ylabel(unidades{j});

        title(['G3/E07 - Evolucao temporal: ' ...
               nomes{j}],'fontsize',15);

        estilo_eixos(gca);

        salvar_figura(fig,fullfile(saida, ...
            sprintf('G3_E07_Fig%02d_Temporal.png',j)));

    end

    % ============================================================
    % FIGURA 05 - VARIACAO TEMPORAL DO RMS B
    % ============================================================

    fig = nova_figura();

    hold on;

    hh = [];

    labs = {};

    for k = 1:numel(registros)

        ir = find(reg==registros(k));

        ipos = find(y(ir)==1,1);

        t0 = tempo(ir(ipos));

        tx = (tempo(ir)-t0)*1000;

        dt = diff(tempo(ir));

        dx = diff(S(ir,2))./dt;

        dx(abs(dt-T)>0.05*T) = NaN;

        cor = cores(mod(k-1,2)+1,:);

        hh(k) = plot(tx(2:end),dx,'-o', ...
                     'color',cor, ...
                     'linewidth',1.5, ...
                     'markersize',4);

        labs{k} = sprintf( ...
            'Registro %02d',registros(k));

    end

    legend(hh,labs,'location','northeast');

    xlabel('Tempo relativo (ms)');

    ylabel('Variacao do RMS (A/s)');

    title('G3/E07 - Taxa de variacao do RMS B', ...
          'fontsize',15);

    estilo_eixos(gca);

    salvar_figura(fig,fullfile(saida, ...
        'G3_E07_Fig05_Derivada.png'));

    % ============================================================
    % FIGURA 06 - F1 POR HISTORICO
    % ============================================================

    fig = nova_figura();

    bar(1:numel(Hs),met(:,5),0.55, ...
        'facecolor',azul);

    labs = arrayfun(@(h) sprintf('H%d',h), ...
                    Hs,'uniformoutput',false);

    set(gca, ...
        'xtick',1:numel(Hs), ...
        'xticklabel',labs);

    ylim([0 1.15]);

    for k = 1:numel(Hs)

        text(k,met(k,5)+0.025, ...
            sprintf('%.3f',met(k,5)), ...
            'horizontalalignment','center', ...
            'fontsize',12);

    end

    xlabel('Numero de janelas no historico');

    ylabel('F1-score na validacao');

    title('G3/E07 - Classificacao temporal', ...
          'fontsize',15);

    estilo_eixos(gca);

    salvar_figura(fig,fullfile(saida, ...
        'G3_E07_Fig06_F1_Historico.png'));

    % ============================================================
    % FIGURA 07 - PERSISTENCIA
    % ============================================================

    fig = nova_figura();

    plot(Ps,100*pers(:,1),'-o', ...
         'color',azul, ...
         'linewidth',1.8, ...
         'markersize',7);

    hold on;

    plot(Ps,100*pers(:,2),'-s', ...
         'color',laranja, ...
         'linewidth',1.8, ...
         'markersize',7);

    ylim([0 108]);

    xlim([0.7 max(Ps)+0.3]);

    set(gca,'xtick',Ps);

    xlabel('Janelas consecutivas exigidas');

    ylabel('Taxa (%)');

    legend({'Recall','Falsos positivos'}, ...
           'location','southwest');

    title('G3/E07 - Persistencia dos alarmes', ...
          'fontsize',15);

    estilo_eixos(gca);

    salvar_figura(fig,fullfile(saida, ...
        'G3_E07_Fig07_Persistencia.png'));

    % ============================================================
    % FIGURA 08 - ATRASO DE DETECCAO
    % ============================================================

    fig = nova_figura();

    plot(Ps,1000*pers(:,3),'-o', ...
         'color',verde, ...
         'linewidth',1.8, ...
         'markersize',7);

    set(gca,'xtick',Ps);

    xlim([0.7 max(Ps)+0.3]);

    xlabel('Janelas consecutivas exigidas');

    ylabel('Atraso relativo medio (ms)');

    title('G3/E07 - Custo temporal da persistencia', ...
          'fontsize',15);

    estilo_eixos(gca);

    salvar_figura(fig,fullfile(saida, ...
        'G3_E07_Fig08_Atraso.png'));

    % ============================================================
    % FIGURAS 09 E 10 - SAIDAS H1 E H5
    % ============================================================

    historicos_plot = [1 numel(Hs)];

    for jj = 1:numel(historicos_plot)

        k = historicos_plot(jj);

        fig = nova_figura();

        hold on;

        iv = find(valid & isfinite(probs(:,k)));

        ipre = iv(y(iv)==0);

        ipos = iv(y(iv)==1);

        % Indices locais das janelas de validacao.
        [~,xpre] = ismember(ipre,iv);

        [~,xpos] = ismember(ipos,iv);

        h1 = plot(xpre,probs(ipre,k),'o', ...
                  'color',azul, ...
                  'markersize',5, ...
                  'linewidth',1.2);

        h2 = plot(xpos,probs(ipos,k),'s', ...
                  'color',laranja, ...
                  'markersize',5, ...
                  'linewidth',1.2);

        h3 = plot([1 numel(iv)],[0.5 0.5], ...
                  '--k','linewidth',1.3);

        ylim([-0.08 1.08]);

        xlabel('Indice da janela de validacao');

        ylabel('Saida da regressao logistica');

        title(sprintf( ...
            'G3/E07 - Validacao temporal H%d',Hs(k)), ...
            'fontsize',15);

        legend([h1 h2 h3], ...
            {'Pre-evento','Pos-evento','Limiar'}, ...
            'location','southeast');

        estilo_eixos(gca);

        salvar_figura(fig,fullfile(saida, ...
            sprintf('G3_E07_Fig%02d_Saida_H%d.png', ...
                    8+jj,Hs(k))));

    end

end

% ================================================================
% 13. MOTOR GRAFICO - FLTK COM FALLBACK
% ================================================================

fprintf('\n--- GERACAO DE FIGURAS ---\n');

motores = {'fltk','gnuplot'};

disponiveis = available_graphics_toolkits();

fprintf('Motores disponiveis:\n');

disp(disponiveis);

sucesso = false;

motor_final = 'nenhum';

% Remover figuras de execucoes anteriores.
antigas = dir(fullfile(SAIDA,'G3_E07_Fig*.png'));

for j = 1:numel(antigas)

    delete(fullfile(SAIDA,antigas(j).name));

end

for k = 1:numel(motores)

    motor = motores{k};

    if ~any(strcmp(disponiveis,motor))

        fprintf('Motor %s indisponivel.\n',motor);

        continue;

    end

    try

        close all;

        graphics_toolkit(motor);

        fprintf('\nTentando motor: %s\n',motor);

        % Teste real de renderizacao e exportacao.

        fig = nova_figura();

        plot(1:10,(1:10).^2,'-o');

        title('Teste de exportacao');

        xlabel('Amostra');

        ylabel('Valor');

        teste_png = fullfile(SAIDA, ...
                              'teste_motor_grafico.png');

        salvar_figura(fig,teste_png);

        delete(teste_png);

        fprintf('Teste de exportacao aprovado.\n');

        % Geracao das dez figuras definitivas.

        gerar_figuras( ...
            S,reg,tempo,y,valid, ...
            HISTORICOS,met,probs, ...
            PERSISTENCIAS,pers,SAIDA,T_JANELA);

        figuras = dir(fullfile( ...
            SAIDA,'G3_E07_Fig*.png'));

        if numel(figuras)~=10

            error('Numero incorreto de figuras.');

        end

        sucesso = true;

        motor_final = motor;

        fprintf('Exportacao concluida com %s.\n', ...
                motor);

        break;

    catch err

        fprintf('\nFalha no motor %s:\n',motor);

        fprintf('%s\n',err.message);

        close all;

        % Limpar somente figuras parciais do E07.

        parciais = dir(fullfile( ...
            SAIDA,'G3_E07_Fig*.png'));

        for j = 1:numel(parciais)

            delete(fullfile(SAIDA,parciais(j).name));

        end

        fprintf('Tentando alternativa grafica...\n');

    end

end

if ~sucesso

    warning(['Nenhum motor grafico conseguiu ' ...
             'exportar. Os CSVs foram preservados.']);

end

% ================================================================
% 14. PROTOCOLO EXPERIMENTAL
% ================================================================

fid = fopen(fullfile(SAIDA, ...
    'G3_E07_Protocolo.txt'),'w');

assert(fid>=0);

fprintf(fid,'G3/E07 - ANALISE TEMPORAL\n\n');

fprintf(fid,'Janelas treino: %d\n',sum(treino));

fprintf(fid,'Janelas validacao: %d\n',sum(valid));

fprintf(fid,'Janelas teste reservadas: %d\n',n_teste);

fprintf(fid,'Duracao da janela: %.8f s\n',T_JANELA);

fprintf(fid,'Regularizacao L2: %.8f\n',LAMBDA);

fprintf(fid,'Limiar RMS B: %.8f A\n',LIMIAR_RMS_B);

fprintf(fid,'Tolerancia estabilizacao: %.4f\n', ...
        TOLERANCIA);

fprintf(fid,'Minimo de janelas estaveis: %d\n', ...
        MIN_ESTAVEIS);

fprintf(fid,'Motor grafico: %s\n',motor_final);

fprintf(fid,'\nHistoricos avaliados: ');

fprintf(fid,'%d ',HISTORICOS);

fprintf(fid,'\nPersistencias avaliadas: ');

fprintf(fid,'%d ',PERSISTENCIAS);

fprintf(fid,'\nRegistros de treinamento: ');

fprintf(fid,'%d ',REG_TREINO);

fprintf(fid,'\nRegistros de validacao: ');

fprintf(fid,'%d ',REG_VALID);

fprintf(fid,'\nRegistros reservados: ');

fprintf(fid,'%d ',REG_TESTE);

fprintf(fid,'\n');

fprintf(fid, ...
    ['\nNormalizacao exclusivamente pelo treinamento.\n' ...
     'Historicos exclusivamente retrospectivos.\n' ...
     'Comparacao dos historicos nas mesmas janelas.\n' ...
     'Sem concatenacao entre registros distintos.\n' ...
     'Sem concatenacao atraves de lacunas temporais.\n' ...
     'Estabilizacao estimada retrospectivamente.\n' ...
     'Teste reservado nao avaliado.\n']);

fclose(fid);

% ================================================================
% 15. SUMARIO FINAL
% ================================================================

figuras = dir(fullfile(SAIDA,'G3_E07_Fig*.png'));

fprintf('\n');
fprintf('========================================\n');
fprintf('G3/E07 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Janelas desenvolvimento: %d\n',N);

fprintf('Janelas treinamento: %d\n',sum(treino));

fprintf('Janelas validacao: %d\n',sum(valid));

fprintf('Janelas teste reservadas: %d\n',n_teste);

fprintf('Historicos avaliados: %d\n', ...
        numel(HISTORICOS));

fprintf('Persistencias avaliadas: %d\n', ...
        numel(PERSISTENCIAS));

fprintf('Janelas comuns de treino: %d\n', ...
        sum(treino(idx_comum)));

fprintf('Janelas comuns de validacao: %d\n', ...
        sum(valid(idx_comum)));

fprintf('Figuras geradas: %d\n',numel(figuras));

fprintf('Motor grafico: %s\n',motor_final);

fprintf('\nResultados em:\n%s\n',SAIDA);

fprintf('\nOBSERVACOES CIENTIFICAS:\n');

fprintf(['Historicos exclusivamente retrospectivos.\n' ...
         'Nenhum historico mistura registros distintos.\n' ...
         'Lacunas temporais auditadas.\n' ...
         'Estabilizacao estimada retrospectivamente.\n' ...
         'Atraso relativo a primeira janela pos-evento.\n' ...
         'Classificacao temporal nao implica ' ...
         'diagnostico fisico.\n' ...
         'Teste reservado nao avaliado.\n']);

fprintf('\n');
