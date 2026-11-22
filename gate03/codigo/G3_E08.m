
1;

% ============================================================
% GATE 3 - EXPERIMENTO 08
% EXPLICABILIDADE E ABLACAO DE CARACTERISTICAS
% Pesquisa de diagnostico de falhas em motores SRM
% GNU Octave 11.3
% ============================================================

clear;
clc;
close all;

% ============================================================
% CONFIGURACAO
% ============================================================

ROOT = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

ENTRADA = fullfile(ROOT, 'Gate2_Exp05', ...
    'G2_E05_Caracteristicas_Corrigido.csv');

MANIFESTO = fullfile(ROOT, 'Gate3_Exp02', ...
    'G3_E02_Manifesto.csv');

SAIDA = fullfile(ROOT, 'Gate3_Exp08');

N_PERM = 100;
SEED = 42;
LAMBDA = 0.01;
MAX_ITER = 200;
TOL = 1e-8;
LIMIAR = 0.5;
TRANS_MS = 200;

rng(SEED);

if ~exist(SAIDA, 'dir')
    mkdir(SAIDA);
end

% ============================================================
% FUNCOES AUXILIARES
% ============================================================

function [cab, dados] = ler_csv(arq)

    fid = fopen(arq, 'r');

    if fid < 0
        error('Nao foi possivel abrir: %s', arq);
    end

    unwind_protect

        linha = fgetl(fid);

        if ~ischar(linha)
            error('Arquivo vazio: %s', arq);
        end

        if numel(linha) >= 3
            if isequal(double(linha(1:3)), [239 187 191])
                linha = linha(4:end);
            end
        end

        cab = strsplit(strtrim(linha), ',', ...
                      'collapsedelimiters', false);

        cab = cellfun(@strtrim, cab, ...
                      'UniformOutput', false);

        nc = numel(cab);

        dados = cell(1500, nc);
        n = 0;
        numero_linha = 1;

        while true

            linha = fgetl(fid);

            if ~ischar(linha)
                break;
            end

            numero_linha = numero_linha + 1;

            if isempty(strtrim(linha))
                continue;
            end

            partes = strsplit(strtrim(linha), ',', ...
                              'collapsedelimiters', false);

            if numel(partes) ~= nc
                error(['CSV inconsistente: %s\n' ...
                       'Linha: %d | Esperado: %d | Encontrado: %d'], ...
                       arq, numero_linha, nc, numel(partes));
            end

            n = n + 1;
            dados(n,:) = partes;

        end

        dados = dados(1:n,:);

    unwind_protect_cleanup
        fclose(fid);
    end_unwind_protect

    fprintf('\nArquivo importado:\n%s\n', arq);
    fprintf('Linhas: %d | Colunas: %d\n', ...
            rows(dados), columns(dados));

end


function idx = coluna(cab, nome)

    idx = find(strcmp(cab, nome));

    if numel(idx) ~= 1
        error('Coluna ausente ou ambigua: %s', nome);
    end

end


function p = sigmoide(z)

    z = max(min(z, 40), -40);
    p = 1 ./ (1 + exp(-z));

end


function L = perda_log(y, p)

    y = y(:);
    p = p(:);

    p = min(max(p, 1e-12), 1 - 1e-12);

    L = -mean(y .* log(p) + ...
              (1-y) .* log(1-p));

end


function L = objetivo(A, y, w, lambda)

    p = sigmoide(A*w);

    L = perda_log(y, p) + ...
        0.5*lambda*sum(w(2:end).^2);

end


function modelo = treinar_logistica(X, y, ...
                                    lambda, max_iter, tol)

    y = y(:);

    assert(rows(X) == numel(y));
    assert(all(isfinite(X(:))));
    assert(numel(unique(y)) == 2);

    mu = mean(X,1);
    sd = std(X,0,1);

    ativos = sd > 1e-12;

    if ~any(ativos)
        error('Conjunto sem caracteristicas variaveis.');
    end

    sd(~ativos) = 1;

    Z = bsxfun(@minus, X(:,ativos), mu(ativos));
    Z = bsxfun(@rdivide, Z, sd(ativos));

    n = rows(Z);
    d = columns(Z);

    A = [ones(n,1), Z];

    w = zeros(d+1,1);

    R = diag([0; ones(d,1)]);

    convergiu = false;

    for it = 1:max_iter

        p = sigmoide(A*w);

        g = A'*(p-y)/n + lambda*R*w;

        v = max(p.*(1-p), 1e-10);

        H = A'*bsxfun(@times, A, v)/n ...
            + lambda*R;

        H = H + 1e-10*eye(d+1);

        passo = H\g;

        valor_antigo = objetivo(A,y,w,lambda);

        alpha = 1;

        aceito = false;

        for ls = 1:30

            candidato = w - alpha*passo;

            valor_novo = objetivo(A,y,candidato,lambda);

            if valor_novo <= valor_antigo + 1e-12
                aceito = true;
                break;
            end

            alpha = alpha/2;

        end

        if ~aceito
            break;
        end

        delta = norm(candidato-w,inf);

        w = candidato;

        if delta < tol
            convergiu = true;
            break;
        end

    end

    coef = zeros(columns(X),1);
    coef(ativos) = w(2:end);

    modelo.mu = mu;
    modelo.sd = sd;
    modelo.ativos = ativos;

    modelo.w = w;
    modelo.coef = coef;
    modelo.intercepto = w(1);

    modelo.iteracoes = it;
    modelo.convergiu = convergiu;

end


function p = prever(modelo, X)

    idx = modelo.ativos;

    Z = bsxfun(@minus, X(:,idx), modelo.mu(idx));
    Z = bsxfun(@rdivide, Z, modelo.sd(idx));

    A = [ones(rows(Z),1), Z];

    p = sigmoide(A*modelo.w);

end


function m = metricas(y,p,limiar)

    y = y(:);
    p = p(:);

    pred = p >= limiar;

    TP = sum(pred == 1 & y == 1);
    TN = sum(pred == 0 & y == 0);
    FP = sum(pred == 1 & y == 0);
    FN = sum(pred == 0 & y == 1);

    m.acc = (TP+TN)/max(numel(y),1);

    m.prec = TP/max(TP+FP,1);

    if TP+FN > 0
        m.recall = TP/(TP+FN);
    else
        m.recall = NaN;
    end

    if FP+TN > 0
        m.fpr = FP/(FP+TN);
    else
        m.fpr = NaN;
    end

    m.f1 = 2*TP/max(2*TP+FP+FN,1);

    m.logloss = perda_log(y,p);

    m.TP = TP;
    m.TN = TN;
    m.FP = FP;
    m.FN = FN;

end


function r = avaliar_subconjunto(nome, idx, ...
                                Xtr, ytr, Xv, yv, ...
                                trans, estavel, ...
                                lambda, max_iter, tol, limiar)

    idx = unique(idx, 'stable');

    mod = treinar_logistica(Xtr(:,idx), ytr, ...
                            lambda, max_iter, tol);

    ptr = prever(mod, Xtr(:,idx));
    pv = prever(mod, Xv(:,idx));

    mt = metricas(ytr,ptr,limiar);
    mv = metricas(yv,pv,limiar);

    r.nome = nome;
    r.idx = idx;
    r.n = numel(idx);
    r.ativos = sum(mod.ativos);

    r.treino = mt;
    r.val = mv;
    r.pv = pv;

    r.recall_trans = mean(pv(trans) >= limiar);
    r.recall_estavel = mean(pv(estavel) >= limiar);

    r.logloss_trans = perda_log(yv(trans),pv(trans));

    r.convergiu = mod.convergiu;

    r.iteracoes = mod.iteracoes;

end


function imprimir_resultado(r)

    fprintf(['%-35s | N=%3d | F1=%.4f | ' ...
             'Recall=%.4f | FPR=%.4f | ' ...
             'LL=%.6f | Trans=%.4f\n'], ...
             r.nome, r.n, r.val.f1, ...
             r.val.recall, r.val.fpr, ...
             r.val.logloss, r.recall_trans);

    if ~r.convergiu
        fprintf('  AVISO: convergencia nao confirmada.\n');
    end

end


function exportar_resultados(arq,R)

    fid = fopen(arq,'w');

    assert(fid >= 0);

    fprintf(fid, ...
       ['configuracao,n_features,n_ativas,' ...
        'acc_treino,f1_treino,logloss_treino,' ...
        'acc_val,precision_val,recall_val,fpr_val,' ...
        'f1_val,logloss_val,recall_transitorio,' ...
        'recall_tardio,logloss_transitorio,' ...
        'TP,TN,FP,FN,convergiu,iteracoes\n']);

    for i = 1:numel(R)

        r = R(i);

        fprintf(fid, ...
            ['%s,%d,%d,' ...
             '%.10f,%.10f,%.10f,' ...
             '%.10f,%.10f,%.10f,%.10f,' ...
             '%.10f,%.10f,%.10f,%.10f,%.10f,' ...
             '%d,%d,%d,%d,%d,%d\n'], ...
             r.nome,r.n,r.ativos, ...
             r.treino.acc,r.treino.f1,r.treino.logloss, ...
             r.val.acc,r.val.prec,r.val.recall, ...
             r.val.fpr,r.val.f1,r.val.logloss, ...
             r.recall_trans,r.recall_estavel, ...
             r.logloss_trans, ...
             r.val.TP,r.val.TN,r.val.FP,r.val.FN, ...
             r.convergiu,r.iteracoes);

    end

    fclose(fid);

end


function f = nova_figura()

    f = figure('visible','off', ...
               'color','w', ...
               'position',[100 100 1350 850]);

end


function salvar_figura(f,saida,nome)

    arq = fullfile(saida,nome);

    print(f,arq,'-dpng','-r220');

    info = dir(arq);

    if isempty(info) || info.bytes < 1000
        error('Falha na exportacao: %s', arq);
    end

    close(f);

    fprintf('Figura exportada: %s\n', nome);

end


function grafico_horizontal(val,rotulos,tit,xlab)

    val = val(:);

    n = numel(val);

    barh(1:n,val,0.7);

    set(gca, ...
        'ytick',1:n, ...
        'yticklabel',rotulos, ...
        'ydir','reverse', ...
        'fontsize',10);

    ylim([0.3 n+0.7]);

    grid on;
    box on;

    title(tit,'fontsize',15);
    xlabel(xlab,'fontsize',12);

end


% ============================================================
% INICIO DO EXPERIMENTO
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E08 - EXPLICABILIDADE E ABLACAO\n');
fprintf('========================================\n');

fprintf('\n--- IMPORTACAO ---\n');

[cab,D] = ler_csv(ENTRADA);
[cm,M] = ler_csv(MANIFESTO);

assert(rows(D)==1422);
assert(columns(D)==24);

assert(rows(M)==1422);
assert(columns(M)==5);

% ============================================================
% IDENTIFICACAO DAS COLUNAS
% ============================================================

ir = coluna(cab,'registro');
iy = coluna(cab,'regiao');
ij = coluna(cab,'janela');
it = coluna(cab,'t_inicio_s');
ifase = coluna(cab,'fase');

mr = coluna(cm,'registro');
my = coluna(cm,'regiao');
mf = coluna(cm,'fase');
mg = coluna(cm,'grupo');
ml = coluna(cm,'linha_csv');

reg = str2double(D(:,ir));
jan = str2double(D(:,ij));
tempo = str2double(D(:,it));

fase = D(:,ifase);
regiao = D(:,iy);
grupo = M(:,mg);

assert(all(reg == str2double(M(:,mr))));

assert(all(strcmp(regiao,M(:,my))));
assert(all(strcmp(fase,M(:,mf))));

assert(all(str2double(M(:,ml)) == ...
           (2:rows(D)+1)'));

assert(all(isfinite(reg)));
assert(all(isfinite(jan)));
assert(all(isfinite(tempo)));

assert(all(ismember(regiao,{'pre','pos'})));
assert(all(ismember(fase,{'A','B','C'})));

assert(all(ismember(grupo, ...
                    {'treino','validacao','teste'})));

% ============================================================
% AUDITORIA DAS PARTICOES
% ============================================================

rtr = unique(reg(strcmp(grupo,'treino')));
rv = unique(reg(strcmp(grupo,'validacao')));
rt = unique(reg(strcmp(grupo,'teste')));

assert(isequal(rtr(:)',[1 2 4 5 7 8 10 11]));
assert(isequal(rv(:)',[3 9]));
assert(isequal(rt(:)',[6 12]));

assert(isempty(intersect(rtr,rv)));
assert(isempty(intersect(rtr,rt)));
assert(isempty(intersect(rv,rt)));

assert(sum(strcmp(grupo,'treino')) == 948);
assert(sum(strcmp(grupo,'validacao')) == 237);
assert(sum(strcmp(grupo,'teste')) == 237);

fprintf('\n--- PARTICOES AUDITADAS ---\n');

fprintf('Treino: ');
fprintf('%02d ',rtr);
fprintf('\n');

fprintf('Validacao: ');
fprintf('%02d ',rv);
fprintf('\n');

fprintf('Teste reservado: ');
fprintf('%02d ',rt);
fprintf('\n');

% ============================================================
% DEFINICAO DAS CARACTERISTICAS
% ============================================================

nomes_base = {

    'rms_A'
    'media_A'
    'desvio_A'
    'pico_abs_A'
    'pico_a_pico_A'
    'fator_crista'
    'mae_derivada_A_s'
    'rms_derivada_A_s'
    'energia_A2_s'
    'potencia_0_5k'
    'potencia_5_20k'
    'potencia_20_50k'
    'potencia_50_100k'
    'centroide_Hz'

};

nomes_curtos = {

    'RMS'
    'Media'
    'Desvio'
    'Pico'
    'Pico-pico'
    'Crista'
    'MAE derivada'
    'RMS derivada'
    'Energia'
    'PSD 0-5k'
    'PSD 5-20k'
    'PSD 20-50k'
    'PSD 50-100k'
    'Centroide'

};

fase_lista = {'A','B','C'};

cols = zeros(1,16);

for j = 1:14
    cols(j) = coluna(cab,nomes_base{j});
end

cols(15) = coluna(cab,'rms_trifasico_A');
cols(16) = coluna(cab,'desequilibrio_rms_pct');

nomes = cell(44,1);
ids = cell(44,1);

for f = 1:3

    for j = 1:14

        k = (f-1)*14+j;

        nomes{k} = [fase_lista{f} ' - ' nomes_curtos{j}];

        ids{k} = sprintf('F%02d',k);

    end

end

nomes{43} = 'RMS trifasico';
nomes{44} = 'Desequilibrio';

ids{43} = 'F43';
ids{44} = 'F44';

% ============================================================
% RETIRADA DO TESTE RESERVADO
% ============================================================

dev = ~strcmp(grupo,'teste');

D = D(dev,:);

reg = reg(dev);
jan = jan(dev);
tempo = tempo(dev);

fase = fase(dev);
regiao = regiao(dev);
grupo = grupo(dev);

Yfase = double(strcmp(regiao,'pos'));

V = str2double(D(:,cols));

assert(all(isfinite(V(:))));

% ============================================================
% RECONSTRUCAO DAS JANELAS TRIFASICAS
% ============================================================

chaves = unique([reg,Yfase,jan],'rows');

N = rows(chaves);

assert(N == 395);

X = zeros(N,44);
Y = zeros(N,1);

R = zeros(N,1);
J = zeros(N,1);
T = zeros(N,1);

G = cell(N,1);

for k = 1:N

    rr = chaves(k,1);
    yy = chaves(k,2);
    jj = chaves(k,3);

    idx = find(reg==rr & Yfase==yy & jan==jj);

    assert(numel(idx)==3);

    for f = 1:3

        ii = idx(strcmp(fase(idx),fase_lista{f}));

        assert(numel(ii)==1);

        faixa = (f-1)*14+(1:14);

        X(k,faixa) = V(ii,1:14);

    end

    globais = V(idx,15:16);

    dif = bsxfun(@minus,globais,globais(1,:));

    assert(max(abs(dif(:))) < 1e-7);

    assert(max(abs(tempo(idx)-tempo(idx(1)))) < 1e-9);

    assert(all(strcmp(grupo(idx),grupo{idx(1)})));

    X(k,43:44) = globais(1,:);

    Y(k) = yy;
    R(k) = rr;
    J(k) = jj;

    T(k) = tempo(idx(1));
    G{k} = grupo{idx(1)};

end

[~,ordem] = sortrows([R,T],[1 2]);

X = X(ordem,:);
Y = Y(ordem);

R = R(ordem);
J = J(ordem);
T = T(ordem);

G = G(ordem);

idxtr = strcmp(G,'treino');
idxv = strcmp(G,'validacao');

Xtr = X(idxtr,:);
ytr = Y(idxtr);

Xv = X(idxv,:);
yv = Y(idxv);

Rv = R(idxv);
Jv = J(idxv);
Tv = T(idxv);

assert(rows(Xtr)==316);
assert(rows(Xv)==79);

assert(sum(yv==0)==21);
assert(sum(yv==1)==58);

fprintf('\n--- JANELAS RECONSTRUIDAS ---\n');

fprintf('Treinamento: %d\n',rows(Xtr));
fprintf('Validacao: %d\n',rows(Xv));
fprintf('Teste reservado: 79\n');

fprintf('Caracteristicas: %d\n',columns(X));

% ============================================================
% DELIMITACAO EXPLORATORIA DO TRANSITORIO
% ============================================================

Trel = NaN(rows(Xv),1);

for rr = rv(:)'

    ip = Rv==rr & yv==1;

    t0 = min(Tv(ip));

    Trel(ip) = 1000*(Tv(ip)-t0);

end

trans = yv==1 & Trel < TRANS_MS;
estavel = yv==1 & Trel >= TRANS_MS;

assert(any(trans));
assert(any(estavel));

fprintf('\n--- REGIMES TEMPORAIS ---\n');

fprintf('Pre-evento: %d\n',sum(yv==0));

fprintf('Transitorio inicial (<%d ms): %d\n', ...
        TRANS_MS,sum(trans));

fprintf('Segmento tardio (>=%d ms): %d\n', ...
        TRANS_MS,sum(estavel));

% ============================================================
% MODELO COMPLETO DE REFERENCIA
% ============================================================

fprintf('\n--- MODELO COMPLETO ---\n');

modelo = treinar_logistica(Xtr,ytr, ...
                           LAMBDA,MAX_ITER,TOL);

p_base = prever(modelo,Xv);

mbase = metricas(yv,p_base,LIMIAR);

fprintf('Convergiu: %d\n',modelo.convergiu);
fprintf('Iteracoes: %d\n',modelo.iteracoes);

fprintf('ACC: %.6f\n',mbase.acc);
fprintf('F1: %.6f\n',mbase.f1);
fprintf('Recall: %.6f\n',mbase.recall);
fprintf('FPR: %.6f\n',mbase.fpr);
fprintf('Log loss: %.10f\n',mbase.logloss);

if ~modelo.convergiu
    error('Modelo de referencia sem convergencia confirmada.');
end

% ============================================================
% E08.1 - IMPORTANCIA DOS COEFICIENTES
% ============================================================

fprintf('\n--- E08.1: COEFICIENTES ---\n');

coef = modelo.coef;

[~,rank_coef] = sort(abs(coef),'descend');

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Coeficientes.csv'),'w');

assert(fid>=0);

fprintf(fid, ...
    'ranking,id,caracteristica,coeficiente,coef_abs,ativa\n');

for i = 1:44

    j = rank_coef(i);

    fprintf(fid,'%d,%s,%s,%.12f,%.12f,%d\n', ...
        i,ids{j},nomes{j},coef(j),abs(coef(j)), ...
        modelo.ativos(j));

    if i <= 10

        fprintf('%2d | %s | %-22s | %+.6f\n', ...
              i,ids{j},nomes{j},coef(j));

    end

end

fclose(fid);

% ============================================================
% E08.2 - IMPORTANCIA POR PERMUTACAO
% ============================================================

fprintf('\n--- E08.2: PERMUTACAO INDIVIDUAL ---\n');

delta_f1 = zeros(44,N_PERM);
delta_ll = zeros(44,N_PERM);

delta_acc = zeros(44,N_PERM);

rng(SEED);

for j = 1:44

    for b = 1:N_PERM

        XP = Xv;

        % Permutacao restrita ao registro.
        % Pode misturar pre e pos-evento do mesmo registro.
        % Nao representa uma perturbacao fisica real.

        for rr = rv(:)'

            idx = find(Rv==rr);

            pp = idx(randperm(numel(idx)));

            XP(idx,j) = Xv(pp,j);

        end

        p = prever(modelo,XP);

        mp = metricas(yv,p,LIMIAR);

        delta_f1(j,b) = mbase.f1-mp.f1;

        delta_ll(j,b) = mp.logloss-mbase.logloss;

        delta_acc(j,b) = mbase.acc-mp.acc;

    end

    fprintf('%s | %-22s | dF1=%+.6f | dLL=%+.8f\n', ...
        ids{j},nomes{j}, ...
        mean(delta_f1(j,:)), ...
        mean(delta_ll(j,:)));

end

perm_f1 = mean(delta_f1,2);
perm_ll = mean(delta_ll,2);

perm_f1_sd = std(delta_f1,0,2);
perm_ll_sd = std(delta_ll,0,2);

[~,rank_perm] = sort(perm_ll,'descend');

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Permutacao.csv'),'w');

assert(fid>=0);

fprintf(fid, ...
    ['ranking,id,caracteristica,delta_f1_media,' ...
     'delta_f1_dp,delta_logloss_media,delta_logloss_dp,' ...
     'delta_acc_media,n_permutacoes\n']);

for i = 1:44

    j = rank_perm(i);

    fprintf(fid, ...
        '%d,%s,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%d\n', ...
        i,ids{j},nomes{j}, ...
        perm_f1(j),perm_f1_sd(j), ...
        perm_ll(j),perm_ll_sd(j), ...
        mean(delta_acc(j,:)),N_PERM);

end

fclose(fid);

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Permutacao_Repeticoes.csv'),'w');

assert(fid>=0);

fprintf(fid,'id,repeticao,delta_f1,delta_logloss,delta_acc\n');

for j = 1:44
    for b = 1:N_PERM

        fprintf(fid,'%s,%d,%.10f,%.10f,%.10f\n', ...
            ids{j},b,delta_f1(j,b), ...
            delta_ll(j,b),delta_acc(j,b));

    end
end

fclose(fid);

% ============================================================
% E08.3 - ABLACAO INDIVIDUAL
% ============================================================

fprintf('\n--- E08.3: ABLACAO INDIVIDUAL ---\n');

base = avaliar_subconjunto('Completo',1:44, ...
        Xtr,ytr,Xv,yv,trans,estavel, ...
        LAMBDA,MAX_ITER,TOL,LIMIAR);

AB = base;

ab_f1 = zeros(44,1);
ab_ll = zeros(44,1);

for j = 1:44

    idx = setdiff(1:44,j);

    nome = ['Sem_' ids{j}];

    r = avaliar_subconjunto(nome,idx, ...
            Xtr,ytr,Xv,yv,trans,estavel, ...
            LAMBDA,MAX_ITER,TOL,LIMIAR);

    AB(end+1) = r;

    ab_f1(j) = base.val.f1-r.val.f1;

    ab_ll(j) = r.val.logloss-base.val.logloss;

    imprimir_resultado(r);

end

exportar_resultados(fullfile(SAIDA, ...
    'G3_E08_Ablacao_Individual.csv'),AB);

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Ablacao_Impactos.csv'),'w');

assert(fid>=0);

fprintf(fid,'id,caracteristica,delta_f1,delta_logloss\n');

for j = 1:44

    fprintf(fid,'%s,%s,%.10f,%.10f\n', ...
        ids{j},nomes{j},ab_f1(j),ab_ll(j));

end

fclose(fid);

% ============================================================
% E08.4 - ABLACAO POR FASES
% ============================================================

fprintf('\n--- E08.4: ABLACAO POR FASES ---\n');

A = 1:14;
B = 15:28;
C = 29:42;

GLOBAL = 43:44;

nomes_fases = {

    'Completo'
    'Somente_A'
    'Somente_B'
    'Somente_C'
    'A_B'
    'A_C'
    'B_C'
    'ABC_sem_globais'
    'Somente_globais'
    'Somente_desequilibrio'
    'Somente_RMS_trifasico'

};

conj_fases = {

    1:44
    A
    B
    C
    [A B]
    [A C]
    [B C]
    1:42
    GLOBAL
    44
    43

};

RF = base;

for i = 1:numel(conj_fases)

    r = avaliar_subconjunto(nomes_fases{i}, ...
            conj_fases{i}, ...
            Xtr,ytr,Xv,yv,trans,estavel, ...
            LAMBDA,MAX_ITER,TOL,LIMIAR);

    RF(i) = r;

    imprimir_resultado(r);

end

exportar_resultados(fullfile(SAIDA, ...
    'G3_E08_Ablacao_Fases.csv'),RF);

% ============================================================
% E08.5 - ABLACAO POR DOMINIO
% ============================================================

fprintf('\n--- E08.5: ABLACAO POR DOMINIO ---\n');

AMP_base = [1 2 3 4 5 6 9];
DIN_base = [7 8];
ESP_base = [10 11 12 13 14];

AMP = [AMP_base,14+AMP_base,28+AMP_base];

DIN = [DIN_base,14+DIN_base,28+DIN_base];

ESP = [ESP_base,14+ESP_base,28+ESP_base];

assert(numel(AMP)==21);
assert(numel(DIN)==6);
assert(numel(ESP)==15);

assert(isequal(sort([AMP DIN ESP GLOBAL]),1:44));

nomes_dom = {

    'Completo'
    'Amplitude'
    'Dinamica'
    'Espectral'
    'Globais'
    'Sem_amplitude'
    'Sem_dinamica'
    'Sem_espectral'
    'Sem_globais'
    'Espectral_A_C'
    'Dinamica_A_C'
    'Amplitude_A_C'

};

conj_dom = {

    1:44
    AMP
    DIN
    ESP
    GLOBAL
    setdiff(1:44,AMP)
    setdiff(1:44,DIN)
    setdiff(1:44,ESP)
    setdiff(1:44,GLOBAL)
    intersect(ESP,[A C])
    intersect(DIN,[A C])
    intersect(AMP,[A C])

};

RD = base;

for i = 1:numel(conj_dom)

    r = avaliar_subconjunto(nomes_dom{i}, ...
            conj_dom{i}, ...
            Xtr,ytr,Xv,yv,trans,estavel, ...
            LAMBDA,MAX_ITER,TOL,LIMIAR);

    RD(i) = r;

    imprimir_resultado(r);

end

exportar_resultados(fullfile(SAIDA, ...
    'G3_E08_Ablacao_Dominios.csv'),RD);

% ============================================================
% E08.6 - REDUCAO PROGRESSIVA
% Ranking baseado exclusivamente no treinamento
% ============================================================

fprintf('\n--- E08.6: REDUCAO DE DIMENSIONALIDADE ---\n');

KS = [1 2 3 5 8 12 16 24 32 44];

RK = base;

for i = 1:numel(KS)

    k = KS(i);

    idx = rank_coef(1:k);

    nome = sprintf('K%d',k);

    r = avaliar_subconjunto(nome,idx, ...
            Xtr,ytr,Xv,yv,trans,estavel, ...
            LAMBDA,MAX_ITER,TOL,LIMIAR);

    RK(i) = r;

    imprimir_resultado(r);

end

exportar_resultados(fullfile(SAIDA, ...
    'G3_E08_Dimensionalidade.csv'),RK);

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Subconjuntos_K.csv'),'w');

assert(fid>=0);

fprintf(fid,'modelo,posicao,id,caracteristica\n');

for i = 1:numel(KS)

    for k = 1:KS(i)

        j = rank_coef(k);

        fprintf(fid,'K%d,%d,%s,%s\n', ...
            KS(i),k,ids{j},nomes{j});

    end

end

fclose(fid);

% ============================================================
% PREDICOES POR JANELA
% ============================================================

fprintf('\n--- EXPORTACAO DAS PREDICOES ---\n');

TODOS = [RF RD RK];

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Predicoes.csv'),'w');

assert(fid>=0);

fprintf(fid, ...
    ['familia,modelo,registro,janela,regiao,' ...
     'tempo_rel_ms,segmento,saida_logistica,predicao\n']);

familias = [repmat({'Fases'},1,numel(RF)), ...
            repmat({'Dominios'},1,numel(RD)), ...
            repmat({'Dimensionalidade'},1,numel(RK))];

for i = 1:numel(TODOS)

    r = TODOS(i);

    for k = 1:rows(Xv)

        if yv(k)==0

            rg = 'pre';
            segmento = 'pre';

        elseif trans(k)

            rg = 'pos';
            segmento = 'transitorio';

        else

            rg = 'pos';
            segmento = 'tardio';

        end

        fprintf(fid,'%s,%s,%d,%d,%s,%.6f,%s,%.12f,%d\n', ...
            familias{i},r.nome,Rv(k),Jv(k),rg,Trel(k), ...
            segmento,r.pv(k),r.pv(k)>=LIMIAR);

    end

end

fclose(fid);

% ============================================================
% LEGENDA DAS CARACTERISTICAS
% ============================================================

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Legenda.csv'),'w');

assert(fid>=0);

fprintf(fid,'id,caracteristica,dominio\n');

for j = 1:44

    if ismember(j,AMP)
        dom = 'amplitude';
    elseif ismember(j,DIN)
        dom = 'dinamica';
    elseif ismember(j,ESP)
        dom = 'espectral';
    else
        dom = 'global';
    end

    fprintf(fid,'%s,%s,%s\n',ids{j},nomes{j},dom);

end

fclose(fid);

% ============================================================
% GERACAO DAS FIGURAS
% ============================================================

fprintf('\n--- GERACAO DAS FIGURAS ---\n');

set(0,'defaultfigurevisible','off');

warning('off','Octave:gnuplot-graphics');

graphics_toolkit('gnuplot');

fprintf('Motor grafico: %s\n',graphics_toolkit());

% ------------------------------------------------------------
% FIGURA 01 - COEFICIENTES COMPLETOS
% ------------------------------------------------------------

f = nova_figura();

set(f,'position',[100 100 1450 1500]);

rotulos = cell(44,1);

for j = 1:44
    rotulos{j} = [ids{j} ' - ' nomes{j}];
end

grafico_horizontal( ...
    coef(rank_coef),rotulos(rank_coef), ...
    'G3/E08 - Coeficientes logisticos padronizados', ...
    'Coeficiente');

salvar_figura(f,SAIDA,'01_Coeficientes.png');

% ------------------------------------------------------------
% FIGURA 02 - PERMUTACAO F1
% ------------------------------------------------------------

f = nova_figura();

[~,ord] = sort(perm_f1,'descend');

idx = ord(1:15);

grafico_horizontal(perm_f1(idx),rotulos(idx), ...
    'G3/E08 - Permutacao: impacto sobre F1', ...
    'Reducao media do F1-score');

salvar_figura(f,SAIDA,'02_Permutacao_F1.png');

% ------------------------------------------------------------
% FIGURA 03 - PERMUTACAO LOG LOSS
% ------------------------------------------------------------

f = nova_figura();

idx = rank_perm(1:15);

grafico_horizontal(perm_ll(idx),rotulos(idx), ...
    'G3/E08 - Permutacao: impacto sobre log loss', ...
    'Aumento medio da log loss');

hold on;

for i = 1:numel(idx)

    j = idx(i);

    xx = [perm_ll(j)-perm_ll_sd(j), ...
          perm_ll(j)+perm_ll_sd(j)];

    plot(xx,[i i],'k-','linewidth',1.4);

    plot([xx(1) xx(1)],[i-.12 i+.12], ...
        'k-','linewidth',1.2);

    plot([xx(2) xx(2)],[i-.12 i+.12], ...
        'k-','linewidth',1.2);

end

hold off;

salvar_figura(f,SAIDA,'03_Permutacao_LogLoss.png');

% ------------------------------------------------------------
% FIGURA 04 - ABLACAO INDIVIDUAL
% ------------------------------------------------------------

f = nova_figura();

[~,ord] = sort(abs(ab_ll),'descend');

idx = ord(1:15);

grafico_horizontal(ab_ll(idx),rotulos(idx), ...
    'G3/E08 - Ablacao individual', ...
    'Variacao da log loss apos remocao e retreino');

salvar_figura(f,SAIDA,'04_Ablacao_Individual.png');

% ------------------------------------------------------------
% FIGURA 05 - FASES
% ------------------------------------------------------------

f = nova_figura();

vf = arrayfun(@(r) r.val.f1,RF);

labf = {

    'Completo'
    'Somente A'
    'Somente B'
    'Somente C'
    'A + B'
    'A + C'
    'B + C'
    'ABC sem globais'
    'Somente globais'
    'Desequilibrio'
    'RMS trifasico'

};

grafico_horizontal(vf,labf, ...
    'G3/E08 - Classificacao por subconjunto de fases', ...
    'F1-score na validacao');

xlim([0 1.12]);

hold on;

for i = 1:numel(vf)

    text(vf(i)+0.015,i,sprintf('%.3f',vf(i)), ...
        'fontsize',10);

end

hold off;

salvar_figura(f,SAIDA,'05_Ablacao_Fases.png');

% ------------------------------------------------------------
% FIGURA 06 - COMPARACAO SEM GLOBAIS
% ------------------------------------------------------------

f = nova_figura();

sel = [1 8 9 10 11 6];

val = arrayfun(@(r) r.val.logloss,RF(sel));

grafico_horizontal(val,labf(sel), ...
    'G3/E08 - Informacao global versus individual', ...
    'Log loss na validacao');

salvar_figura(f,SAIDA,'06_Informacao_Global.png');

% ------------------------------------------------------------
% FIGURA 07 - DOMINIOS
% ------------------------------------------------------------

f = nova_figura();

vd = arrayfun(@(r) r.val.f1,RD);

labd = {

    'Completo'
    'Amplitude'
    'Dinamica'
    'Espectral'
    'Globais'
    'Sem amplitude'
    'Sem dinamica'
    'Sem espectral'
    'Sem globais'
    'Espectral A + C'
    'Dinamica A + C'
    'Amplitude A + C'

};

grafico_horizontal(vd,labd, ...
    'G3/E08 - Ablacao por dominio das caracteristicas', ...
    'F1-score na validacao');

xlim([0 1.12]);

hold on;

for i = 1:numel(vd)

    text(vd(i)+0.015,i,sprintf('%.3f',vd(i)), ...
        'fontsize',10);

end

hold off;

salvar_figura(f,SAIDA,'07_Ablacao_Dominios.png');

% ------------------------------------------------------------
% FIGURA 08 - DIMENSIONALIDADE
% ------------------------------------------------------------

f = nova_figura();

vk = arrayfun(@(r) r.val.f1,RK);
llk = arrayfun(@(r) r.val.logloss,RK);

subplot(2,1,1);

plot(KS,vk,'o-','linewidth',2,'markersize',7);

ylim([0 1.08]);

grid on;

title('G3/E08 - Desempenho versus dimensionalidade');

xlabel('Numero de caracteristicas');
ylabel('F1-score');

set(gca,'xtick',KS);

subplot(2,1,2);

semilogy(KS,max(llk,1e-12), ...
        's-','linewidth',2,'markersize',7);

grid on;

xlabel('Numero de caracteristicas');
ylabel('Log loss');

set(gca,'xtick',KS);

salvar_figura(f,SAIDA,'08_Dimensionalidade.png');

% ------------------------------------------------------------
% FIGURA 09 - TRANSITORIO
% ------------------------------------------------------------

f = nova_figura();

sel = [1 2 3 4 6 8 9 10];

rec_trans = arrayfun(@(r) r.recall_trans,RF(sel));

rec_tardio = arrayfun(@(r) r.recall_estavel,RF(sel));

barh([rec_trans(:),rec_tardio(:)],'grouped');

set(gca, ...
    'ytick',1:numel(sel), ...
    'yticklabel',labf(sel), ...
    'ydir','reverse', ...
    'fontsize',11);

xlim([0 1.08]);

grid on;

legend({'Transitorio inicial','Segmento tardio'}, ...
       'location','southoutside');

xlabel('Recall');

title('G3/E08 - Desempenho durante o transitorio');

salvar_figura(f,SAIDA,'09_Transitorio.png');

% ------------------------------------------------------------
% FIGURA 10 - MATRIZ DE CONFUSAO A + C
% ------------------------------------------------------------

f = nova_figura();

rAC = RF(6);

CM = [rAC.val.TN,rAC.val.FP;
      rAC.val.FN,rAC.val.TP];

imagesc(CM);

colormap(flipud(gray(256)));

caxis([0 max(CM(:))+1]);

colorbar;

axis square;

set(gca, ...
    'xtick',[1 2], ...
    'ytick',[1 2], ...
    'xticklabel',{'Pre','Pos'}, ...
    'yticklabel',{'Pre','Pos'}, ...
    'fontsize',13);

xlabel('Classe prevista');
ylabel('Classe real');

title('G3/E08 - Matriz de confusao: fases A + C');

for i = 1:2
    for j = 1:2

        if CM(i,j)>max(CM(:))/2
            cor = 'w';
        else
            cor = 'k';
        end

        text(j,i,sprintf('%d',CM(i,j)), ...
             'horizontalalignment','center', ...
             'fontsize',24, ...
             'fontweight','bold', ...
             'color',cor);

    end
end

salvar_figura(f,SAIDA,'10_Confusao_A_C.png');

% ============================================================
% REGISTRO DO PROTOCOLO
% ============================================================

fid = fopen(fullfile(SAIDA, ...
    'G3_E08_Protocolo.txt'),'w');

assert(fid>=0);

fprintf(fid,'G3/E08 - EXPLICABILIDADE E ABLACAO\n\n');

fprintf(fid,'Octave: %s\n',version());
fprintf(fid,'Entrada: %s\n',ENTRADA);

fprintf(fid,'Seed: %d\n',SEED);

fprintf(fid,'Permutacoes por caracteristica: %d\n',N_PERM);

fprintf(fid,'Lambda: %.8f\n',LAMBDA);
fprintf(fid,'Limiar: %.4f\n',LIMIAR);

fprintf(fid,'Transitorio exploratorio: %d ms\n',TRANS_MS);

fprintf(fid,'Treinamento: 316 janelas\n');
fprintf(fid,'Validacao: 79 janelas\n');
fprintf(fid,'Teste reservado: 79 janelas\n');

fprintf(fid,'Teste reservado: registros 06 e 12\n');

fprintf(fid,'Janelas transitorias: %d\n',sum(trans));
fprintf(fid,'Janelas tardias: %d\n',sum(estavel));

fprintf(fid,'Normalizacao exclusivamente pelo treino.\n');

fprintf(fid,'Ranking K exclusivamente pelo treino.\n');

fprintf(fid,'Permutacoes restritas ao mesmo registro.\n');

fprintf(fid,'Permutacoes nao representam falhas fisicas.\n');

fprintf(fid,'Ablacoes incluem retreinamento integral.\n');

fprintf(fid,'Regressao logistica L2 com Newton amortecido.\n');

fprintf(fid,'Log loss avalia saidas nao calibradas.\n');

fprintf(fid,'Importancia nao implica causalidade fisica.\n');

fprintf(fid,'Segmento tardio nao comprova estabilizacao.\n');

fprintf(fid,'Nenhum arquivo MAT acessado.\n');

fprintf(fid,'Teste reservado nao avaliado.\n');

fclose(fid);

% ============================================================
% RESUMO FINAL
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E08 - RESUMO CIENTIFICO\n');
fprintf('========================================\n');

fprintf('\n--- TOP 5 COEFICIENTES ---\n');

for i = 1:5

    j = rank_coef(i);

    fprintf('%d | %s | %s | %+.8f\n', ...
        i,ids{j},nomes{j},coef(j));

end

fprintf('\n--- TOP 5 PERMUTACAO LOG LOSS ---\n');

for i = 1:5

    j = rank_perm(i);

    fprintf('%d | %s | %s | %+.8f\n', ...
        i,ids{j},nomes{j},perm_ll(j));

end

fprintf('\n--- MODELOS POR FASE ---\n');

for i = 1:numel(RF)
    imprimir_resultado(RF(i));
end

fprintf('\n--- MODELOS POR DOMINIO ---\n');

for i = 1:numel(RD)
    imprimir_resultado(RD(i));
end

fprintf('\n--- DIMENSIONALIDADE ---\n');

for i = 1:numel(RK)
    imprimir_resultado(RK(i));
end

fprintf('\n--- MATRIZ DE CONFUSAO A + C ---\n');

disp(CM);

fprintf('\n--- AUDITORIA DE CONVERGENCIA ---\n');

fprintf('Modelo completo: %d\n',modelo.convergiu);

fprintf('Ablacao individual: %d/%d\n', ...
    sum([AB.convergiu]),numel(AB));

fprintf('Modelos por fase: %d/%d\n', ...
    sum([RF.convergiu]),numel(RF));

fprintf('Modelos por dominio: %d/%d\n', ...
    sum([RD.convergiu]),numel(RD));

fprintf('Modelos K: %d/%d\n', ...
    sum([RK.convergiu]),numel(RK));

fprintf('\n========================================\n');
fprintf('G3/E08 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Treinamento: 316 janelas\n');
fprintf('Validacao: 79 janelas\n');
fprintf('Teste reservado: 79 janelas\n');

fprintf('Caracteristicas individuais: 44\n');

fprintf('Permutacoes executadas: %d\n',44*N_PERM);

fprintf('Ablacoes individuais: 44\n');

fprintf('Configuracoes por fase: %d\n',numel(RF));

fprintf('Configuracoes por dominio: %d\n',numel(RD));

fprintf('Dimensionalidades avaliadas: %d\n',numel(RK));

fprintf('Figuras geradas: 10\n');

fprintf('\nResultados em:\n%s\n',SAIDA);

fprintf('\nOBSERVACOES CIENTIFICAS:\n');

fprintf('Importancia estatistica nao implica causalidade.\n');

fprintf('Correlacao pode mascarar importancia individual.\n');

fprintf('Fases A e C excluem informacoes globais da fase B.\n');

fprintf('Potencias espectrais podem carregar amplitude.\n');

fprintf('Resultados de validacao sao exploratorios.\n');

fprintf('Teste reservado nao avaliado.\n');

fprintf('\n');
