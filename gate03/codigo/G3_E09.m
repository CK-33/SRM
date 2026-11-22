
1;

% ============================================================
% GATE 3 - EXPERIMENTO 09
% AVALIACAO PRELIMINAR DE ROBUSTEZ
% Pesquisa de diagnostico de falhas em motores SRM
% GNU Octave 11.3
%
% IMPORTANTE:
% Perturbacoes no espaco das caracteristicas.
% Nao representa injecao de AWGN nas formas de onda.
% Nenhum arquivo MAT sera acessado.
% ============================================================

clear;
clc;
close all;

% ============================================================
% 1. CONFIGURACAO
% ============================================================

ROOT = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

ENTRADA = fullfile(ROOT,'Gate2_Exp05', ...
    'G2_E05_Caracteristicas_Corrigido.csv');

MANIFESTO = fullfile(ROOT,'Gate3_Exp02', ...
    'G3_E02_Manifesto.csv');

SAIDA = fullfile(ROOT,'Gate3_Exp09');

SEED = 42;
N_REP = 100;

LAMBDA = 0.01;
MAX_ITER = 200;
TOL = 1e-8;

LIMIAR = 0.5;
TRANS_MS = 200;

NIVEIS_RUIDO = [0 .01 .05 .10 .25 .50 1 2];

NIVEIS_GANHO = [0 .01 .02 .05 .10 .20 .30];

NIVEIS_PSD = [0 .01 .05 .10 .25 .50 1 2];

NIVEIS_PERDA = [0 .05 .10 .20 .30 .50 .75];

rng(SEED);

if ~exist(SAIDA,'dir')
    mkdir(SAIDA);
end

% ============================================================
% 2. FUNCOES AUXILIARES
% ============================================================

function [cab,D] = ler_csv(arq)

    fid = fopen(arq,'r');

    assert(fid>=0,['Erro ao abrir: ' arq]);

    unwind_protect

        linha = fgetl(fid);

        assert(ischar(linha));

        if numel(linha)>=3

            if isequal(double(linha(1:3)),[239 187 191])
                linha = linha(4:end);
            end

        end

        cab = strsplit(strtrim(linha),',', ...
                      'collapsedelimiters',false);

        cab = cellfun(@strtrim,cab, ...
                      'UniformOutput',false);

        nc = numel(cab);

        D = cell(1500,nc);

        n = 0;
        numero = 1;

        while true

            linha = fgetl(fid);

            if ~ischar(linha)
                break;
            end

            numero = numero+1;

            if isempty(strtrim(linha))
                continue;
            end

            p = strsplit(strtrim(linha),',', ...
                         'collapsedelimiters',false);

            if numel(p)~=nc

                error('CSV inconsistente: linha %d',numero);

            end

            n = n+1;

            D(n,:) = p;

        end

        D = D(1:n,:);

    unwind_protect_cleanup

        fclose(fid);

    end_unwind_protect

    fprintf('\nArquivo: %s\n',arq);
    fprintf('Linhas: %d | Colunas: %d\n', ...
            rows(D),columns(D));

end


function i = coluna(cab,nome)

    i = find(strcmp(cab,nome));

    assert(numel(i)==1, ...
           ['Coluna ausente ou ambigua: ' nome]);

end


function p = sigmoide(z)

    z = max(min(z,40),-40);

    p = 1./(1+exp(-z));

end


function L = logloss(y,p)

    y = y(:);
    p = p(:);

    p = min(max(p,1e-12),1-1e-12);

    L = -mean(y.*log(p)+(1-y).*log(1-p));

end


function L = objetivo(A,y,w,lambda)

    L = logloss(y,sigmoide(A*w)) ...
        + 0.5*lambda*sum(w(2:end).^2);

end


function mod = treinar_logistica(X,y,lambda,maxiter,tol)

    y = y(:);

    mu = mean(X,1);
    sd = std(X,0,1);

    ativos = sd>1e-12;

    assert(any(ativos));

    sd(~ativos) = 1;

    Z = bsxfun(@minus,X(:,ativos),mu(ativos));

    Z = bsxfun(@rdivide,Z,sd(ativos));

    n = rows(Z);
    d = columns(Z);

    A = [ones(n,1),Z];

    w = zeros(d+1,1);

    R = diag([0;ones(d,1)]);

    convergiu = false;

    for it = 1:maxiter

        p = sigmoide(A*w);

        g = A'*(p-y)/n+lambda*R*w;

        v = max(p.*(1-p),1e-10);

        H = A'*bsxfun(@times,A,v)/n ...
            +lambda*R+1e-10*eye(d+1);

        passo = H\g;

        ant = objetivo(A,y,w,lambda);

        alpha = 1;
        aceito = false;

        for ls = 1:30

            candidato = w-alpha*passo;

            novo = objetivo(A,y,candidato,lambda);

            if novo<=ant+1e-12

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

        if delta<tol

            convergiu = true;
            break;

        end

    end

    coef = zeros(columns(X),1);

    coef(ativos) = w(2:end);

    mod.mu = mu;
    mod.sd = sd;
    mod.ativos = ativos;
    mod.w = w;
    mod.coef = coef;

    mod.convergiu = convergiu;
    mod.iteracoes = it;

end


function p = prever(mod,X)

    a = mod.ativos;

    Z = bsxfun(@minus,X(:,a),mod.mu(a));

    Z = bsxfun(@rdivide,Z,mod.sd(a));

    p = sigmoide([ones(rows(Z),1),Z]*mod.w);

end


function m = metricas(y,p,trans,tardio,R,limiar)

    y = y(:);
    p = p(:);

    pred = p>=limiar;

    TP = sum(pred==1 & y==1);
    TN = sum(pred==0 & y==0);
    FP = sum(pred==1 & y==0);
    FN = sum(pred==0 & y==1);

    m.acc = (TP+TN)/numel(y);

    m.prec = TP/max(TP+FP,1);

    m.recall = TP/max(TP+FN,1);

    m.fpr = FP/max(FP+TN,1);

    m.f1 = 2*TP/max(2*TP+FP+FN,1);

    m.bacc = (m.recall+1-m.fpr)/2;

    m.ll = logloss(y,p);

    m.trans = mean(pred(trans));

    m.tardio = mean(pred(tardio));

    m.TP = TP;
    m.TN = TN;
    m.FP = FP;
    m.FN = FN;

    registros = unique(R);

    detectados = 0;
    falsos_eventos = 0;

    for rr = registros(:)'

        ip = R==rr & y==1;
        in = R==rr & y==0;

        detectados = detectados+any(pred(ip));

        falsos_eventos = falsos_eventos+any(pred(in));

    end

    m.eventos = detectados/numel(registros);

    m.reg_com_fp = falsos_eventos/numel(registros);

end


function p = prever_modelo(modelo,X)

    if strcmp(modelo.tipo,'logistica')

        p = prever(modelo.mod,X(:,modelo.idx));

    elseif strcmp(modelo.tipo,'deterministico')

        p = double(X(:,44)>modelo.limiar);

    else

        error('Tipo de modelo desconhecido.');

    end

end


function Z = avaliar_modelos(M,X,y,trans,tardio,R,limiar)

    % Colunas:
    % 1 F1
    % 2 Recall
    % 3 FPR
    % 4 Acuracia
    % 5 Acuracia balanceada
    % 6 Log loss
    % 7 Recall transitorio
    % 8 Recall tardio
    % 9 Eventos detectados
    % 10 Registros com falsos alarmes

    Z = zeros(numel(M),10);

    for k = 1:numel(M)

        p = prever_modelo(M(k),X);

        m = metricas(y,p,trans,tardio,R,limiar);

        Z(k,:) = [m.f1,m.recall,m.fpr,m.acc, ...
                  m.bacc,m.ll,m.trans,m.tardio, ...
                  m.eventos,m.reg_com_fp];

    end

end


function X = restringir_features(X)

    % Restricoes individuais elementares.
    % Nao garante consistencia fisica global.

    for f = 0:2

        % Desvio, pico, pico-pico, derivadas,
        % energia e potencias espectrais.

        idx = f*14+[3 4 5 7 8 9 10 11 12 13];

        X(:,idx) = max(X(:,idx),0);

        % Fator de crista.
        j = f*14+6;

        X(:,j) = max(X(:,j),1);

        % RMS.
        j = f*14+1;

        X(:,j) = max(X(:,j),0);

        % Centroide limitado pela frequencia de Nyquist.
        j = f*14+14;

        X(:,j) = min(max(X(:,j),0),100000);

    end

    X(:,43:44) = max(X(:,43:44),0);

end


function X = ganho_fases(X0,g)

    % g contem um ganho positivo por fase.
    %
    % Para um sinal x multiplicado por g:
    % RMS, media, desvio, pico e derivadas -> g
    % Energia e potencia espectral -> g^2
    % Fator de crista e centroide -> invariantes.
    %
    % Os indicadores globais sao reconstruidos
    % por formulas identificadas apenas no treino,
    % externamente a esta funcao.

    X = X0;

    for f = 1:3

        base = (f-1)*14;

        lineares = base+[1 2 3 4 5 7 8];

        quadraticas = base+[9 10 11 12 13];

        X(:,lineares) = X0(:,lineares)*g(f);

        X(:,quadraticas) = X0(:,quadraticas)*g(f)^2;

    end

end


function G = calcular_globais(X,tipo_rms,tipo_des)

    I = X(:,[1 15 29]);

    media = mean(I,2);

    maximo = max(I,[],2);
    minimo = min(I,[],2);

    den = max(media,1e-12);

    desv = max(abs(bsxfun(@minus,I,media)),[],2);

    candidatos_rms = [

        sqrt(mean(I.^2,2)), ...
        sqrt(sum(I.^2,2)), ...
        mean(I,2)

    ];

    candidatos_des = [

        100*desv./den, ...
        100*(maximo-minimo)./den, ...
        100*(maximo-minimo)./max(maximo,1e-12), ...
        100*std(I,1,2)./den, ...
        100*std(I,0,2)./den

    ];

    G = [candidatos_rms(:,tipo_rms), ...
         candidatos_des(:,tipo_des)];

end


function [tr,td,erro_rms,erro_des] = identificar_globais(X)

    erros_r = zeros(3,1);

    erros_d = zeros(5,1);

    for i = 1:3

        G = calcular_globais(X,i,1);

        erros_r(i) = max(abs(G(:,1)-X(:,43)));

    end

    for i = 1:5

        G = calcular_globais(X,1,i);

        erros_d(i) = max(abs(G(:,2)-X(:,44)));

    end

    [erro_rms,tr] = min(erros_r);

    [erro_des,td] = min(erros_d);

end


function X = perturbar_psd(X0,mu,nivel,Q)

    % Perturbacao exploratoria em potencias espectrais.
    %
    % A intensidade aumenta nas bandas superiores.
    % Nao equivale a AWGN fisico nem possui SNR em dB.
    %
    % Q fornece multiplicadores nao negativos
    % por janela, fase e banda.

    X = X0;

    pesos = [.10 .25 .50 1];

    for f = 1:3

        idx = (f-1)*14+(10:13);

        escala = max(abs(mu(idx)),1e-15);

        delta = bsxfun(@times,Q(:,idx), ...
                       pesos.*escala);

        X(:,idx) = X0(:,idx)+nivel*delta;

    end

    % Centroide, derivadas e energia sao mantidos.
    % Portanto, trata-se de uma perturbacao parcial
    % da entrada e nao de um novo sinal fisico.

end


function [X,disponivel] = omitir_fases(X0,fases,mu)

    X = X0;

    disponivel = true(1,44);

    for f = fases(:)'

        idx = (f-1)*14+(1:14);

        X(:,idx) = repmat(mu(idx),rows(X),1);

        disponivel(idx) = false;

    end

    if ~isempty(fases)

        % Qualquer fase ausente invalida os
        % indicadores trifasicos originais.

        X(:,43:44) = repmat(mu(43:44),rows(X),1);

        disponivel(43:44) = false;

    end

end


function q = quantil_simples(x,p)

    x = sort(x(:));

    n = numel(x);

    if n==1

        q = x(1);
        return;

    end

    h = 1+(n-1)*p;

    a = floor(h);
    b = ceil(h);

    q = x(a)+(h-a)*(x(b)-x(a));

end


function S = resumir(Z)

    % Z: modelos x metricas x repeticoes

    S.media = mean(Z,3);

    S.dp = std(Z,0,3);

    S.p05 = zeros(size(S.media));
    S.p95 = zeros(size(S.media));

    for i = 1:size(Z,1)

        for j = 1:size(Z,2)

            v = squeeze(Z(i,j,:));

            S.p05(i,j) = quantil_simples(v,.05);

            S.p95(i,j) = quantil_simples(v,.95);

        end

    end

end


function registrar_resumo(fid,exp,nivel,S,M,nrep)

    for k = 1:numel(M)

        fprintf(fid,'%s,%.8f,%s,%d', ...
                exp,nivel,M(k).nome,nrep);

        for j = 1:10

            fprintf(fid,',%.10f,%.10f,%.10f,%.10f', ...
                S.media(k,j),S.dp(k,j), ...
                S.p05(k,j),S.p95(k,j));

        end

        fprintf(fid,'\n');

    end

end


function registrar_repeticoes(fid,exp,nivel,Z,M)

    for b = 1:size(Z,3)

        for k = 1:numel(M)

            fprintf(fid,'%s,%.8f,%s,%d', ...
                    exp,nivel,M(k).nome,b);

            fprintf(fid,',%.10f',Z(k,:,b));

            fprintf(fid,'\n');

        end

    end

end


function imprimir_resumo(nivel,S,M)

    fprintf('\nNivel: %.4f\n',nivel);

    for k = 1:numel(M)

        fprintf(['%-20s | F1=%.4f | Rec=%.4f | ' ...
                 'FPR=%.4f | Trans=%.4f\n'], ...
                 M(k).nome,S.media(k,1), ...
                 S.media(k,2),S.media(k,3), ...
                 S.media(k,7));

    end

end


function f = nova_figura()

    f = figure('visible','off', ...
               'color','w', ...
               'position',[100 100 1400 850]);

end


function salvar(f,pasta,nome)

    arq = fullfile(pasta,nome);

    print(f,arq,'-dpng','-r220');

    d = dir(arq);

    assert(~isempty(d));

    assert(d.bytes>1000);

    close(f);

    fprintf('Figura exportada: %s\n',nome);

end


function curva(niveis,S,M,metrica,tit,xlab,ylab)

    cores = lines(numel(M));

    marcadores = {'o','s','^','d','v','>','<','p','h','x'};

    hold on;

    h = zeros(1,numel(M));

    for k = 1:numel(M)

        y = zeros(1,numel(niveis));

        for i = 1:numel(niveis)

            y(i) = S{i}.media(k,metrica);

        end

        h(k) = plot(niveis,y, ...
                   ['-' marcadores{k}], ...
                   'color',cores(k,:), ...
                   'linewidth',1.6, ...
                   'markersize',5);

    end

    hold off;
    grid on;
    box on;

    ylim([0 1.05]);

    xlabel(xlab);
    ylabel(ylab);

    title(tit);

    legend(h,{M.nome}, ...
           'location','eastoutside', ...
           'interpreter','none');

    set(gca,'fontsize',11);

end


% ============================================================
% 3. IMPORTACAO E AUDITORIA
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E09 - ROBUSTEZ PRELIMINAR\n');
fprintf('========================================\n');

[cab,D] = ler_csv(ENTRADA);

[cm,M] = ler_csv(MANIFESTO);

assert(rows(D)==1422 && columns(D)==24);

assert(rows(M)==1422 && columns(M)==5);

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

assert(all(reg==str2double(M(:,mr))));

assert(all(strcmp(regiao,M(:,my))));

assert(all(strcmp(fase,M(:,mf))));

assert(all(str2double(M(:,ml))==(2:1423)'));

assert(all(ismember(regiao,{'pre','pos'})));

assert(all(ismember(fase,{'A','B','C'})));

assert(all(ismember(grupo, ...
    {'treino','validacao','teste'})));

rtr = unique(reg(strcmp(grupo,'treino')));

rv = unique(reg(strcmp(grupo,'validacao')));

rt = unique(reg(strcmp(grupo,'teste')));

assert(isequal(rtr(:)',[1 2 4 5 7 8 10 11]));

assert(isequal(rv(:)',[3 9]));

assert(isequal(rt(:)',[6 12]));

assert(sum(strcmp(grupo,'treino'))==948);

assert(sum(strcmp(grupo,'validacao'))==237);

assert(sum(strcmp(grupo,'teste'))==237);

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
% 4. CARACTERISTICAS
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
    'rms_trifasico_A'
    'desequilibrio_rms_pct'

};

cols = zeros(1,16);

for j = 1:16

    cols(j) = coluna(cab,nomes_base{j});

end

% Excluir teste antes de converter features.

dev = ~strcmp(grupo,'teste');

D = D(dev,:);

reg = reg(dev);

jan = jan(dev);

tempo = tempo(dev);

fase = fase(dev);

regiao = regiao(dev);

grupo = grupo(dev);

V = str2double(D(:,cols));

assert(all(isfinite(V(:))));

Yfase = double(strcmp(regiao,'pos'));

chaves = unique([reg,Yfase,jan],'rows');

N = rows(chaves);

assert(N==395);

X = zeros(N,44);

Y = zeros(N,1);

R = zeros(N,1);

T = zeros(N,1);

J = zeros(N,1);

G = cell(N,1);

fases = {'A','B','C'};

for k = 1:N

    rr = chaves(k,1);

    yy = chaves(k,2);

    jj = chaves(k,3);

    idx = find(reg==rr & Yfase==yy & jan==jj);

    assert(numel(idx)==3);

    for f = 1:3

        ii = idx(strcmp(fase(idx),fases{f}));

        assert(numel(ii)==1);

        X(k,(f-1)*14+(1:14)) = V(ii,1:14);

    end

    glob = V(idx,15:16);

    dif = bsxfun(@minus,glob,glob(1,:));

    assert(max(abs(dif(:)))<1e-7);

    assert(max(abs(tempo(idx)-tempo(idx(1))))<1e-9);

    assert(all(strcmp(grupo(idx),grupo{idx(1)})));

    X(k,43:44) = glob(1,:);

    Y(k) = yy;

    R(k) = rr;

    T(k) = tempo(idx(1));

    J(k) = jj;

    G{k} = grupo{idx(1)};

end

[~,ord] = sortrows([R T],[1 2]);

X = X(ord,:);

Y = Y(ord);

R = R(ord);

T = T(ord);

J = J(ord);

G = G(ord);

itr = strcmp(G,'treino');

iv = strcmp(G,'validacao');

Xtr = X(itr,:);

ytr = Y(itr);

Xv = X(iv,:);

yv = Y(iv);

Rv = R(iv);

Tv = T(iv);

Jv = J(iv);

assert(rows(Xtr)==316);

assert(rows(Xv)==79);

assert(sum(yv==0)==21);

assert(sum(yv==1)==58);

mu = mean(Xtr,1);

sd = std(Xtr,0,1);

fprintf('\n--- JANELAS RECONSTRUIDAS ---\n');

fprintf('Treino: %d\n',rows(Xtr));

fprintf('Validacao: %d\n',rows(Xv));

fprintf('Teste reservado: 79\n');

% ============================================================
% 5. SEGMENTOS TEMPORAIS
% ============================================================

Trel = NaN(rows(Xv),1);

for rr = rv(:)'

    ip = Rv==rr & yv==1;

    t0 = min(Tv(ip));

    Trel(ip) = 1000*(Tv(ip)-t0);

end

trans = yv==1 & Trel<TRANS_MS;

tardio = yv==1 & Trel>=TRANS_MS;

assert(sum(trans)==32);

assert(sum(tardio)==26);

fprintf('\n--- SEGMENTOS ---\n');

fprintf('Pre-evento: %d\n',sum(yv==0));

fprintf('Transitorio: %d\n',sum(trans));

fprintf('Tardio: %d\n',sum(tardio));

% ============================================================
% 6. IDENTIFICACAO DAS FORMULAS GLOBAIS
% ============================================================

[tipo_rms,tipo_des,err_rms,err_des] = ...
    identificar_globais(Xtr);

fprintf('\n--- AUDITORIA DOS INDICADORES GLOBAIS ---\n');

fprintf('Formula RMS candidata: %d\n',tipo_rms);

fprintf('Erro maximo RMS: %.12g\n',err_rms);

fprintf('Formula desequilibrio candidata: %d\n',tipo_des);

fprintf('Erro maximo desequilibrio: %.12g\n',err_des);

% Nao prosseguir com ganho fisicamente consistente
% se a formula original nao puder ser reproduzida.

GLOBAL_OK = err_rms<1e-5 && err_des<1e-4;

if ~GLOBAL_OK

    fprintf(['AVISO: formulas globais nao reproduzidas.\n' ...
             'E09.3 sera omitido para evitar conclusoes invalidas.\n']);

end

% ============================================================
% 7. TREINAMENTO DOS MODELOS
% ============================================================

fprintf('\n--- TREINAMENTO DOS MODELOS ---\n');

A = 1:14;

B = 15:28;

C = 29:42;

ESP = [10:14,24:28,38:42];

DIN = [7:8,21:22,35:36];

base = treinar_logistica(Xtr,ytr, ...
                         LAMBDA,MAX_ITER,TOL);

assert(base.convergiu);

[~,ord_coef] = sort(abs(base.coef),'descend');

K8 = ord_coef(1:8)';

nomes_modelos = {

    'Completo'
    'Desequilibrio'
    'Globais'
    'K8'
    'Fase_B'
    'Fases_A_C'
    'Espectral_ABC'
    'Espectral_A_C'
    'Dinamica_ABC'
    'Deterministico'

};

subconjuntos = {

    1:44
    44
    43:44
    K8
    B
    [A C]
    ESP
    [10:14,38:42]
    DIN
    44

};

MODELOS = struct([]);

for k = 1:9

    MODELOS(k).nome = nomes_modelos{k};

    MODELOS(k).idx = subconjuntos{k};

    MODELOS(k).tipo = 'logistica';

    MODELOS(k).mod = treinar_logistica( ...
        Xtr(:,subconjuntos{k}),ytr, ...
        LAMBDA,MAX_ITER,TOL);

    MODELOS(k).limiar = LIMIAR;

    assert(MODELOS(k).mod.convergiu, ...
           ['Falha de convergencia: ' nomes_modelos{k}]);

    fprintf('%-20s | Features=%d | Iteracoes=%d\n', ...
        MODELOS(k).nome, ...
        numel(MODELOS(k).idx), ...
        MODELOS(k).mod.iteracoes);

end

% Detector univariado deterministico:
% limiar = maior desequilibrio pre-evento no treino
% mais uma pequena margem numerica.
% Nenhum dado pos-evento ou de validacao utilizado
% na determinacao deste limiar.

LIM_DET = max(Xtr(ytr==0,44))+1e-6;

MODELOS(10).nome = 'Deterministico';

MODELOS(10).idx = 44;

MODELOS(10).tipo = 'deterministico';

MODELOS(10).mod = [];

MODELOS(10).limiar = LIM_DET;

fprintf('Limiar deterministico: %.10f %%\n',LIM_DET);

NM = numel(MODELOS);

NV = rows(Xv);

% Exportar os modelos para reproducibilidade.

fid = fopen(fullfile(SAIDA, ...
    'G3_E09_Modelos.csv'),'w');

assert(fid>=0);

fprintf(fid,'modelo,tipo,feature_id,coef_padronizado,media_treino,dp_treino,intercepto,limiar\n');

for k = 1:NM

    if strcmp(MODELOS(k).tipo,'logistica')

        mod = MODELOS(k).mod;

        for j = 1:numel(MODELOS(k).idx)

            fprintf(fid,'%s,%s,F%02d,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
                MODELOS(k).nome,MODELOS(k).tipo, ...
                MODELOS(k).idx(j), ...
                mod.coef(j),mod.mu(j),mod.sd(j), ...
                mod.w(1),MODELOS(k).limiar);

        end

    else

        fprintf(fid,'%s,%s,F44,NaN,NaN,NaN,NaN,%.12g\n', ...
                MODELOS(k).nome,MODELOS(k).tipo,LIM_DET);

    end

end

fclose(fid);

% ============================================================
% 8. ARQUIVOS DE RESULTADOS
% ============================================================

metricas_nomes = {

    'f1'
    'recall'
    'fpr'
    'acc'
    'bacc'
    'logloss'
    'recall_transitorio'
    'recall_tardio'
    'eventos_detectados'
    'registros_com_fp'

};

fidR = fopen(fullfile(SAIDA, ...
    'G3_E09_Resumo.csv'),'w');

fidD = fopen(fullfile(SAIDA, ...
    'G3_E09_Repeticoes.csv'),'w');

assert(fidR>=0 && fidD>=0);

fprintf(fidR,'experimento,nivel,modelo,n_repeticoes');

fprintf(fidD,'experimento,nivel,modelo,repeticao');

for j = 1:10

    fprintf(fidR,',%s_media,%s_dp,%s_p05,%s_p95', ...
        metricas_nomes{j},metricas_nomes{j}, ...
        metricas_nomes{j},metricas_nomes{j});

    fprintf(fidD,',%s',metricas_nomes{j});

end

fprintf(fidR,'\n');

fprintf(fidD,'\n');

% ============================================================
% 9. E09.1 - REFERENCIA SEM PERTURBACOES
% ============================================================

fprintf('\n--- E09.1: REFERENCIA ---\n');

BASE = avaliar_modelos( ...
    MODELOS,Xv,yv,trans,tardio,Rv,LIMIAR);

SBASE = resumir(BASE);

registrar_resumo(fidR,'Referencia',0,SBASE,MODELOS,1);

registrar_repeticoes(fidD,'Referencia',0,BASE,MODELOS);

imprimir_resumo(0,SBASE,MODELOS);

% Auditoria de reproducao do E08.

esperado = [1 1 1 1 .9913 1 .9913 1 1];

if any(abs(BASE(1:9,1)'-esperado)>5e-4)

    warning(['Alguns modelos nao reproduziram o F1 do E08. ' ...
             'Investigar antes de comparar conclusoes.']);

end

% ============================================================
% 10. E09.2 - RUIDO ADITIVO NAS FEATURES
% ============================================================

fprintf('\n--- E09.2: RUIDO ADITIVO ---\n');

% X_perturbado = X + alpha * sigma_treino * N(0,1)
%
% Perturbacao independente entre entradas e janelas.
% Nao equivale a AWGN na corrente instantanea.
%
% Mesmas realizacoes aleatorias entre niveis e modelos.

SR = cell(1,numel(NIVEIS_RUIDO));

for i = 1:numel(NIVEIS_RUIDO)

    nivel = NIVEIS_RUIDO(i);

    Z = zeros(NM,10,N_REP);

    rng(SEED+100);

    for b = 1:N_REP

        E = randn(NV,44);

        if nivel==0

            XP = Xv;

        else

            XP = Xv+nivel*bsxfun(@times,E,sd);

            XP = restringir_features(XP);

        end

        Z(:,:,b) = avaliar_modelos( ...
            MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

    end

    SR{i} = resumir(Z);

    registrar_resumo(fidR,'Ruido_features',nivel, ...
                     SR{i},MODELOS,N_REP);

    registrar_repeticoes(fidD,'Ruido_features', ...
                          nivel,Z,MODELOS);

    imprimir_resumo(nivel,SR{i},MODELOS);

end

% ============================================================
% 11. E09.3 - ERROS SISTEMATICOS DE GANHO
% ============================================================

fprintf('\n--- E09.3: ERROS DE GANHO ---\n');

SG = cell(1,numel(NIVEIS_GANHO));

SC = cell(1,numel(NIVEIS_GANHO));

if GLOBAL_OK

    for i = 1:numel(NIVEIS_GANHO)

        nivel = NIVEIS_GANHO(i);

        Z = zeros(NM,10,N_REP);

        ZC = zeros(NM,10,2);

        rng(SEED+200);

        % Erros independentes entre fases.
        % Ganho constante durante cada registro.

        for b = 1:N_REP

            XP = Xv;

            for rr = rv(:)'

                idx = Rv==rr;

                g = 1+nivel*(2*rand(1,3)-1);

                XA = ganho_fases(Xv(idx,:),g);

                if nivel>0

                    XA(:,43:44) = calcular_globais( ...
                        XA,tipo_rms,tipo_des);

                end

                XP(idx,:) = XA;

            end

            Z(:,:,b) = avaliar_modelos( ...
                MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

        end

        SG{i} = resumir(Z);

        registrar_resumo(fidR,'Ganho_independente', ...
                         nivel,SG{i},MODELOS,N_REP);

        registrar_repeticoes(fidD,'Ganho_independente', ...
                              nivel,Z,MODELOS);

        imprimir_resumo(nivel,SG{i},MODELOS);

        % Ganho comum:
        % todas as fases aumentam ou diminuem juntas.

        sinais = [-1 1];

        for b = 1:2

            g = (1+sinais(b)*nivel)*ones(1,3);

            XP = ganho_fases(Xv,g);

            if nivel>0

                XP(:,43:44) = calcular_globais( ...
                    XP,tipo_rms,tipo_des);

            end

            ZC(:,:,b) = avaliar_modelos( ...
                MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

        end

        SC{i} = resumir(ZC);

        registrar_resumo(fidR,'Ganho_comum',nivel, ...
                         SC{i},MODELOS,2);

        registrar_repeticoes(fidD,'Ganho_comum', ...
                              nivel,ZC,MODELOS);

    end

else

    fprintf('E09.3 omitido por falha na auditoria global.\n');

end

% ============================================================
% 12. E09.4 - PERTURBACAO ESPECTRAL
% ============================================================

fprintf('\n--- E09.4: PERTURBACAO ESPECTRAL ---\n');

SP = cell(1,numel(NIVEIS_PSD));

% Referencia das potencias baseada exclusivamente
% no segmento pre-evento dos registros de treino.

mu_pre = mean(Xtr(ytr==0,:),1);

for i = 1:numel(NIVEIS_PSD)

    nivel = NIVEIS_PSD(i);

    Z = zeros(NM,10,N_REP);

    rng(SEED+300);

    for b = 1:N_REP

        Q = abs(randn(NV,44));

        XP = perturbar_psd(Xv,mu_pre,nivel,Q);

        Z(:,:,b) = avaliar_modelos( ...
            MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

    end

    SP{i} = resumir(Z);

    registrar_resumo(fidR,'Perturbacao_PSD',nivel, ...
                     SP{i},MODELOS,N_REP);

    registrar_repeticoes(fidD,'Perturbacao_PSD', ...
                          nivel,Z,MODELOS);

    imprimir_resumo(nivel,SP{i},MODELOS);

end

% ============================================================
% 13. E09.5 - INDISPONIBILIDADE DE FEATURES
% ============================================================

fprintf('\n--- E09.5: PERDA DE CARACTERISTICAS ---\n');

SM = cell(1,numel(NIVEIS_PERDA));

% Estrategia de fallback:
% substituir entrada indisponivel pela media
% de treinamento correspondente.
%
% Equivale a zero no espaco padronizado.
%
% Nenhum modelo recebe indicador de ausencia
% e nenhum modelo e retreinado.

MU = repmat(mu,NV,1);

for i = 1:numel(NIVEIS_PERDA)

    nivel = NIVEIS_PERDA(i);

    Z = zeros(NM,10,N_REP);

    rng(SEED+400);

    for b = 1:N_REP

        mascara = rand(NV,44)<nivel;

        XP = Xv;

        XP(mascara) = MU(mascara);

        Z(:,:,b) = avaliar_modelos( ...
            MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

    end

    SM{i} = resumir(Z);

    registrar_resumo(fidR,'Perda_features',nivel, ...
                     SM{i},MODELOS,N_REP);

    registrar_repeticoes(fidD,'Perda_features', ...
                          nivel,Z,MODELOS);

    imprimir_resumo(nivel,SM{i},MODELOS);

end

% ============================================================
% 14. E09.6 - INDISPONIBILIDADE DE FASES
% ============================================================

fprintf('\n--- E09.6: PERDA DE FASES ---\n');

cenarios = {

    []
    1
    2
    3
    [1 2]
    [1 3]
    [2 3]
    [1 2 3]

};

nomes_cenarios = {

    'Completo'
    'Sem_A'
    'Sem_B'
    'Sem_C'
    'Sem_A_B'
    'Sem_A_C'
    'Sem_B_C'
    'Sem_ABC'

};

NF = numel(cenarios);

SF = cell(1,NF);

F1_FASES = zeros(NM,NF);

FPR_FASES = zeros(NM,NF);

DISP = false(NM,NF);

fidF = fopen(fullfile(SAIDA, ...
    'G3_E09_Perda_Fases.csv'),'w');

assert(fidF>=0);

fprintf(fidF, ...
    ['cenario,modelo,n_features_indisponiveis,' ...
     'inputs_integralmente_disponiveis,' ...
     'f1,recall,fpr,bacc,recall_transitorio\n']);

for i = 1:NF

    [XP,disponivel] = omitir_fases(Xv,cenarios{i},mu);

    Z = avaliar_modelos( ...
        MODELOS,XP,yv,trans,tardio,Rv,LIMIAR);

    SF{i} = resumir(Z);

    F1_FASES(:,i) = Z(:,1);

    FPR_FASES(:,i) = Z(:,3);

    registrar_resumo(fidR,nomes_cenarios{i},i-1, ...
                     SF{i},MODELOS,1);

    registrar_repeticoes(fidD,nomes_cenarios{i}, ...
                          i-1,Z,MODELOS);

    fprintf('\nCenario: %s\n',nomes_cenarios{i});

    for k = 1:NM

        faltantes = sum(~disponivel(MODELOS(k).idx));

        DISP(k,i) = faltantes==0;

        fprintf(['%-20s | F1=%.4f | FPR=%.4f | ' ...
                 'Entradas ausentes=%d\n'], ...
                 MODELOS(k).nome,Z(k,1),Z(k,3),faltantes);

        fprintf(fidF,'%s,%s,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
                nomes_cenarios{i},MODELOS(k).nome, ...
                faltantes,DISP(k,i), ...
                Z(k,1),Z(k,2),Z(k,3),Z(k,5),Z(k,7));

    end

end

fclose(fidF);

fclose(fidR);

fclose(fidD);

% ============================================================
% 15. ANALISE DA DEGRADACAO
% ============================================================

fprintf('\n--- ANALISE DE DEGRADACAO ---\n');

fid = fopen(fullfile(SAIDA, ...
    'G3_E09_Degradacao.csv'),'w');

assert(fid>=0);

fprintf(fid, ...
    ['experimento,nivel,modelo,f1_base,f1_perturbado,' ...
     'delta_f1,fpr_base,fpr_perturbado,delta_fpr,' ...
     'recall_trans_base,recall_trans_perturbado\n']);

familias = {'Ruido_features','Perturbacao_PSD','Perda_features'};

niveis_all = {NIVEIS_RUIDO,NIVEIS_PSD,NIVEIS_PERDA};

resultados_all = {SR,SP,SM};

if GLOBAL_OK

    familias{end+1} = 'Ganho_independente';

    niveis_all{end+1} = NIVEIS_GANHO;

    resultados_all{end+1} = SG;

    familias{end+1} = 'Ganho_comum';

    niveis_all{end+1} = NIVEIS_GANHO;

    resultados_all{end+1} = SC;

end

for a = 1:numel(familias)

    niveis = niveis_all{a};

    dados = resultados_all{a};

    for i = 1:numel(niveis)

        V = dados{i}.media;

        for k = 1:NM

            fprintf(fid,'%s,%.8f,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
                familias{a},niveis(i),MODELOS(k).nome, ...
                BASE(k,1),V(k,1),BASE(k,1)-V(k,1), ...
                BASE(k,3),V(k,3),V(k,3)-BASE(k,3), ...
                BASE(k,7),V(k,7));

        end

    end

end

fclose(fid);

% ============================================================
% 16. CURVAS DE ROBUSTEZ INTEGRADAS
% ============================================================

% Area normalizada sob curva de desempenho.
% Comparacoes somente dentro de cada familia.
%
% Nao combinar familias com unidades e mecanismos
% de perturbacao diferentes em um ranking unico.

AREAS = zeros(NM,numel(familias),3);

fid = fopen(fullfile(SAIDA, ...
    'G3_E09_Areas_Normalizadas.csv'),'w');

assert(fid>=0);

fprintf(fid,'experimento,modelo,area_f1,area_bacc,area_recall_transitorio\n');

for a = 1:numel(familias)

    niveis = niveis_all{a};

    dados = resultados_all{a};

    normalizador = max(niveis)-min(niveis);

    for k = 1:NM

        f1 = zeros(size(niveis));

        ba = zeros(size(niveis));

        tr = zeros(size(niveis));

        for i = 1:numel(niveis)

            f1(i) = dados{i}.media(k,1);

            ba(i) = dados{i}.media(k,5);

            tr(i) = dados{i}.media(k,7);

        end

        AREAS(k,a,1) = trapz(niveis,f1)/normalizador;

        AREAS(k,a,2) = trapz(niveis,ba)/normalizador;

        AREAS(k,a,3) = trapz(niveis,tr)/normalizador;

        fprintf(fid,'%s,%s,%.10f,%.10f,%.10f\n', ...
            familias{a},MODELOS(k).nome, ...
            AREAS(k,a,1),AREAS(k,a,2),AREAS(k,a,3));

    end

end

fclose(fid);

% ============================================================
% 17. GERACAO DE FIGURAS
% ============================================================

fprintf('\n--- GERACAO DE FIGURAS ---\n');

warning('off','Octave:gnuplot-graphics');

graphics_toolkit('gnuplot');

set(0,'defaultfigurevisible','off');

NFIG = 0;

% ------------------------------------------------------------
% FIGURA 01 - REFERENCIA
% ------------------------------------------------------------

f = nova_figura();

barh(BASE(:,[1 2 5]),'grouped');

set(gca,'ytick',1:NM, ...
        'yticklabel',{MODELOS.nome}, ...
        'ydir','reverse', ...
        'fontsize',11, ...
        'ticklabelinterpreter','none');

xlim([0 1.08]);

grid on;

title('G3/E09 - Desempenho de referencia');

xlabel('Metrica');

legend({'F1','Recall','Acuracia balanceada'}, ...
       'location','southoutside');

salvar(f,SAIDA,'01_Referencia.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 02 - RUIDO: F1
% ------------------------------------------------------------

f = nova_figura();

curva(NIVEIS_RUIDO,SR,MODELOS,1, ...
    'G3/E09 - Ruido aditivo nas caracteristicas', ...
    'Intensidade (desvios-padrao de treino)', ...
    'F1-score medio');

salvar(f,SAIDA,'02_Ruido_F1.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 03 - RUIDO: FALSOS POSITIVOS
% ------------------------------------------------------------

f = nova_figura();

curva(NIVEIS_RUIDO,SR,MODELOS,3, ...
    'G3/E09 - Falsos positivos sob ruido', ...
    'Intensidade (desvios-padrao de treino)', ...
    'FPR medio');

salvar(f,SAIDA,'03_Ruido_FPR.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 04 - RUIDO: TRANSITORIO
% ------------------------------------------------------------

f = nova_figura();

curva(NIVEIS_RUIDO,SR,MODELOS,7, ...
    'G3/E09 - Robustez durante o transitorio', ...
    'Intensidade (desvios-padrao de treino)', ...
    'Recall transitorio medio');

salvar(f,SAIDA,'04_Ruido_Transitorio.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURAS 05 E 06 - GANHO
% ------------------------------------------------------------

if GLOBAL_OK

    f = nova_figura();

    curva(100*NIVEIS_GANHO,SG,MODELOS,5, ...
        'G3/E09 - Erros independentes de ganho', ...
        'Erro maximo de ganho por fase (%)', ...
        'Acuracia balanceada media');

    salvar(f,SAIDA,'05_Ganho_Independente.png');

    NFIG = NFIG+1;

    f = nova_figura();

    curva(100*NIVEIS_GANHO,SC,MODELOS,5, ...
        'G3/E09 - Variacao comum de ganho', ...
        'Magnitude do erro de ganho (%)', ...
        'Acuracia balanceada media');

    salvar(f,SAIDA,'06_Ganho_Comum.png');

    NFIG = NFIG+1;

end

% ------------------------------------------------------------
% FIGURA 07 - PERTURBACAO ESPECTRAL
% ------------------------------------------------------------

f = nova_figura();

curva(NIVEIS_PSD,SP,MODELOS,5, ...
    'G3/E09 - Perturbacao das potencias espectrais', ...
    'Multiplicador da perturbacao espectral', ...
    'Acuracia balanceada media');

salvar(f,SAIDA,'07_Perturbacao_Espectral.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 08 - PERDA DE FEATURES
% ------------------------------------------------------------

f = nova_figura();

curva(100*NIVEIS_PERDA,SM,MODELOS,5, ...
    'G3/E09 - Indisponibilidade de caracteristicas', ...
    'Probabilidade de perda por entrada (%)', ...
    'Acuracia balanceada media');

salvar(f,SAIDA,'08_Perdas_Features.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 09 - MAPA DE F1 SOB PERDA DE FASES
% ------------------------------------------------------------

f = nova_figura();

imagesc(F1_FASES);

colormap(jet(256));

caxis([0 1]);

colorbar;

set(gca, ...
    'xtick',1:NF, ...
    'xticklabel',{'ABC','-A','-B','-C', ...
                  '-AB','-AC','-BC','-ABC'}, ...
    'ytick',1:NM, ...
    'yticklabel',{MODELOS.nome}, ...
    'ticklabelinterpreter','none', ...
    'fontsize',10);

title('G3/E09 - F1 sob indisponibilidade de fases');

xlabel('Cenario de indisponibilidade');

for i = 1:NM

    for j = 1:NF

        if DISP(i,j)
            sufixo = '*';
        else
            sufixo = '';
        end

        if F1_FASES(i,j)<.25 || F1_FASES(i,j)>.8
            cor = 'w';
        else
            cor = 'k';
        end

        text(j,i,sprintf('%.2f%s',F1_FASES(i,j),sufixo), ...
            'horizontalalignment','center', ...
            'fontweight','bold', ...
            'fontsize',10, ...
            'color',cor);

    end

end

salvar(f,SAIDA,'09_Perda_Fases_F1.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 10 - FPR SOB PERDA DE FASES
% ------------------------------------------------------------

f = nova_figura();

imagesc(FPR_FASES);

colormap(jet(256));

caxis([0 1]);

colorbar;

set(gca, ...
    'xtick',1:NF, ...
    'xticklabel',{'ABC','-A','-B','-C', ...
                  '-AB','-AC','-BC','-ABC'}, ...
    'ytick',1:NM, ...
    'yticklabel',{MODELOS.nome}, ...
    'ticklabelinterpreter','none', ...
    'fontsize',10);

title('G3/E09 - Falsos positivos sob perda de fases');

xlabel('Cenario de indisponibilidade');

for i = 1:NM

    for j = 1:NF

        if FPR_FASES(i,j)<.25 || FPR_FASES(i,j)>.8
            cor = 'w';
        else
            cor = 'k';
        end

        text(j,i,sprintf('%.2f',FPR_FASES(i,j)), ...
            'horizontalalignment','center', ...
            'fontweight','bold', ...
            'fontsize',10, ...
            'color',cor);

    end

end

salvar(f,SAIDA,'10_Perda_Fases_FPR.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 11 - DISPERSAO MONTE CARLO
% ------------------------------------------------------------

f = nova_figura();

sel = [1 2 4 6 8 10];

cores = lines(numel(sel));

hold on;

hh = zeros(1,numel(sel));

for j = 1:numel(sel)

    k = sel(j);

    md = zeros(size(NIVEIS_RUIDO));

    lo = zeros(size(NIVEIS_RUIDO));

    hi = zeros(size(NIVEIS_RUIDO));

    for i = 1:numel(NIVEIS_RUIDO)

        md(i) = SR{i}.media(k,5);

        lo(i) = SR{i}.p05(k,5);

        hi(i) = SR{i}.p95(k,5);

    end

    hh(j) = plot(NIVEIS_RUIDO,md,'o-', ...
                 'color',cores(j,:), ...
                 'linewidth',2);

    plot(NIVEIS_RUIDO,lo,'--', ...
         'color',cores(j,:), ...
         'linewidth',.8);

    plot(NIVEIS_RUIDO,hi,'--', ...
         'color',cores(j,:), ...
         'linewidth',.8);

end

hold off;

ylim([0 1.05]);

grid on;

xlabel('Intensidade do ruido nas caracteristicas');

ylabel('Acuracia balanceada');

title('G3/E09 - Media e percentis 5 e 95');

legend(hh,{MODELOS(sel).nome}, ...
       'location','eastoutside', ...
       'interpreter','none');

salvar(f,SAIDA,'11_Dispersao_MonteCarlo.png');

NFIG = NFIG+1;

% ------------------------------------------------------------
% FIGURA 12 - AREAS NORMALIZADAS
% ------------------------------------------------------------

f = nova_figura();

imagesc(AREAS(:,:,2));

colormap(jet(256));

caxis([0 1]);

colorbar;

set(gca, ...
    'xtick',1:numel(familias), ...
    'xticklabel',familias, ...
    'ytick',1:NM, ...
    'yticklabel',{MODELOS.nome}, ...
    'ticklabelinterpreter','none', ...
    'fontsize',10);

xtickangle(25);

title('G3/E09 - Area normalizada: acuracia balanceada');

for i = 1:NM

    for j = 1:numel(familias)

        v = AREAS(i,j,2);

        if v<.25 || v>.8
            cor = 'w';
        else
            cor = 'k';
        end

        text(j,i,sprintf('%.3f',v), ...
             'horizontalalignment','center', ...
             'fontweight','bold', ...
             'color',cor);

    end

end

salvar(f,SAIDA,'12_Areas_Robustez.png');

NFIG = NFIG+1;

% ============================================================
% 18. PROTOCOLO CIENTIFICO
% ============================================================

fid = fopen(fullfile(SAIDA, ...
    'G3_E09_Protocolo.txt'),'w');

assert(fid>=0);

fprintf(fid,'G3/E09 - ROBUSTEZ PRELIMINAR\n\n');

fprintf(fid,'Octave: %s\n',version());

fprintf(fid,'Seed principal: %d\n',SEED);

fprintf(fid,'Repeticoes por nivel estocastico: %d\n',N_REP);

fprintf(fid,'Treinamento: 316 janelas\n');

fprintf(fid,'Validacao: 79 janelas\n');

fprintf(fid,'Teste reservado: 79 janelas\n');

fprintf(fid,'Registros reservados: 06 e 12\n');

fprintf(fid,'Modelos logisticos: 9\n');

fprintf(fid,'Detector deterministico: 1\n');

fprintf(fid,'Lambda: %.8f\n',LAMBDA);

fprintf(fid,'Limiar logistico: %.8f\n',LIMIAR);

fprintf(fid,'Limiar deterministico: %.10f\n',LIM_DET);

fprintf(fid,'Auditoria dos globais: %d\n',GLOBAL_OK);

fprintf(fid,'Formula candidata RMS: %d\n',tipo_rms);

fprintf(fid,'Formula candidata desequilibrio: %d\n',tipo_des);

fprintf(fid,'Erro maximo RMS: %.12g\n',err_rms);

fprintf(fid,'Erro maximo desequilibrio: %.12g\n',err_des);

fprintf(fid,'\nNiveis de ruido: ');

fprintf(fid,'%.4f ',NIVEIS_RUIDO);

fprintf(fid,'\nNiveis de ganho: ');

fprintf(fid,'%.4f ',NIVEIS_GANHO);

fprintf(fid,'\nNiveis de perturbacao PSD: ');

fprintf(fid,'%.4f ',NIVEIS_PSD);

fprintf(fid,'\nNiveis de perda: ');

fprintf(fid,'%.4f ',NIVEIS_PERDA);

fprintf(fid,'\n\nLIMITACOES:\n');

fprintf(fid,'Perturbacoes aplicadas no espaco das features.\n');

fprintf(fid,'Ruido nas features nao representa AWGN no sinal.\n');

fprintf(fid,'Niveis de ruido nao correspondem a SNR em dB.\n');

fprintf(fid,'Perturbacao PSD nao recalcula centroide ou energia.\n');

fprintf(fid,'Restricoes individuais nao garantem coerencia fisica.\n');

fprintf(fid,'Ganho usa formulas globais auditadas no treino.\n');

fprintf(fid,'Todos os modelos permanecem congelados.\n');

fprintf(fid,'Normalizacao e imputacao baseadas no treino.\n');

fprintf(fid,'Perda de fase invalida todas as features globais.\n');

fprintf(fid,'Asterisco na figura 09 indica entradas disponiveis.\n');

fprintf(fid,'Saidas com entradas ausentes sao testes de estresse.\n');

fprintf(fid,'Nao representam diagnostico validado para implantacao.\n');

fprintf(fid,'Percentis representam dispersao Monte Carlo.\n');

fprintf(fid,'Nao representam intervalos de confianca populacionais.\n');

fprintf(fid,'Repeticoes nao criam novos registros independentes.\n');

fprintf(fid,'Log loss do detector binario nao e comparavel.\n');

fprintf(fid,'Areas comparaveis apenas dentro da mesma familia.\n');

fprintf(fid,'Nenhum arquivo MAT acessado.\n');

fprintf(fid,'Teste reservado nao avaliado.\n');

fclose(fid);

% ============================================================
% 19. RESUMO FINAL
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E09 - RESUMO CIENTIFICO\n');
fprintf('========================================\n');

fprintf('\n--- REFERENCIA ---\n');

imprimir_resumo(0,SBASE,MODELOS);

fprintf('\n--- RUIDO MODERADO: 0.25 SIGMA ---\n');

i = find(NIVEIS_RUIDO==.25);

imprimir_resumo(.25,SR{i},MODELOS);

fprintf('\n--- RUIDO SEVERO: 1 SIGMA ---\n');

i = find(NIVEIS_RUIDO==1);

imprimir_resumo(1,SR{i},MODELOS);

fprintf('\n--- GANHO INDEPENDENTE: 10%% ---\n');

if GLOBAL_OK

    i = find(NIVEIS_GANHO==.10);

    imprimir_resumo(.10,SG{i},MODELOS);

else

    fprintf('Subexperimento omitido.\n');

end

fprintf('\n--- PERTURBACAO ESPECTRAL: NIVEL 1 ---\n');

i = find(NIVEIS_PSD==1);

imprimir_resumo(1,SP{i},MODELOS);

fprintf('\n--- PERDA DE 30%% DAS FEATURES ---\n');

i = find(NIVEIS_PERDA==.30);

imprimir_resumo(.30,SM{i},MODELOS);

fprintf('\n--- PERDA DA FASE B ---\n');

imprimir_resumo(2,SF{3},MODELOS);

fprintf('\n--- AREAS NORMALIZADAS: BACC ---\n');

for a = 1:numel(familias)

    fprintf('\nFamilia: %s\n',familias{a});

    [~,ord] = sort(AREAS(:,a,2),'descend');

    for j = 1:NM

        k = ord(j);

        fprintf('%2d | %-20s | %.6f\n', ...
                j,MODELOS(k).nome,AREAS(k,a,2));

    end

end

fprintf('\n========================================\n');
fprintf('G3/E09 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Modelos avaliados: %d\n',NM);

fprintf('Treinamento: 316 janelas\n');

fprintf('Validacao: 79 janelas\n');

fprintf('Teste reservado: 79 janelas\n');

fprintf('Repeticoes por nivel estocastico: %d\n',N_REP);

fprintf('Cenarios de perda de fase: %d\n',NF);

fprintf('Auditoria dos globais: %d\n',GLOBAL_OK);

fprintf('Figuras geradas: %d\n',NFIG);

fprintf('Motor grafico: %s\n',graphics_toolkit());

fprintf('\nResultados em:\n%s\n',SAIDA);

fprintf('\nOBSERVACOES CIENTIFICAS:\n');

fprintf('Modelos treinados sem perturbacoes.\n');

fprintf('Nenhum retreinamento durante testes de estresse.\n');

fprintf('Perturbacoes nas caracteristicas, nao nos sinais.\n');

fprintf('Repeticoes Monte Carlo nao sao novos registros.\n');

fprintf('Dados de teste reservados nao avaliados.\n');

fprintf('\n');
