
% ============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 05
% ANALISE NAO SUPERVISIONADA
%
% GNU Octave 11.3.0
%
% METODOS:
% 1. PCA
% 2. K-means (K=2,3,4)
% 3. Distancia de Mahalanobis
%
% SEGURANCA:
% - Nao acessa arquivos .mat
% - Nao utiliza registros de teste
% - Ajustes realizados somente no treino
% - Validacao apenas para avaliacao exploratoria
% - Sem classes fisicas de falha presumidas
% ============================================================

clear;
clc;
close all;

% ============================================================
% 1. CAMINHOS
% ============================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

arquivo_dados = fullfile(base, ...
    'Gate2_Exp05','G2_E05_Caracteristicas.csv');

arquivo_manifesto = fullfile(base, ...
    'Gate3_Exp02','G3_E02_Manifesto.csv');

saida = fullfile(base,'Gate3_Exp05');

if ~exist(saida,'dir')
    mkdir(saida);
end

graphics_toolkit('gnuplot');

rand('seed',42);
randn('seed',42);

fprintf('\n========================================\n');
fprintf('G3/E05 - ANALISE NAO SUPERVISIONADA\n');
fprintf('========================================\n');

% ============================================================
% 2. LEITURA DOS CSVs
% ============================================================

fid=fopen(arquivo_dados,'rt');

if fid<0
    error('CSV G2/E05 nao encontrado.');
end

cab=fgetl(fid);
cab=strrep(cab,char([239 187 191]),'');
nomes=strtrim(strsplit(cab,','));

C=textscan(fid,repmat('%s',1,numel(nomes)), ...
    'Delimiter',',','ReturnOnError',false);

fclose(fid);

N=numel(C{1});

for j=1:numel(C)
    if numel(C{j})~=N
        error('CSV inconsistente.');
    end
end

fid=fopen(arquivo_manifesto,'rt');

if fid<0
    error('Manifesto G3/E02 nao encontrado.');
end

cab=fgetl(fid);
cab=strrep(cab,char([239 187 191]),'');
nm=strtrim(strsplit(cab,','));

M=textscan(fid,repmat('%s',1,numel(nm)), ...
    'Delimiter',',','ReturnOnError',false);

fclose(fid);

if numel(M{1})~=N
    error('Manifesto e caracteristicas incompatíveis.');
end

% ============================================================
% 3. COLUNAS
% ============================================================

col=@(h,s) find(strcmp(h,s));

campos={'registro','regiao','fase','janela'};

ix=zeros(1,4);

for j=1:4
    z=col(nomes,campos{j});
    if numel(z)~=1
        error('Coluna ausente: %s',campos{j});
    end
    ix(j)=z;
end

campos_m={'linha_csv','registro','fase','regiao','grupo'};

im=zeros(1,5);

for j=1:5
    z=col(nm,campos_m{j});
    if numel(z)~=1
        error('Coluna invalida no manifesto: %s',campos_m{j});
    end
    im(j)=z;
end

registro=str2double(C{ix(1)});
regiao=C{ix(2)};
fase_txt=C{ix(3)};
janela=str2double(C{ix(4)});
grupo=M{im(5)};

if any(~isfinite(registro)) || any(~isfinite(janela))
    error('Identificadores numericos invalidos.');
end

if any(str2double(M{im(1)})~=(2:N+1)') || ...
   any(str2double(M{im(2)})~=registro)
    error('Manifesto fora de ordem.');
end

% ============================================================
% 4. CARACTERISTICAS NUMERICAS
% ============================================================

features={ ...
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

F=zeros(N,16);

for j=1:16

    z=col(nomes,features{j});

    if numel(z)~=1
        error('Caracteristica ausente: %s',features{j});
    end

    F(:,j)=str2double(C{z});
end

if any(~isfinite(F(:)))
    error('Caracteristicas nao finitas.');
end

% ============================================================
% 5. REGIAO E FASE
% ============================================================

y=NaN(N,1);
fase=NaN(N,1);

for i=1:N

    s=lower(strtrim(regiao{i}));
    p=upper(strtrim(fase_txt{i}));

    if strcmp(s,'pre')
        y(i)=0;
    elseif strcmp(s,'pos')
        y(i)=1;
    end

    if strcmp(p,'A')
        fase(i)=1;
    elseif strcmp(p,'B')
        fase(i)=2;
    elseif strcmp(p,'C')
        fase(i)=3;
    end

    if ~strcmpi(strtrim(fase_txt{i}), ...
                strtrim(M{im(3)}{i})) || ...
       ~strcmpi(strtrim(regiao{i}), ...
                strtrim(M{im(4)}{i}))
        error('Manifesto inconsistente.');
    end
end

if any(isnan(y)) || any(isnan(fase))
    error('Regiao ou fase invalida.');
end

treino=strcmp(grupo,'treino');
valid=strcmp(grupo,'validacao');
teste=strcmp(grupo,'teste');

if sum(treino)~=948 || sum(valid)~=237 || ...
   sum(teste)~=237
    error('Particao G3/E02 alterada.');
end

% ============================================================
% 6. AGREGACAO TRIFASICA
% ============================================================

% Cada janela possui:
% 14 caracteristicas por fase (42)
% + 2 caracteristicas globais (2)
%
% Total: 44 caracteristicas.
%
% Evita repetir as caracteristicas globais
% tres vezes como amostras independentes.

chaves=unique([registro y janela],'rows');

R=[];
W=[];
Y=[];
G={};
X=[];

n_teste=0;

for j=1:rows(chaves)

    idx=find( ...
        registro==chaves(j,1) & ...
        y==chaves(j,2) & ...
        janela==chaves(j,3));

    if numel(idx)~=3
        error('Janela trifasica incompleta.');
    end

    [f,ord]=sort(fase(idx));

    if ~isequal(f(:)',[1 2 3])
        error('Fases inconsistentes.');
    end

    idx=idx(ord);

    if numel(unique(grupo(idx)))~=1
        error('Grupo inconsistente.');
    end

    if strcmp(grupo{idx(1)},'teste')
        n_teste=n_teste+1;
        continue;
    end

    globais=F(idx,15:16);

    tol=1e-7*max(1,max(abs(globais(:))));

    if max(max(globais))-min(min(globais))>tol
        % As duas caracteristicas globais possuem
        % escalas diferentes. Verificar por coluna.
        for q=1:2
            a=globais(:,q);
            tq=1e-7*max(1,max(abs(a)));
            if max(a)-min(a)>tq
                error('Caracteristica global inconsistente.');
            end
        end
    end

    vetor=[F(idx(1),1:14), ...
           F(idx(2),1:14), ...
           F(idx(3),1:14), ...
           F(idx(1),15:16)];

    R(end+1,1)=registro(idx(1));
    W(end+1,1)=janela(idx(1));
    Y(end+1,1)=y(idx(1));
    G{end+1,1}=grupo{idx(1)};
    X(end+1,:)=vetor;
end

J=rows(X);

if J~=395 || n_teste~=79
    error('Contagem inesperada de janelas.');
end

fprintf('Janelas desenvolvimento: %d\n',J);
fprintf('Janelas teste reservadas: %d\n',n_teste);
fprintf('Caracteristicas por janela: %d\n',columns(X));

% ============================================================
% 7. NORMALIZACAO ROBUSTA
% ============================================================

idx_train=strcmp(G,'treino');
idx_valid=strcmp(G,'validacao');

mu=mean(X(idx_train,:),1);
sigma=std(X(idx_train,:),0,1);

% Remover caracteristicas constantes
ativas=isfinite(sigma) & sigma>1e-12;

X=X(:,ativas);
mu=mu(ativas);
sigma=sigma(ativas);

P=columns(X);

if P<2
    error('Numero insuficiente de caracteristicas.');
end

Z=(X-repmat(mu,J,1))./repmat(sigma,J,1);

fprintf('Caracteristicas ativas: %d\n',P);

% ============================================================
% 8. PCA - AJUSTE NO TREINO
% ============================================================

Ztrain=Z(idx_train,:);

% Centralizacao residual
centro=mean(Ztrain,1);

Ztrain=Ztrain-repmat(centro,rows(Ztrain),1);

[U,S,V]=svd(Ztrain,'econ');

sing=diag(S);

variancia=sing.^2;

explicada=100*variancia/sum(variancia);
acumulada=cumsum(explicada);

% Projetar desenvolvimento
Zc=Z-repmat(centro,J,1);
scores=Zc*V;

fprintf('\n--- PCA ---\n');
fprintf('PC1: %.2f%%\n',explicada(1));
fprintf('PC2: %.2f%%\n',explicada(2));
fprintf('PC1+PC2: %.2f%%\n',acumulada(2));

% ============================================================
% 9. EXPORTAR PCA
% ============================================================

fid=fopen(fullfile(saida,'G3_E05_PCA.csv'),'wt');

fprintf(fid,'registro,grupo,regiao,janela,PC1,PC2,PC3\n');

for j=1:J

    if Y(j)==0
        reg='pre';
    else
        reg='pos';
    end

    pc3=NaN;

    if columns(scores)>=3
        pc3=scores(j,3);
    end

    fprintf(fid,'%d,%s,%s,%d,%.12g,%.12g,%.12g\n', ...
        R(j),G{j},reg,W(j), ...
        scores(j,1),scores(j,2),pc3);
end

fclose(fid);

fid=fopen(fullfile(saida, ...
    'G3_E05_PCA_Variancia.csv'),'wt');

fprintf(fid,'componente,variancia_pct,acumulada_pct\n');

for j=1:numel(explicada)
    fprintf(fid,'%d,%.12g,%.12g\n', ...
        j,explicada(j),acumulada(j));
end

fclose(fid);

% ============================================================
% 10. K-MEANS
% ============================================================

% K-means implementado sem pacote statistics.
% Espaco: primeiras componentes principais
% que explicam pelo menos 95%% da variancia.
%
% Ajuste somente com treinamento.

ncomp=find(acumulada>=95,1);

if isempty(ncomp)
    ncomp=columns(scores);
end

ncomp=max(2,ncomp);

D=scores(:,1:ncomp);

fprintf('\nComponentes para clustering: %d\n',ncomp);

Ks=[2 3 4];
n_restarts=20;
max_iter=200;

clusters=zeros(J,numel(Ks));

inertias=zeros(numel(Ks),1);

for kk=1:numel(Ks)

    K=Ks(kk);

    melhor_inercia=Inf;
    melhor_centros=[];
    melhor_labels=[];

    Dt=D(idx_train,:);

    nt=rows(Dt);

    for rep=1:n_restarts

        % Inicializacao por amostras distintas
        perm=randperm(nt);
        centros=Dt(perm(1:K),:);

        labels=zeros(nt,1);

        for iter=1:max_iter

            dist=zeros(nt,K);

            for c=1:K
                dif=Dt-repmat(centros(c,:),nt,1);
                dist(:,c)=sum(dif.^2,2);
            end

            [~,novos]=min(dist,[],2);

            if iter>1 && all(novos==labels)
                break;
            end

            labels=novos;

            for c=1:K

                membros=Dt(labels==c,:);

                if isempty(membros)
                    centros(c,:)=Dt(randi(nt),:);
                else
                    centros(c,:)=mean(membros,1);
                end
            end
        end

        % Inercia final
        inertia=0;

        for c=1:K
            dif=Dt(labels==c,:)- ...
                repmat(centros(c,:),sum(labels==c),1);

            inertia=inertia+sum(dif(:).^2);
        end

        if inertia<melhor_inercia
            melhor_inercia=inertia;
            melhor_centros=centros;
            melhor_labels=labels;
        end
    end

    inertias(kk)=melhor_inercia;

    % Atribuicao de todas as janelas de desenvolvimento
    dist=zeros(J,K);

    for c=1:K
        dif=D-repmat(melhor_centros(c,:),J,1);
        dist(:,c)=sum(dif.^2,2);
    end

    [~,clusters(:,kk)]=min(dist,[],2);

    fprintf('K=%d | Inercia treino=%.6f\n', ...
        K,melhor_inercia);
end

% ============================================================
% 11. EXPORTAR CLUSTERS
% ============================================================

fid=fopen(fullfile(saida, ...
    'G3_E05_Clusters.csv'),'wt');

fprintf(fid, ...
    'registro,grupo,regiao,janela,K2,K3,K4\n');

for j=1:J

    if Y(j)==0
        reg='pre';
    else
        reg='pos';
    end

    fprintf(fid,'%d,%s,%s,%d,%d,%d,%d\n', ...
        R(j),G{j},reg,W(j), ...
        clusters(j,1),clusters(j,2),clusters(j,3));
end

fclose(fid);

% ============================================================
% 12. MAHALANOBIS
% ============================================================

% Referencia pre-evento apenas do treino.
%
% Utiliza subespaco PCA de dimensao reduzida
% para evitar matriz de covariancia singular.
%
% Distancia quadratica regularizada.

idx_pretrain=idx_train & Y==0;

nma=min([5,columns(scores),sum(idx_pretrain)-2]);

Q=scores(:,1:nma);

Qpre=Q(idx_pretrain,:);

mu_pre=mean(Qpre,1);

cov_pre=cov(Qpre);

if isscalar(cov_pre)
    cov_pre=reshape(cov_pre,1,1);
end

reg=1e-4*trace(cov_pre)/nma;

if ~isfinite(reg) || reg<=0
    reg=1e-4;
end

cov_reg=cov_pre+reg*eye(nma);

inv_cov=pinv(cov_reg);

mahal=zeros(J,1);

for j=1:J
    d=Q(j,:)-mu_pre;
    mahal(j)=sqrt(max(0,d*inv_cov*d'));
end

% Limiar empirico 99%% no pre-evento de treino.
% Nao pressupoe distribuicao qui-quadrado.

ref=sort(mahal(idx_pretrain));

pos_lim=max(1,ceil(0.99*numel(ref)));

limiar=ref(pos_lim);

anomalia=mahal>limiar;

fprintf('\n--- MAHALANOBIS ---\n');
fprintf('Dimensao: %d\n',nma);
fprintf('Limiar empirico: %.6f\n',limiar);

% ============================================================
% 13. EXPORTAR ANOMALIAS
% ============================================================

fid=fopen(fullfile(saida, ...
    'G3_E05_Anomalias.csv'),'wt');

fprintf(fid, ...
    'registro,grupo,regiao,janela,distancia,limiar,anomalia\n');

for j=1:J

    if Y(j)==0
        regiao_txt='pre';
    else
        regiao_txt='pos';
    end

    fprintf(fid,'%d,%s,%s,%d,%.12g,%.12g,%d\n', ...
        R(j),G{j},regiao_txt,W(j), ...
        mahal(j),limiar,anomalia(j));
end

fclose(fid);

% ============================================================
% 14. METRICAS DESCRITIVAS
% ============================================================

fprintf('\n--- TAXAS DE ANOMALIA ---\n');

for g=1:2

    if g==1
        idxg=idx_train;
        nome='TREINO';
    else
        idxg=idx_valid;
        nome='VALIDACAO';
    end

    for classe=0:1

        idx=idxg & Y==classe;

        taxa=mean(anomalia(idx));

        fprintf('%s | classe=%d | taxa=%.4f\n', ...
            nome,classe,taxa);
    end
end

% ============================================================
% 15. FIGURA 1 - PCA
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 650]);

idx0=idx_valid & Y==0;
idx1=idx_valid & Y==1;

plot(scores(idx0,1),scores(idx0,2),'bo', ...
    'markersize',6,'linewidth',1.2);

hold on;

plot(scores(idx1,1),scores(idx1,2),'rs', ...
    'markersize',6,'linewidth',1.2);

xlabel(sprintf('PC1 (%.1f%%)',explicada(1)));
ylabel(sprintf('PC2 (%.1f%%)',explicada(2)));

title('G3/E05 - PCA | Validacao');

legend({'Pre-evento','Pos-evento'}, ...
    'location','best');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E05_Fig01_PCA.png'),'-dpng','-r200');

close(fig);

% ============================================================
% 16. FIGURA 2 - VARIANCIA PCA
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 600]);

nplot=min(15,numel(explicada));

bar(1:nplot,explicada(1:nplot));

xlabel('Componente principal');
ylabel('Variancia explicada (%)');

title('G3/E05 - Variancia explicada pela PCA');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E05_Fig02_Variancia.png'),'-dpng','-r200');

close(fig);

% ============================================================
% 17. FIGURA 3 - K-MEANS
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 650]);

cores={'b','r','g','m'};

K=3;
kk=find(Ks==K);

hold on;

for c=1:K

    idx=idx_valid & clusters(:,kk)==c;

    plot(scores(idx,1),scores(idx,2), ...
        [cores{c} 'o'], ...
        'markersize',6,'linewidth',1.2);
end

xlabel('PC1');
ylabel('PC2');

title('G3/E05 - K-means K=3 | Validacao');

legend({'Cluster 1','Cluster 2','Cluster 3'}, ...
    'location','best');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E05_Fig03_Kmeans.png'),'-dpng','-r200');

close(fig);

% ============================================================
% 18. FIGURA 4 - INERCIA
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1000 550]);

plot(Ks,inertias,'-o', ...
    'linewidth',2,'markersize',8);

xlabel('Numero de clusters K');
ylabel('Inercia no treinamento');

title('G3/E05 - Inercia do K-means');

set(gca,'xtick',Ks);

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E05_Fig04_Inercia.png'),'-dpng','-r200');

close(fig);

% ============================================================
% 19. FIGURA 5 - MAHALANOBIS
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 600]);

plot(find(idx0),mahal(idx0),'bo', ...
    'markersize',6);

hold on;

plot(find(idx1),mahal(idx1),'rs', ...
    'markersize',6);

plot([1 J],[limiar limiar],'k--', ...
    'linewidth',1.5);

xlabel('Indice da janela de desenvolvimento');
ylabel('Distancia de Mahalanobis');

title('G3/E05 - Distancia de Mahalanobis');

legend({'Pre-evento','Pos-evento','Limiar'}, ...
    'location','best');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E05_Fig05_Mahalanobis.png'),'-dpng','-r200');

close(fig);

% ============================================================
% 20. PROTOCOLO
% ============================================================

fid=fopen(fullfile(saida, ...
    'G3_E05_Protocolo.txt'),'wt');

fprintf(fid,'G3/E05 - Analise nao supervisionada\n\n');

fprintf(fid,'PCA ajustada apenas no treinamento.\n');
fprintf(fid,'K-means ajustado apenas no treinamento.\n');
fprintf(fid,'Mahalanobis ajustado no pre-evento de treino.\n');

fprintf(fid,'\nDimensoes:\n');
fprintf(fid,'Janelas desenvolvimento: %d\n',J);
fprintf(fid,'Janelas teste reservadas: %d\n',n_teste);
fprintf(fid,'Caracteristicas ativas: %d\n',P);
fprintf(fid,'Componentes clustering: %d\n',ncomp);
fprintf(fid,'Componentes Mahalanobis: %d\n',nma);

fprintf(fid,'\nLimitacoes:\n');
fprintf(fid,'1. Pre/pos sao proxies temporais.\n');
fprintf(fid,'2. Sem classes fisicas de falha.\n');
fprintf(fid,'3. Registros podem compartilhar trajetorias.\n');
fprintf(fid,'4. Teste reservado.\n');
fprintf(fid,'5. Clusters nao sao tipos de falha.\n');
fprintf(fid,'6. Normalizacao ajustada apenas no treino.\n');
fprintf(fid,'7. Sem comprovacao de generalizacao fisica.\n');

fclose(fid);

fprintf('\n========================================\n');
fprintf('G3/E05 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Janelas desenvolvimento: %d\n',J);
fprintf('Janelas teste reservadas: %d\n',n_teste);
fprintf('PCA: %d componentes ativas\n',P);
fprintf('K-means: K=2,3,4\n');
fprintf('Mahalanobis: %d componentes\n',nma);
fprintf('Figuras geradas: 5\n');
fprintf('Motor: gnuplot\n');

fprintf('\nResultados em:\n%s\n',saida);
