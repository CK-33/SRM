
% ==============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 06
% APRENDIZADO SUPERVISIONADO
%
% GNU Octave 11.3.0
%
% Modelos:
%   M1 - Regressao logistica L2
%   M2 - LDA regularizada
%   M3 - Classificador RBF por centros
%
% Entrada:
%   G2_E05_Caracteristicas.csv
%   G3_E02_Manifesto.csv
%
% Saida:
%   Gate3_Exp06
%
% Regras:
%   - Nao abrir arquivos .mat
%   - Nao acessar dados brutos
%   - Nao utilizar o conjunto de teste
%   - Uma amostra por janela trifasica
%   - Ajustes somente no treinamento
%   - Validacao por registros separados
%   - Rotulos pre/pos sao proxies temporais
% ==============================================================

clear;
clc;
close all;

% ==============================================================
% 1. CONFIGURACAO
% ==============================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

dados = fullfile(base, ...
    'Gate2_Exp05','G2_E05_Caracteristicas.csv');

manifesto = fullfile(base, ...
    'Gate3_Exp02','G3_E02_Manifesto.csv');

saida = fullfile(base,'Gate3_Exp06');

if ~exist(saida,'dir')
    mkdir(saida);
end

graphics_toolkit('gnuplot');

rand('seed',42);
randn('seed',42);

fprintf('\n========================================\n');
fprintf('G3/E06 - APRENDIZADO SUPERVISIONADO\n');
fprintf('========================================\n');

% ==============================================================
% 2. LEITURA DOS CSVs
% ==============================================================

fid=fopen(dados,'rt');

if fid<0
    error('Arquivo de caracteristicas nao encontrado.');
end

cab=fgetl(fid);
cab=strrep(cab,char([239 187 191]),'');
nomes=strtrim(strsplit(cab,','));

C=textscan(fid,repmat('%s',1,numel(nomes)), ...
    'Delimiter',',','ReturnOnError',false);

fclose(fid);

N=numel(C{1});

fid=fopen(manifesto,'rt');

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
    error('Quantidade de linhas incompatível.');
end

% ==============================================================
% 3. IDENTIFICACAO DAS COLUNAS
% ==============================================================

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
        error('Manifesto invalido: %s',campos_m{j});
    end
    im(j)=z;
end

registro=str2double(C{ix(1)});
regiao=C{ix(2)};
fase_txt=C{ix(3)};
janela=str2double(C{ix(4)});
grupo=M{im(5)};

if any(~isfinite(registro)) || any(~isfinite(janela))
    error('Identificadores invalidos.');
end

if any(str2double(M{im(1)})~=(2:N+1)') || ...
   any(str2double(M{im(2)})~=registro)
    error('Manifesto fora de ordem.');
end

% ==============================================================
% 4. CARACTERISTICAS
% ==============================================================

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

% ==============================================================
% 5. ROTULOS E FASES
% ==============================================================

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
    error('Rotulos invalidos.');
end

treino=strcmp(grupo,'treino');
valid=strcmp(grupo,'validacao');
teste=strcmp(grupo,'teste');

if sum(treino)~=948 || sum(valid)~=237 || ...
   sum(teste)~=237
    error('Particao G3/E02 alterada.');
end

% ==============================================================
% 6. AGREGACAO TRIFASICA
% ==============================================================

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

    for q=15:16
        a=F(idx,q);
        tol=1e-7*max(1,max(abs(a)));

        if max(a)-min(a)>tol
            error('Caracteristica global inconsistente.');
        end
    end

    if strcmp(grupo{idx(1)},'teste')
        n_teste=n_teste+1;
        continue;
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
    error('Quantidade inesperada de janelas.');
end

idx_train=strcmp(G,'treino');
idx_valid=strcmp(G,'validacao');

if sum(idx_train)~=316 || sum(idx_valid)~=79
    error('Particao agregada incorreta.');
end

fprintf('Janelas treinamento: %d\n',sum(idx_train));
fprintf('Janelas validacao: %d\n',sum(idx_valid));
fprintf('Janelas teste reservadas: %d\n',n_teste);
fprintf('Caracteristicas por janela: %d\n',columns(X));

% ==============================================================
% 7. NORMALIZACAO
% ==============================================================

mu=mean(X(idx_train,:),1);
sigma=std(X(idx_train,:),0,1);

ativas=isfinite(sigma) & sigma>1e-12;

X=X(:,ativas);
mu=mu(ativas);
sigma=sigma(ativas);

Z=(X-repmat(mu,J,1))./repmat(sigma,J,1);

Ztr=Z(idx_train,:);
Zva=Z(idx_valid,:);

ytr=Y(idx_train);
yva=Y(idx_valid);

fprintf('Caracteristicas ativas: %d\n',columns(Z));

% ==============================================================
% 8. MODELO 1 - REGRESSAO LOGISTICA L2
% ==============================================================

% Otimizacao por gradiente descendente.
% Regularizacao aplicada apenas aos pesos.

fprintf('\n--- MODELO 1: LOGISTICA L2 ---\n');

lambda=0.1;
taxa=0.05;
max_iter=5000;

A=[ones(rows(Ztr),1),Ztr];
B=[ones(rows(Zva),1),Zva];

w=zeros(columns(A),1);

for iter=1:max_iter

    z=A*w;
    z=max(-40,min(40,z));

    p=1./(1+exp(-z));

    grad=A'*(p-ytr)/numel(ytr);

    reg=[0;w(2:end)];

    grad=grad+lambda*reg/numel(ytr);

    w=w-taxa*grad;
end

ptr=1./(1+exp(-max(-40,min(40,A*w))));
pva=1./(1+exp(-max(-40,min(40,B*w))));

pred1_tr=ptr>=0.5;
pred1_va=pva>=0.5;

% ==============================================================
% 9. MODELO 2 - LDA REGULARIZADA
% ==============================================================

fprintf('--- MODELO 2: LDA ---\n');

X0=Ztr(ytr==0,:);
X1=Ztr(ytr==1,:);

m0=mean(X0,1);
m1=mean(X1,1);

S0=cov(X0);
S1=cov(X1);

n0=rows(X0);
n1=rows(X1);

Sp=((n0-1)*S0+(n1-1)*S1)/(n0+n1-2);

d=columns(Ztr);

reg=0.1*trace(Sp)/d;

if reg<=0 || ~isfinite(reg)
    reg=0.1;
end

Sp=Sp+reg*eye(d);

v=Sp\(m1-m0)';

prior0=n0/(n0+n1);
prior1=n1/(n0+n1);

b=-0.5*(m1+m0)*v+log(prior1/prior0);

score_tr=Ztr*v+b;
score_va=Zva*v+b;

pred2_tr=score_tr>=0;
pred2_va=score_va>=0;

p2tr=1./(1+exp(-max(-40,min(40,score_tr))));
p2va=1./(1+exp(-max(-40,min(40,score_va))));

% ==============================================================
% 10. MODELO 3 - RBF POR CENTROS
% ==============================================================

% Classificador nao linear por similaridade.
% Centros: todas as amostras de treinamento.
%
% Distancia euclidiana no espaco padronizado.
% Kernel gaussiano e regressao ridge.
%
% Este modelo e uma referencia RBF, nao uma SVM.

fprintf('--- MODELO 3: RBF ---\n');

nt=rows(Ztr);
nv=rows(Zva);

% Distancias treino-treino
Dtr=zeros(nt,nt);

for i=1:nt
    dif=Ztr-repmat(Ztr(i,:),nt,1);
    Dtr(:,i)=sum(dif.^2,2);
end

distancias=sqrt(Dtr(Dtr>0));

if isempty(distancias)
    error('Distancias RBF degeneradas.');
end

largura=median(distancias);

gamma=1/(2*largura^2);

Ktr=exp(-gamma*Dtr);

% Distancias validacao-treino
Dva=zeros(nv,nt);

for i=1:nt
    dif=Zva-repmat(Ztr(i,:),nv,1);
    Dva(:,i)=sum(dif.^2,2);
end

Kva=exp(-gamma*Dva);

alpha=(Ktr+0.1*eye(nt))\(2*ytr-1);

s3tr=Ktr*alpha;
s3va=Kva*alpha;

pred3_tr=s3tr>=0;
pred3_va=s3va>=0;

p3tr=1./(1+exp(-max(-40,min(40,s3tr))));
p3va=1./(1+exp(-max(-40,min(40,s3va))));

% ==============================================================
% 11. METRICAS
% ==============================================================

pred_tr=[pred1_tr,pred2_tr,pred3_tr];
pred_va=[pred1_va,pred2_va,pred3_va];

prob_tr=[ptr,p2tr,p3tr];
prob_va=[pva,p2va,p3va];

nomes_modelos={'Logistica_L2','LDA','RBF'};

metricas=zeros(6,9);

linha=0;

fprintf('\n========================================\n');
fprintf('RESULTADOS\n');
fprintf('========================================\n');

for conjunto=1:2

    if conjunto==1
        yt=ytr;
        yp=pred_tr;
        nome='TREINO';
    else
        yt=yva;
        yp=pred_va;
        nome='VALIDACAO';
    end

    fprintf('\n%s\n',nome);

    for m=1:3

        linha=linha+1;

        TP=sum(yp(:,m)==1 & yt==1);
        TN=sum(yp(:,m)==0 & yt==0);
        FP=sum(yp(:,m)==1 & yt==0);
        FN=sum(yp(:,m)==0 & yt==1);

        acc=(TP+TN)/numel(yt);
        prec=TP/max(1,TP+FP);
        rec=TP/max(1,TP+FN);
        fpr=FP/max(1,FP+TN);
        f1=2*prec*rec/max(eps,prec+rec);

        metricas(linha,:)=[conjunto,m, ...
            acc,prec,rec,fpr,f1,TP,TN];

        fprintf('%-15s ACC=%.4f F1=%.4f ', ...
            nomes_modelos{m},acc,f1);

        fprintf('Recall=%.4f FPR=%.4f\n',rec,fpr);
    end
end

% ==============================================================
% 12. EXPORTAR METRICAS
% ==============================================================

fid=fopen(fullfile(saida, ...
    'G3_E06_Metricas.csv'),'wt');

fprintf(fid, ...
    'conjunto,modelo,accuracy,precision,recall,fpr,f1,TP,TN,FP,FN\n');

for conjunto=1:2

    if conjunto==1
        yt=ytr;
        yp=pred_tr;
        nome='treino';
    else
        yt=yva;
        yp=pred_va;
        nome='validacao';
    end

    for m=1:3

        TP=sum(yp(:,m)==1 & yt==1);
        TN=sum(yp(:,m)==0 & yt==0);
        FP=sum(yp(:,m)==1 & yt==0);
        FN=sum(yp(:,m)==0 & yt==1);

        acc=(TP+TN)/numel(yt);
        prec=TP/max(1,TP+FP);
        rec=TP/max(1,TP+FN);
        fpr=FP/max(1,FP+TN);
        f1=2*prec*rec/max(eps,prec+rec);

        fprintf(fid, ...
            '%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%d,%d\n', ...
            nome,nomes_modelos{m}, ...
            acc,prec,rec,fpr,f1,TP,TN,FP,FN);
    end
end

fclose(fid);

% ==============================================================
% 13. EXPORTAR PREDICOES
% ==============================================================

fid=fopen(fullfile(saida, ...
    'G3_E06_Predicoes.csv'),'wt');

fprintf(fid, ...
    ['registro,grupo,regiao,janela,' ...
     'logistica_prob,logistica_pred,' ...
     'lda_prob,lda_pred,' ...
     'rbf_score,rbf_pred\n']);

for j=1:J

    if Y(j)==0
        reg='pre';
    else
        reg='pos';
    end

    if idx_train(j)
        q=find(find(idx_train)==j);
        prob=prob_tr(q,:);
        pred=pred_tr(q,:);
        rbfscore=s3tr(q);
    else
        q=find(find(idx_valid)==j);
        prob=prob_va(q,:);
        pred=pred_va(q,:);
        rbfscore=s3va(q);
    end

    fprintf(fid, ...
        '%d,%s,%s,%d,%.12g,%d,%.12g,%d,%.12g,%d\n', ...
        R(j),G{j},reg,W(j), ...
        prob(1),pred(1), ...
        prob(2),pred(2), ...
        rbfscore,pred(3));
end

fclose(fid);

% ==============================================================
% 14. FIGURA 1 - F1 NA VALIDACAO
% ==============================================================

f1va=zeros(1,3);

for m=1:3

    TP=sum(pred_va(:,m)==1 & yva==1);
    FP=sum(pred_va(:,m)==1 & yva==0);
    FN=sum(pred_va(:,m)==0 & yva==1);

    f1va(m)=2*TP/max(1,2*TP+FP+FN);
end

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 600]);

bar(1:3,f1va,0.55);

set(gca,'xtick',1:3);
set(gca,'xticklabel', ...
    {'Logistica L2','LDA','RBF'});

ylim([0 1.08]);
xlim([0.4 3.6]);

ylabel('F1 na validacao');

title('G3/E06 - Desempenho dos classificadores');

grid on;
box off;

set(gca,'fontsize',13);
set(gcf,'paperpositionmode','auto');

for m=1:3
    text(m,f1va(m)+0.025, ...
        sprintf('%.3f',f1va(m)), ...
        'horizontalalignment','center', ...
        'fontsize',12);
end

drawnow();

print(fig,fullfile(saida, ...
    'G3_E06_Fig01_F1.png'),'-dpng','-r200');

close(fig);

% ==============================================================
% 15. FIGURA 2 - SCORES NA VALIDACAO
% ==============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1150 650]);

hold on;

plot(find(yva==0),pva(yva==0),'bo', ...
    'markersize',6,'linewidth',1.2);

plot(find(yva==1),pva(yva==1),'rs', ...
    'markersize',6,'linewidth',1.2);

plot([1 numel(yva)],[0.5 0.5],'k--', ...
    'linewidth',1.5);

xlabel('Indice da janela de validacao');
ylabel('Saida da regressao logistica');

title('G3/E06 - Regressao logistica | Validacao');

legend({'Pre-evento','Pos-evento','Limiar'}, ...
    'location','northeast');

ylim([-0.05 1.05]);

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E06_Fig02_Logistica.png'),'-dpng','-r200');

close(fig);

% ==============================================================
% 16. FIGURA 3 - LDA
% ==============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1150 650]);

hold on;

plot(find(yva==0),score_va(yva==0),'bo', ...
    'markersize',6,'linewidth',1.2);

plot(find(yva==1),score_va(yva==1),'rs', ...
    'markersize',6,'linewidth',1.2);

plot([1 numel(yva)],[0 0],'k--', ...
    'linewidth',1.5);

xlabel('Indice da janela de validacao');
ylabel('Score discriminante');

title('G3/E06 - LDA | Validacao');

legend({'Pre-evento','Pos-evento','Limiar'}, ...
    'location','northeast');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E06_Fig03_LDA.png'),'-dpng','-r200');

close(fig);

% ==============================================================
% 17. FIGURA 4 - RBF
% ==============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1150 650]);

hold on;

plot(find(yva==0),s3va(yva==0),'bo', ...
    'markersize',6,'linewidth',1.2);

plot(find(yva==1),s3va(yva==1),'rs', ...
    'markersize',6,'linewidth',1.2);

plot([1 numel(yva)],[0 0],'k--', ...
    'linewidth',1.5);

xlabel('Indice da janela de validacao');
ylabel('Score RBF');

title('G3/E06 - Classificador RBF | Validacao');

legend({'Pre-evento','Pos-evento','Limiar'}, ...
    'location','northeast');

grid on;
box off;

set(gca,'fontsize',12);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E06_Fig04_RBF.png'),'-dpng','-r200');

close(fig);

% ==============================================================
% 18. FIGURA 5 - MATRIZ DE CONFUSAO
% ==============================================================

% Regressao logistica na validacao

TP=sum(pred1_va==1 & yva==1);
TN=sum(pred1_va==0 & yva==0);
FP=sum(pred1_va==1 & yva==0);
FN=sum(pred1_va==0 & yva==1);

CM=[TN FP;FN TP];

fig=figure('visible','on','color','w', ...
    'position',[100 100 850 650]);

imagesc(CM);

colormap(flipud(gray(256)));
colorbar;

set(gca,'xtick',[1 2]);
set(gca,'ytick',[1 2]);

set(gca,'xticklabel',{'Pre','Pos'});
set(gca,'yticklabel',{'Pre','Pos'});

xlabel('Classe prevista');
ylabel('Classe observada');

title('G3/E06 - Matriz de confusao | Logistica');

for i=1:2
    for j=1:2

        if CM(i,j)>max(CM(:))/2
            cor='w';
        else
            cor='k';
        end

        text(j,i,sprintf('%d',CM(i,j)), ...
            'horizontalalignment','center', ...
            'fontsize',17, ...
            'color',cor);
    end
end

set(gca,'fontsize',13);
set(gcf,'paperpositionmode','auto');

drawnow();

print(fig,fullfile(saida, ...
    'G3_E06_Fig05_Confusao.png'),'-dpng','-r200');

close(fig);

% ==============================================================
% 19. DOCUMENTACAO
% ==============================================================

fid=fopen(fullfile(saida, ...
    'G3_E06_Protocolo.txt'),'wt');

fprintf(fid,'G3/E06 - Aprendizado supervisionado\n\n');

fprintf(fid,'Modelos:\n');
fprintf(fid,'1. Regressao logistica L2\n');
fprintf(fid,'2. LDA regularizada\n');
fprintf(fid,'3. Classificador RBF por centros\n');

fprintf(fid,'\nJanelas treinamento: %d\n',sum(idx_train));
fprintf(fid,'Janelas validacao: %d\n',sum(idx_valid));
fprintf(fid,'Janelas teste reservadas: %d\n',n_teste);

fprintf(fid,'\nCaracteristicas ativas: %d\n',columns(Z));

fprintf(fid,'\nHiperparametros:\n');
fprintf(fid,'Logistica lambda: %.6g\n',lambda);
fprintf(fid,'LDA regularizacao: %.6g\n',reg);
fprintf(fid,'RBF gamma: %.12g\n',gamma);

fprintf(fid,'\nLimitacoes:\n');
fprintf(fid,'Rotulos pre/pos sao proxies temporais.\n');
fprintf(fid,'Nao existem classes fisicas verificadas.\n');
fprintf(fid,'Sem referencia saudavel independente.\n');
fprintf(fid,'Registros podem compartilhar trajetorias.\n');
fprintf(fid,'Teste nao utilizado.\n');
fprintf(fid,'Nenhuma generalizacao fisica demonstrada.\n');

fclose(fid);

% ==============================================================
% 20. ENCERRAMENTO
% ==============================================================

fprintf('\n========================================\n');
fprintf('G3/E06 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Janelas treinamento: %d\n',sum(idx_train));
fprintf('Janelas validacao: %d\n',sum(idx_valid));
fprintf('Janelas teste reservadas: %d\n',n_teste);

fprintf('Modelos avaliados: 3\n');
fprintf('Figuras geradas: 5\n');
fprintf('Motor grafico: gnuplot\n');

fprintf('\nResultados em:\n%s\n',saida);

fprintf('\nOBSERVACAO CIENTIFICA:\n');
fprintf('Classificacao pre/pos nao equivale a diagnostico\n');
fprintf('de tipos fisicos de falha.\n');
fprintf('Teste reservado nao avaliado.\n');
