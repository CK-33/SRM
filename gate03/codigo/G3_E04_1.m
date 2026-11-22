
% ============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 04.1
% Auditoria de normalizacao e sensibilidade
%
% GNU Octave 11.3.0
% Toolkit grafico: GNUPLOT
%
% ENTRADAS:
% Gate2_Exp05/G2_E05_Caracteristicas.csv
% Gate3_Exp02/G3_E02_Manifesto.csv
%
% SAIDA:
% Gate3_Exp04.1/
%
% SEGURANCA:
% - Nao acessa arquivos MAT
% - Nao acessa registros de teste para analise
% - Nao modifica experimentos anteriores
% - Nao recalibra parametros pela validacao
% - Nao cria observacoes fisicas independentes
% ============================================================

clear;
clc;
close all;

% ============================================================
% 1. CONFIGURACAO
% ============================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

arquivo_dados = fullfile(base, ...
    'Gate2_Exp05','G2_E05_Caracteristicas.csv');

arquivo_manifesto = fullfile(base, ...
    'Gate3_Exp02','G3_E02_Manifesto.csv');

saida = fullfile(base,'Gate3_Exp04.1');

if ~exist(saida,'dir')
    mkdir(saida);
end

% Parametros originais do G3/E04
k_cusum = 0.5;
h_cusum = 5.0;

lambda = 0.20;
L_ewma = 3.0;

% Transicoes artificiais em unidades de sigma.
% Nao representam falhas fisicas.
amplitudes = [0.25 0.50 1.00 2.00 3.00];

% Comprimento das sequencias artificiais
N_pre = 10;
N_pos = 30;

% Inicio da transicao artificial
inicio = N_pre + 1;

% Comprimento da rampa
N_rampa = 10;

% Numero de realizacoes por amplitude
N_rep = 100;

% Semente reprodutivel
rand('seed',42);
randn('seed',42);

nomes_var = {'RMS_B','Desequilibrio'};

% ============================================================
% 2. GNUPLOT
% ============================================================

motores = available_graphics_toolkits();

if ~any(strcmp(motores,'gnuplot'))
    error('Toolkit gnuplot indisponivel.');
end

graphics_toolkit('gnuplot');

set(0,'defaultfigurecolor',[1 1 1]);
set(0,'defaultaxesfontsize',11);

fprintf('\n========================================\n');
fprintf('G3/E04.1 - AUDITORIA DE SENSIBILIDADE\n');
fprintf('========================================\n');
fprintf('Motor grafico: gnuplot\n');

% ============================================================
% 3. LEITURA DOS CSVs
% ============================================================

fid = fopen(arquivo_dados,'rt');

if fid < 0
    error('Arquivo de caracteristicas nao encontrado.');
end

cab = fgetl(fid);
cab = strrep(cab,char([239 187 191]),'');
nomes = strtrim(strsplit(cab,','));

C = textscan(fid,repmat('%s',1,numel(nomes)), ...
    'Delimiter',',','ReturnOnError',false);

fclose(fid);

N = numel(C{1});

for j=1:numel(C)
    if numel(C{j})~=N
        error('CSV de caracteristicas inconsistente.');
    end
end

fid = fopen(arquivo_manifesto,'rt');

if fid < 0
    error('Manifesto G3/E02 nao encontrado.');
end

cab = fgetl(fid);
cab = strrep(cab,char([239 187 191]),'');
nomes_m = strtrim(strsplit(cab,','));

M = textscan(fid,repmat('%s',1,numel(nomes_m)), ...
    'Delimiter',',','ReturnOnError',false);

fclose(fid);

if numel(M{1})~=N
    error('Manifesto com numero incorreto de linhas.');
end

% ============================================================
% 4. IDENTIFICACAO DAS COLUNAS
% ============================================================

col = @(h,s) find(strcmp(h,s));

campos = {'registro','regiao','fase','janela', ...
          'rms_A','desequilibrio_rms_pct'};

ix = zeros(1,numel(campos));

for j=1:numel(campos)

    z = col(nomes,campos{j});

    if numel(z)~=1
        error('Coluna ausente ou duplicada: %s', ...
              campos{j});
    end

    ix(j)=z;
end

campos_m = {'linha_csv','registro','fase', ...
            'regiao','grupo'};

im = zeros(1,numel(campos_m));

for j=1:numel(campos_m)

    z=col(nomes_m,campos_m{j});

    if numel(z)~=1
        error('Coluna invalida no manifesto: %s', ...
              campos_m{j});
    end

    im(j)=z;
end

registro = str2double(C{ix(1)});
regiao = C{ix(2)};
fase_txt = C{ix(3)};
janela = str2double(C{ix(4)});

rms = str2double(C{ix(5)});
imb = str2double(C{ix(6)});

grupo = M{im(5)};

if any(~isfinite([registro janela rms imb](:)))
    error('Dados numericos invalidos.');
end

if any(registro~=floor(registro)) || ...
   any(janela~=floor(janela))
    error('Identificadores nao inteiros.');
end

linhas_m = str2double(M{im(1)});
registros_m = str2double(M{im(2)});

if any(linhas_m(:)~=(2:N+1)') || ...
   any(registros_m~=registro)
    error('Manifesto fora de ordem.');
end

% ============================================================
% 5. VALIDACAO DE FASES E REGIOES
% ============================================================

y = NaN(N,1);
fase = NaN(N,1);

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
                strtrim(M{im(3)}{i}))
        error('Fase inconsistente na linha %d.',i);
    end

    if ~strcmpi(strtrim(regiao{i}), ...
                strtrim(M{im(4)}{i}))
        error('Regiao inconsistente na linha %d.',i);
    end
end

if any(isnan(y)) || any(isnan(fase))
    error('Fase ou regiao invalida.');
end

treino = strcmp(grupo,'treino');
valid = strcmp(grupo,'validacao');
teste = strcmp(grupo,'teste');

if any(~(treino|valid|teste))
    error('Grupo desconhecido.');
end

if sum(treino)~=948 || ...
   sum(valid)~=237 || ...
   sum(teste)~=237
    error('Particao diferente do G3/E02.');
end

% ============================================================
% 6. AGREGACAO TRIFASICA
% ============================================================

chaves = unique([registro y janela],'rows');

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
        error('Grupos diferentes na mesma janela.');
    end

    tol=1e-7*max(1,max(abs(imb(idx))));

    if max(imb(idx))-min(imb(idx))>tol
        error('Desequilibrio global inconsistente.');
    end

    if strcmp(grupo{idx(1)},'teste')
        n_teste=n_teste+1;
        continue;
    end

    R(end+1,1)=registro(idx(1));
    W(end+1,1)=janela(idx(1));
    Y(end+1,1)=y(idx(1));
    G{end+1,1}=grupo{idx(1)};

    % RMS fase B + desequilibrio global
    X(end+1,:)=[rms(idx(2)),imb(idx(1))];
end

J=rows(X);

if J~=395 || n_teste~=79
    error('Contagem inesperada de janelas.');
end

fprintf('\nJanelas de desenvolvimento: %d\n',J);
fprintf('Janelas de teste reservadas: %d\n',n_teste);

% ============================================================
% 7. NORMALIZACAO
% ============================================================

idx_ref=strcmp(G,'treino') & Y==0;

mu=mean(X(idx_ref,:),1);
sigma=std(X(idx_ref,:),0,1);

if any(~isfinite(mu)) || ...
   any(~isfinite(sigma)) || any(sigma<=0)
    error('Referencia estatistica degenerada.');
end

Z=(X-repmat(mu,J,1))./repmat(sigma,J,1);

fprintf('\n--- REFERENCIA DE TREINO ---\n');

for v=1:2
    fprintf('%s: media=%.8f sigma=%.8f\n', ...
        nomes_var{v},mu(v),sigma(v));
end

% ============================================================
% 8. AUDITORIA DOS DESVIOS PADRONIZADOS
% ============================================================

arquivo=fullfile(saida, ...
    'G3_E04_1_Normalizacao.csv');

fid=fopen(arquivo,'wt');

fprintf(fid, ...
    'grupo,variavel,regiao,n,minimo,mediana,maximo,media,desvio\n');

grupos={'treino','validacao'};

for g=1:2

    idxg=strcmp(G,grupos{g});

    for v=1:2

        for classe=0:1

            idx=idxg & Y==classe;

            z=Z(idx,v);

            fprintf(fid, ...
                '%s,%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
                grupos{g},nomes_var{v},classe,numel(z), ...
                min(z),median(z),max(z),mean(z),std(z));

            fprintf('%s | %s | classe=%d | z=[%.3f, %.3f]\n', ...
                grupos{g},nomes_var{v},classe, ...
                min(z),max(z));
        end
    end
end

fclose(fid);

% ============================================================
% 9. FUNCOES DE DETECCAO
% ============================================================

% Implementacao inline por loops.
%
% CUSUM:
% S(t)=max(0,S(t-1)+z(t)-k)
%
% EWMA:
% E(t)=lambda*z(t)+(1-lambda)*E(t-1)
%
% Limite EWMA:
% L*sqrt(lambda/(2-lambda)*
%        (1-(1-lambda)^(2*t)))

% ============================================================
% 10. EXPERIMENTO DE SENSIBILIDADE
% ============================================================

% Sinais artificiais em unidades de sigma.
%
% Cada realizacao:
% - ruido normal N(0,1)
% - 10 janelas iniciais sem deslocamento
% - rampa de 10 janelas
% - patamar ate o final
%
% A simulacao nao representa um motor fisico.
%
% A mesma realizacao de ruido e usada para
% todas as amplitudes, permitindo comparacao
% pareada entre cenarios.
%
% Os parametros dos detectores permanecem fixos.

T=N_pre+N_pos;

resultados=zeros(numel(amplitudes),2,4);

arquivo=fullfile(saida, ...
    'G3_E04_1_Sensibilidade.csv');

fid=fopen(arquivo,'wt');

fprintf(fid, ...
 ['amplitude_sigma,metodo,repeticoes,' ...
  'taxa_deteccao,media_atraso_janelas,' ...
  'falsos_alarmes_pre,primeira_janela_media\n']);

for a=1:numel(amplitudes)

    amp=amplitudes(a);

    det_c=0;
    det_e=0;

    atrasos_c=[];
    atrasos_e=[];

    fp_c=0;
    fp_e=0;

    for rep=1:N_rep

        % Mesma semente por repeticao e amplitude:
        % comparabilidade dos cenarios.
        randn('seed',1000+rep);

        z=randn(T,1);

        deslocamento=zeros(T,1);

        for t=inicio:T

            q=(t-inicio+1)/N_rampa;
            q=min(1,q);

            deslocamento(t)=amp*q;
        end

        z=z+deslocamento;

        sc=zeros(T,1);
        se=zeros(T,1);

        ac=false(T,1);
        ae=false(T,1);

        c=0;
        e=0;

        for t=1:T

            c=max(0,c+z(t)-k_cusum);
            e=lambda*z(t)+(1-lambda)*e;

            lim_e=L_ewma*sqrt( ...
                lambda/(2-lambda)* ...
                (1-(1-lambda)^(2*t)));

            sc(t)=c;
            se(t)=e;

            ac(t)=c>h_cusum;
            ae(t)=e>lim_e;
        end

        % Falsos alarmes no periodo anterior
        fp_c=fp_c+any(ac(1:N_pre));
        fp_e=fp_e+any(ae(1:N_pre));

        % Primeiro alarme apos o inicio da rampa
        ic=find(ac(inicio:end),1);
        ie=find(ae(inicio:end),1);

        if ~isempty(ic)
            det_c=det_c+1;
            atrasos_c(end+1)=ic;
        end

        if ~isempty(ie)
            det_e=det_e+1;
            atrasos_e(end+1)=ie;
        end
    end

    taxa_c=det_c/N_rep;
    taxa_e=det_e/N_rep;

    if isempty(atrasos_c)
        atraso_c=NaN;
    else
        atraso_c=mean(atrasos_c);
    end

    if isempty(atrasos_e)
        atraso_e=NaN;
    else
        atraso_e=mean(atrasos_e);
    end

    resultados(a,1,:)=[taxa_c atraso_c fp_c/N_rep atraso_c];
    resultados(a,2,:)=[taxa_e atraso_e fp_e/N_rep atraso_e];

    fprintf(fid, ...
        '%.4f,CUSUM,%d,%.8f,%.8f,%.8f,%.8f\n', ...
        amp,N_rep,taxa_c,atraso_c,fp_c/N_rep,atraso_c);

    fprintf(fid, ...
        '%.4f,EWMA,%d,%.8f,%.8f,%.8f,%.8f\n', ...
        amp,N_rep,taxa_e,atraso_e,fp_e/N_rep,atraso_e);

    fprintf(['Amplitude %.2f sigma | ' ...
             'CUSUM %.1f%% | EWMA %.1f%%\n'], ...
        amp,100*taxa_c,100*taxa_e);
end

fclose(fid);

% ============================================================
% 11. FIGURA 1 - DISTRIBUICAO DOS Z-SCORES
% ============================================================

% Grupos pre e pos mostrados separadamente.
% Escala logaritmica nao e usada porque z
% pode assumir valores negativos.

for v=1:2

    idxpre=strcmp(G,'validacao') & Y==0;
    idxpos=strcmp(G,'validacao') & Y==1;

    zp=Z(idxpre,v);
    zq=Z(idxpos,v);

    fig=figure('visible','on','color','w', ...
        'position',[100 100 1100 520]);

    subplot(1,2,1);

    hist(zp,15);
    title('Pre-evento');
    xlabel('Desvio padronizado z');
    ylabel('Frequencia');
    grid on;

    subplot(1,2,2);

    hist(zq,15);
    title('Pos-evento');
    xlabel('Desvio padronizado z');
    ylabel('Frequencia');
    grid on;

    set(gcf,'paperpositionmode','auto');

    drawnow();

    print(fig,fullfile(saida, ...
        sprintf('G3_E04_1_Fig%02d_Normalizacao.png',v)), ...
        '-dpng','-r200');

    close(fig);
end

% ============================================================
% 12. FIGURA 3 - TAXA DE DETECCAO
% ============================================================

taxa_c=squeeze(resultados(:,1,1));
taxa_e=squeeze(resultados(:,2,1));

fig=figure('visible','on','color','w', ...
    'position',[100 100 1050 520]);

plot(amplitudes,100*taxa_c,'-o', ...
    'linewidth',2,'markersize',7);

hold on;

plot(amplitudes,100*taxa_e,'-s', ...
    'linewidth',2,'markersize',7);

xlabel('Amplitude da mudanca (sigma)');
ylabel('Realizacoes com alarme pos-mudanca (%)');

title('G3/E04.1 - Sensibilidade a mudancas graduais');

ylim([0 105]);
xlim([0 max(amplitudes)+0.2]);

legend({'CUSUM','EWMA'},'location','southeast');

grid on;
box off;

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E04_1_Fig03_TaxaDeteccao.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 13. FIGURA 4 - ATRASO CONDICIONAL
% ============================================================

atraso_c=squeeze(resultados(:,1,2));
atraso_e=squeeze(resultados(:,2,2));

fig=figure('visible','on','color','w', ...
    'position',[100 100 1050 520]);

plot(amplitudes,atraso_c,'-o', ...
    'linewidth',2,'markersize',7);

hold on;

plot(amplitudes,atraso_e,'-s', ...
    'linewidth',2,'markersize',7);

xlabel('Amplitude da mudanca (sigma)');
ylabel('Primeiro alarme apos inicio da rampa (janelas)');

title('G3/E04.1 - Atraso condicional de deteccao');

legend({'CUSUM','EWMA'},'location','northeast');

grid on;
box off;

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E04_1_Fig04_Atraso.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 14. FIGURA 5 - EXEMPLO COM ESCALA VISIVEL
% ============================================================

% Exemplo de mudanca moderada de 2 sigma.
% A mesma sequencia alimenta os dois detectores.

randn('seed',2026);

amp=2.0;

z=randn(T,1);
deslocamento=zeros(T,1);

for t=inicio:T
    q=min(1,(t-inicio+1)/N_rampa);
    deslocamento(t)=amp*q;
end

z=z+deslocamento;

sc=zeros(T,1);
se=zeros(T,1);
le=zeros(T,1);

c=0;
e=0;

for t=1:T

    c=max(0,c+z(t)-k_cusum);
    e=lambda*z(t)+(1-lambda)*e;

    sc(t)=c;
    se(t)=e;

    le(t)=L_ewma*sqrt( ...
        lambda/(2-lambda)* ...
        (1-(1-lambda)^(2*t)));
end

% CUSUM - escala local
fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 520]);

plot(1:T,sc,'-o', ...
    'linewidth',1.6,'markersize',4);

hold on;

plot([1 T],[h_cusum h_cusum],'--', ...
    'linewidth',1.7);

plot([inicio inicio],ylim(),'k:', ...
    'linewidth',1.4);

xlabel('Indice da janela artificial');
ylabel('Estatistica CUSUM');

title('G3/E04.1 - CUSUM em mudanca gradual de 2 sigma');

legend({'CUSUM','Limiar','Inicio da mudanca'}, ...
    'location','northwest');

grid on;
box off;

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E04_1_Fig05_CUSUM_Gradual.png'), ...
    '-dpng','-r200');

close(fig);

% EWMA
fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 520]);

plot(1:T,se,'-o', ...
    'linewidth',1.6,'markersize',4);

hold on;

plot(1:T,le,'--','linewidth',1.7);

plot([inicio inicio],ylim(),'k:', ...
    'linewidth',1.4);

xlabel('Indice da janela artificial');
ylabel('Estatistica EWMA');

title('G3/E04.1 - EWMA em mudanca gradual de 2 sigma');

legend({'EWMA','Limiar','Inicio da mudanca'}, ...
    'location','northwest');

grid on;
box off;

set(gcf,'paperpositionmode','auto');
drawnow();

print(fig,fullfile(saida, ...
    'G3_E04_1_Fig06_EWMA_Gradual.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 15. PROTOCOLO
% ============================================================

fid=fopen(fullfile(saida, ...
    'G3_E04_1_Protocolo.txt'),'wt');

fprintf(fid,'G3/E04.1 - Auditoria de sensibilidade\n\n');

fprintf(fid,'CUSUM: k=%.3f h=%.3f\n', ...
    k_cusum,h_cusum);

fprintf(fid,'EWMA: lambda=%.3f L=%.3f\n', ...
    lambda,L_ewma);

fprintf(fid,'\nReferencia estatistica:\n');
fprintf(fid,'Pre-evento do treinamento.\n');

fprintf(fid,'\nExperimento artificial:\n');
fprintf(fid,'Ruido gaussiano padronizado N(0,1).\n');
fprintf(fid,'Amplitude: 0.25 a 3 sigma.\n');
fprintf(fid,'Rampa: %d janelas.\n',N_rampa);
fprintf(fid,'Repeticoes: %d por amplitude.\n',N_rep);

fprintf(fid,'\nLimitacoes:\n');
fprintf(fid,'1. Nao representa uma simulacao fisica SRM.\n');
fprintf(fid,'2. Realizacoes nao sao registros independentes.\n');
fprintf(fid,'3. Ruido artificial sem estrutura de comutacao.\n');
fprintf(fid,'4. Sem avaliacao do conjunto de teste.\n');
fprintf(fid,'5. Atrasos expressos em janelas artificiais.\n');
fprintf(fid,'6. Parametros nao otimizados.\n');
fprintf(fid,'7. Nao existe referencia saudavel comprovada.\n');
fprintf(fid,'8. Atraso medio condicionado a deteccao.\n');
fprintf(fid,'9. Falsos alarmes pre sao por realizacao.\n');

fclose(fid);

% ============================================================
% 16. FINALIZACAO
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E04.1 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Janelas reais de desenvolvimento: %d\n',J);
fprintf('Janelas reais de teste reservadas: %d\n',n_teste);

fprintf('Amplitudes artificiais: %d\n', ...
    numel(amplitudes));

fprintf('Realizacoes por amplitude: %d\n',N_rep);

fprintf('Figuras geradas: 6\n');
fprintf('Motor grafico: gnuplot\n');

fprintf('\nResultados salvos em:\n%s\n',saida);
