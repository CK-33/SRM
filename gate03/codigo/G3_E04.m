
% ============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 04
% CUSUM E EWMA - ANALISE SEQUENCIAL PRELIMINAR
% GNU Octave 11.3.0
%
% Entradas:
% Gate2_Exp05/G2_E05_Caracteristicas.csv
% Gate3_Exp02/G3_E02_Manifesto.csv
%
% Nao acessa arquivos MAT.
% Nao avalia os registros de teste.
% ============================================================

clear; clc; close all;

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

entrada = fullfile(base,'Gate2_Exp05', ...
    'G2_E05_Caracteristicas.csv');

manifesto = fullfile(base,'Gate3_Exp02', ...
    'G3_E02_Manifesto.csv');

saida = fullfile(base,'Gate3_Exp04');

if ~exist(saida,'dir')
    mkdir(saida);
end

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

k_cusum = 0.5;
h_cusum = 5.0;

lambda = 0.20;
L_ewma = 3.0;

% CUSUM e EWMA unilaterais superiores.
% Caracteristicas monitoradas:
% 1 = RMS da fase B
% 2 = Desequilibrio trifasico

nomes_var = {'RMS_B','Desequilibrio'};

if ~any(strcmp(available_graphics_toolkits(),'gnuplot'))
    error('gnuplot nao disponivel.');
end

graphics_toolkit('gnuplot');

% ------------------------------------------------------------
% 2. LEITURA DOS DADOS
% ------------------------------------------------------------

fid=fopen(entrada,'rt');
if fid<0, error('CSV de caracteristicas ausente.'); end

cab=fgetl(fid);
cab=strrep(cab,char([239 187 191]),'');
nomes=strtrim(strsplit(cab,','));

C=textscan(fid,repmat('%s',1,numel(nomes)), ...
    'Delimiter',',','ReturnOnError',false);
fclose(fid);

N=numel(C{1});

for j=1:numel(C)
    if numel(C{j})~=N
        error('CSV de caracteristicas inconsistente.');
    end
end

fid=fopen(manifesto,'rt');
if fid<0, error('Manifesto G3/E02 ausente.'); end

cab=fgetl(fid);
cab=strrep(cab,char([239 187 191]),'');
nm=strtrim(strsplit(cab,','));

M=textscan(fid,repmat('%s',1,numel(nm)), ...
    'Delimiter',',','ReturnOnError',false);
fclose(fid);

if numel(M{1})~=N
    error('Manifesto e dados com tamanhos diferentes.');
end

% ------------------------------------------------------------
% 3. IDENTIFICAR COLUNAS
% ------------------------------------------------------------

col = @(h,s) find(strcmp(h,s));

campos = {'registro','regiao','fase','janela', ...
          'rms_A','desequilibrio_rms_pct'};

ix=zeros(1,numel(campos));

for j=1:numel(campos)
    z=col(nomes,campos{j});
    if numel(z)~=1
        error('Coluna invalida: %s',campos{j});
    end
    ix(j)=z;
end

campos_m={'linha_csv','registro','fase','regiao','grupo'};
im=zeros(1,numel(campos_m));

for j=1:numel(campos_m)
    z=col(nm,campos_m{j});
    if numel(z)~=1
        error('Coluna invalida no manifesto: %s', ...
              campos_m{j});
    end
    im(j)=z;
end

registro=str2double(C{ix(1)});
regiao=C{ix(2)};
fase_txt=C{ix(3)};
janela=str2double(C{ix(4)});
rms=str2double(C{ix(5)});
imb=str2double(C{ix(6)});

grupo=M{im(5)};

if any(~isfinite([registro janela rms imb](:)))
    error('Dados numericos invalidos.');
end

if any(registro~=floor(registro)) || ...
   any(janela~=floor(janela))
    error('Identificadores invalidos.');
end

if any(str2double(M{im(1)})~=(2:N+1)') || ...
   any(str2double(M{im(2)})~=registro)
    error('Manifesto nao corresponde aos dados.');
end

y=NaN(N,1);
fase=NaN(N,1);

for i=1:N
    s=lower(strtrim(regiao{i}));
    p=upper(strtrim(fase_txt{i}));

    if strcmp(s,'pre'), y(i)=0; end
    if strcmp(s,'pos'), y(i)=1; end

    if strcmp(p,'A'), fase(i)=1; end
    if strcmp(p,'B'), fase(i)=2; end
    if strcmp(p,'C'), fase(i)=3; end

    if ~strcmpi(strtrim(fase_txt{i}), ...
                strtrim(M{im(3)}{i})) || ...
       ~strcmpi(strtrim(regiao{i}), ...
                strtrim(M{im(4)}{i}))
        error('Manifesto inconsistente na linha %d.',i);
    end
end

if any(isnan(y)) || any(isnan(fase))
    error('Fase ou regiao desconhecida.');
end

treino=strcmp(grupo,'treino');
valid=strcmp(grupo,'validacao');
teste=strcmp(grupo,'teste');

if any(~(treino|valid|teste))
    error('Grupo invalido.');
end

if sum(treino)~=948 || sum(valid)~=237 || ...
   sum(teste)~=237
    error('Divisao diferente do G3/E02.');
end

% ------------------------------------------------------------
% 4. CONSTRUIR JANELAS TRIFASICAS
% ------------------------------------------------------------

chaves=unique([registro y janela],'rows');

R=[]; W=[]; Y=[]; G={}; X=[];

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
        continue;
    end

    R(end+1,1)=registro(idx(1));
    W(end+1,1)=janela(idx(1));
    Y(end+1,1)=y(idx(1));
    G{end+1,1}=grupo{idx(1)};

    % Uma observacao por janela trifasica.
    % RMS da fase B e desequilibrio global.
    X(end+1,:)=[rms(idx(2)),imb(idx(1))];
end

J=rows(X);

if J~=395
    error('Esperadas 395 janelas de desenvolvimento.');
end

fprintf('\n====================================\n');
fprintf('G3/E04 - CUSUM E EWMA\n');
fprintf('====================================\n');
fprintf('Janelas de desenvolvimento: %d\n',J);
fprintf('Janelas de teste reservadas: 79\n');

% ------------------------------------------------------------
% 5. REFERENCIA PRE-EVENTO DE TREINO
% ------------------------------------------------------------

idx_ref=strcmp(G,'treino') & Y==0;

mu=mean(X(idx_ref,:),1);
sigma=std(X(idx_ref,:),0,1);

if any(~isfinite(mu)) || ...
   any(~isfinite(sigma)) || any(sigma<=0)
    error('Referencia estatistica degenerada.');
end

fprintf('\nReferencias de treinamento:\n');

for v=1:2
    fprintf('%s: media=%.8f desvio=%.8f\n', ...
        nomes_var{v},mu(v),sigma(v));
end

% Normalizacao usando somente treinamento.
Z=(X-repmat(mu,J,1))./repmat(sigma,J,1);

% ------------------------------------------------------------
% 6. PROCESSAMENTO SEQUENCIAL
% ------------------------------------------------------------

% A ordem de processamento e:
%   1. registro
%   2. regiao
%   3. indice de janela
%
% As regioes sao processadas separadamente.
% O estado reinicia entre pre e pos.
%
% Isso NAO representa uma linha do tempo continua.

SC=zeros(J,2);
SE=zeros(J,2);

AC=zeros(J,2);
AE=zeros(J,2);

regs=unique(R);

for r=regs(:)'

    for classe=0:1

        idx=find(R==r & Y==classe);

        if isempty(idx)
            continue;
        end

        [~,ord]=sort(W(idx));
        idx=idx(ord);

        if numel(unique(W(idx)))~=numel(idx)
            error('Indices de janela duplicados.');
        end

        for v=1:2

            c=0;
            e=0;

            for t=1:numel(idx)

                j=idx(t);
                z=Z(j,v);

                % CUSUM unilateral superior.
                c=max(0,c+z-k_cusum);

                % EWMA unilateral em escala padronizada.
                e=lambda*z+(1-lambda)*e;

                % Limite EWMA dependente do numero
                % de observacoes desde o reinicio.
                limite=L_ewma*sqrt( ...
                    lambda/(2-lambda)* ...
                    (1-(1-lambda)^(2*t)));

                SC(j,v)=c;
                SE(j,v)=e;

                AC(j,v)=c>h_cusum;
                AE(j,v)=e>limite;
            end
        end
    end
end

% ------------------------------------------------------------
% 7. METRICAS POR REGISTRO E REGIAO
% ------------------------------------------------------------

% Alarmes por janela, nao por fase.
% As metricas sao descritivas porque as sequencias
% pre e pos foram reinicializadas separadamente.

arquivo=fullfile(saida,'G3_E04_Metricas.csv');
fid=fopen(arquivo,'wt');

fprintf(fid, ...
 ['grupo,registro,variavel,metodo,n_pre,n_pos,' ...
  'FP,TP,FPR,recall,primeiro_alarme_pos\n']);

for r=regs(:)'

    ir=R==r;
    g=G{find(ir,1)};

    for v=1:2

        for metodo=1:2

            if metodo==1
                A=AC(:,v);
                nome='CUSUM';
            else
                A=AE(:,v);
                nome='EWMA';
            end

            ipre=ir & Y==0;
            ipos=ir & Y==1;

            npre=sum(ipre);
            npos=sum(ipos);

            FP=sum(A(ipre));
            TP=sum(A(ipos));

            fpr=FP/max(1,npre);
            recall=TP/max(1,npos);

            % Primeiro alarme na sequencia pos,
            % expresso em numero ordinal de janela.
            ids=find(ipos);
            [~,ord]=sort(W(ids));
            ids=ids(ord);

            p=find(A(ids),1);

            if isempty(p)
                primeiro=NaN;
            else
                primeiro=p;
            end

            fprintf(fid, ...
              '%s,%d,%s,%s,%d,%d,%d,%d,%.8f,%.8f,%.0f\n', ...
              g,r,nomes_var{v},nome,npre,npos, ...
              FP,TP,fpr,recall,primeiro);

            fprintf(['Registro %02d | %-14s | %-5s | ' ...
                     'FPR=%.3f Recall=%.3f Primeiro=%g\n'], ...
                r,nomes_var{v},nome, ...
                fpr,recall,primeiro);
        end
    end
end

fclose(fid);

% ------------------------------------------------------------
% 8. EXPORTAR TRAJETORIAS DE DESENVOLVIMENTO
% ------------------------------------------------------------

arquivo=fullfile(saida,'G3_E04_Trajetorias.csv');
fid=fopen(arquivo,'wt');

fprintf(fid, ...
 ['registro,grupo,regiao,janela,variavel,' ...
  'valor_original,z,CUSUM,EWMA,' ...
  'alarme_CUSUM,alarme_EWMA\n']);

for j=1:J

    for v=1:2

        if Y(j)==0
            reg='pre';
        else
            reg='pos';
        end

        fprintf(fid, ...
          '%d,%s,%s,%d,%s,%.12g,%.12g,%.12g,%.12g,%d,%d\n', ...
          R(j),G{j},reg,W(j),nomes_var{v}, ...
          X(j,v),Z(j,v),SC(j,v),SE(j,v), ...
          AC(j,v),AE(j,v));
    end
end

fclose(fid);

% ------------------------------------------------------------
% 9. FIGURAS - GNUPLOT
% ------------------------------------------------------------

% Escolher registro 03, pertencente a validacao.
% Mostrar somente sequencia pos-evento.
%
% CUSUM e EWMA sao desenhados em figuras
% separadas porque possuem escalas distintas.

registro_fig=3;
variavel_fig=1;

idx=find(R==registro_fig & Y==1);

[~,ord]=sort(W(idx));
idx=idx(ord);

if isempty(idx)
    error('Registro de validacao sem janelas pos.');
end

n=numel(idx);
t=1:n;

for metodo=1:2

    if metodo==1

        valores=SC(idx,variavel_fig);
        limites=h_cusum*ones(n,1);

        titulo='CUSUM';
        nome_arquivo='G3_E04_Fig01_CUSUM.png';

    else

        valores=SE(idx,variavel_fig);

        limites=L_ewma*sqrt( ...
            lambda/(2-lambda)* ...
            (1-(1-lambda).^(2*t')));

        titulo='EWMA';
        nome_arquivo='G3_E04_Fig02_EWMA.png';
    end

    fig=figure('visible','on','color','w', ...
        'position',[100 100 1100 540]);

    plot(t,valores,'-o', ...
        'linewidth',1.8,'markersize',4);

    hold on;

    plot(t,limites,'--', ...
        'linewidth',1.6);

    xlabel('Ordem da janela pos-evento');
    ylabel('Estatistica padronizada');

    title(sprintf( ...
        'G3/E04 - %s | Registro %02d | RMS fase B', ...
        titulo,registro_fig));

    legend({'Estatistica','Limiar'}, ...
        'location','northwest');

    xlim([1 max(2,n)]);

    set(gca,'fontsize',12);

    grid on;
    box off;

    set(gcf,'paperpositionmode','auto');
    drawnow();

    print(fig,fullfile(saida,nome_arquivo), ...
        '-dpng','-r200');

    close(fig);
end

% ------------------------------------------------------------
% 10. PROTOCOLO
% ------------------------------------------------------------

fid=fopen(fullfile(saida, ...
    'G3_E04_Protocolo.txt'),'wt');

fprintf(fid,'G3/E04 - CUSUM e EWMA\n\n');

fprintf(fid,'CUSUM: k=%.4f h=%.4f\n', ...
    k_cusum,h_cusum);

fprintf(fid,'EWMA: lambda=%.4f L=%.4f\n', ...
    lambda,L_ewma);

fprintf(fid,'\nReferencia:\n');
fprintf(fid,'Media e desvio do pre-evento de treino.\n');

fprintf(fid,'\nLimitacoes:\n');
fprintf(fid,'1. Teste reservado.\n');
fprintf(fid,'2. Pre/pos sao proxies temporais.\n');
fprintf(fid,'3. Sequencias pre e pos reiniciadas.\n');
fprintf(fid,'4. Ordem cronologica nao comprovada.\n');
fprintf(fid,'5. Primeiro alarme nao e atraso fisico.\n');
fprintf(fid,'6. Parametros exploratorios nao calibrados.\n');
fprintf(fid,'7. Sem referencia saudavel independente.\n');
fprintf(fid,'8. Sem validacao fisica de falhas.\n');
fprintf(fid,'9. Nenhuma independencia entre registros presumida.\n');

fclose(fid);

fprintf('\n====================================\n');
fprintf('G3/E04 - EXECUCAO CONCLUIDA\n');
fprintf('====================================\n');

fprintf('Resultados: %s\n',saida);
fprintf('Motor grafico: gnuplot\n');
fprintf('Teste nao avaliado.\n');
