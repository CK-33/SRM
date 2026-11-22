
% ============================================================
% PESQUISA SRM - GATE 3 / EXPERIMENTO 03.1
%
% Auditoria dos detectores determinísticos
% e padronizacao grafica com GNUPLOT
%
% GNU Octave 11.3.0
% Sem pacotes adicionais
%
% ENTRADAS:
%   Gate2_Exp05/G2_E05_Caracteristicas.csv
%   Gate3_Exp02/G3_E02_Manifesto.csv
%   Gate3_Exp03/G3_E03_Limiares.csv
%
% SAIDAS:
%   Gate3_Exp03.1/
%
% REGRAS:
%   - Nao abrir arquivos MAT
%   - Nao recalcular limiares
%   - Nao avaliar registros de teste
%   - Uma observacao por janela trifasica
%   - Graficos exclusivamente com gnuplot
% ============================================================

clear;
clc;
close all;

% ============================================================
% 0. CONFIGURACAO
% ============================================================

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

arq_dados = fullfile(base, ...
    'Gate2_Exp05', 'G2_E05_Caracteristicas.csv');

arq_manifesto = fullfile(base, ...
    'Gate3_Exp02', 'G3_E02_Manifesto.csv');

arq_limiares = fullfile(base, ...
    'Gate3_Exp03', 'G3_E03_Limiares.csv');

saida = fullfile(base, 'Gate3_Exp03.1');

if ~exist(saida,'dir')
    mkdir(saida);
end

% Configuracao grafica obrigatoria.
motores = available_graphics_toolkits();

if ~any(strcmp(motores,'gnuplot'))
    error('O toolkit gnuplot nao esta disponivel.');
end

graphics_toolkit('gnuplot');

set(0,'defaultfigurecolor',[1 1 1]);
set(0,'defaultaxesfontsize',11);

fprintf('\n========================================\n');
fprintf('G3/E03.1 - AUDITORIA DETERMINISTICA\n');
fprintf('========================================\n');
fprintf('Motor grafico: gnuplot\n');

% ============================================================
% 1. LEITURA DOS CSVs
% ============================================================

% Leitura de caracteristicas
fid = fopen(arq_dados,'rt');
if fid < 0, error('CSV de dados nao encontrado.'); end

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

% Leitura de manifesto
fid = fopen(arq_manifesto,'rt');
if fid < 0, error('Manifesto nao encontrado.'); end

cab = fgetl(fid);
cab = strrep(cab,char([239 187 191]),'');
nomes_m = strtrim(strsplit(cab,','));

M = textscan(fid,repmat('%s',1,numel(nomes_m)), ...
    'Delimiter',',','ReturnOnError',false);
fclose(fid);

if numel(M{1})~=N
    error('Manifesto com quantidade de linhas diferente.');
end

% Leitura de limiares
fid = fopen(arq_limiares,'rt');
if fid < 0, error('CSV de limiares nao encontrado.'); end

cab = fgetl(fid);
cab = strrep(cab,char([239 187 191]),'');
nomes_l = strtrim(strsplit(cab,','));

L = textscan(fid,repmat('%s',1,numel(nomes_l)), ...
    'Delimiter',',','ReturnOnError',false);
fclose(fid);

% ============================================================
% 2. IDENTIFICACAO DAS COLUNAS
% ============================================================

col = @(nomes,nome) find(strcmp(nomes,nome));

cr = col(nomes,'registro');
cy = col(nomes,'regiao');
cf = col(nomes,'fase');
cw = col(nomes,'janela');

c_rms = col(nomes,'rms_A');
c_fc  = col(nomes,'fator_crista');
c_imb = col(nomes,'desequilibrio_rms_pct');

cm_linha = col(nomes_m,'linha_csv');
cm_reg = col(nomes_m,'registro');
cm_fase = col(nomes_m,'fase');
cm_regiao = col(nomes_m,'regiao');
cm_grupo = col(nomes_m,'grupo');

cl_det = col(nomes_l,'detector');
cl_fase = col(nomes_l,'fase');
cl_op = col(nomes_l,'operador');
cl_lim = col(nomes_l,'limiar');

campos = {cr,cy,cf,cw,c_rms,c_fc,c_imb, ...
    cm_linha,cm_reg,cm_fase,cm_regiao,cm_grupo, ...
    cl_det,cl_fase,cl_op,cl_lim};

for j=1:numel(campos)
    if numel(campos{j})~=1
        error('Coluna obrigatoria ausente ou duplicada.');
    end
end

registro = str2double(C{cr});
janela = str2double(C{cw});

rms = str2double(C{c_rms});
fc = str2double(C{c_fc});
imb = str2double(C{c_imb});

fase_txt = C{cf};
regiao_txt = C{cy};
grupo = M{cm_grupo};

numericos = [registro janela rms fc imb];

if any(~isfinite(numericos(:)))
    error('Caracteristicas numericas invalidas.');
end

if any(registro~=floor(registro)) || ...
   any(janela~=floor(janela))
    error('Identificadores nao inteiros.');
end

% ============================================================
% 3. AUDITORIA DO MANIFESTO
% ============================================================

linhas = str2double(M{cm_linha});
reg_m = str2double(M{cm_reg});

if any(linhas(:)~=(2:N+1)')
    error('Manifesto fora de ordem.');
end

if any(registro~=reg_m)
    error('Registros inconsistentes com manifesto.');
end

y = NaN(N,1);
fase = NaN(N,1);

for i=1:N

    s = lower(strtrim(regiao_txt{i}));
    p = upper(strtrim(fase_txt{i}));

    if strcmp(s,'pre'), y(i)=0; end
    if strcmp(s,'pos'), y(i)=1; end

    if strcmp(p,'A'), fase(i)=1; end
    if strcmp(p,'B'), fase(i)=2; end
    if strcmp(p,'C'), fase(i)=3; end

    if ~strcmpi(strtrim(fase_txt{i}), ...
                strtrim(M{cm_fase}{i}))
        error('Fase inconsistente na linha %d.',i);
    end

    if ~strcmpi(strtrim(regiao_txt{i}), ...
                strtrim(M{cm_regiao}{i}))
        error('Regiao inconsistente na linha %d.',i);
    end

end

if any(isnan(y)) || any(isnan(fase))
    error('Fase ou regiao nao reconhecida.');
end

treino = strcmp(grupo,'treino');
valid = strcmp(grupo,'validacao');
teste = strcmp(grupo,'teste');

if any(~(treino | valid | teste))
    error('Grupo desconhecido.');
end

if sum(treino)~=948 || ...
   sum(valid)~=237 || ...
   sum(teste)~=237
    error('Particao diferente do G3/E02.');
end

% ============================================================
% 4. RECUPERAR LIMIARES FIXOS
% ============================================================

lim_rms = NaN(3,1);
lim_fc = NaN(3,1);
lim_imb = NaN;

cont_rms = zeros(3,1);
cont_fc = zeros(3,1);
cont_imb = 0;

for i=1:numel(L{1})

    det = strtrim(L{cl_det}{i});
    ph = upper(strtrim(L{cl_fase}{i}));
    op = lower(strtrim(L{cl_op}{i}));
    v = str2double(L{cl_lim}{i});

    if ~isfinite(v)
        error('Limiar nao numerico.');
    end

    if strcmp(det,'RMS') || strcmp(det,'FatorCrista')

        p = double(ph)-double('A')+1;

        if numel(p)~=1 || p<1 || p>3
            error('Fase invalida no CSV de limiares.');
        end

        if strcmp(det,'RMS')
            if ~strcmp(op,'maior')
                error('Operador RMS inesperado.');
            end
            lim_rms(p)=v;
            cont_rms(p)=cont_rms(p)+1;
        else
            if ~strcmp(op,'menor')
                error('Operador de crista inesperado.');
            end
            lim_fc(p)=v;
            cont_fc(p)=cont_fc(p)+1;
        end

    elseif strcmp(det,'Desequilibrio')

        if ~strcmp(ph,'ABC') || ~strcmp(op,'maior')
            error('Configuracao de desequilibrio invalida.');
        end

        lim_imb=v;
        cont_imb=cont_imb+1;

    else
        error('Detector desconhecido no CSV de limiares.');
    end
end

if any(cont_rms~=1) || any(cont_fc~=1) || ...
   cont_imb~=1 || any(~isfinite(lim_rms)) || ...
   any(~isfinite(lim_fc)) || ~isfinite(lim_imb)
    error('Limiar ausente ou duplicado.');
end

fprintf('\nLimiares fixos recuperados:\n');

for p=1:3
    fprintf('Fase %s: RMS %.6f A | FC %.6f\n', ...
        char(64+p),lim_rms(p),lim_fc(p));
end

fprintf('Desequilibrio: %.6f %%\n',lim_imb);

% ============================================================
% 5. MONTAGEM DAS JANELAS TRIFASICAS
% ============================================================

% Chave correta:
% registro + regiao + janela

chaves = unique([registro y janela],'rows');
Jtotal = rows(chaves);

% Usaremos apenas desenvolvimento.
% As janelas de teste nao entram na analise.

R = [];
W = [];
Y = [];
G = {};

RMS3 = [];
FC3 = [];
IMB = [];

for k=1:Jtotal

    idx = find( ...
        registro==chaves(k,1) & ...
        y==chaves(k,2) & ...
        janela==chaves(k,3));

    if numel(idx)~=3
        error('Janela trifasica incompleta.');
    end

    [f,ordem] = sort(fase(idx));

    if ~isequal(f(:)',[1 2 3])
        error('Fases incorretas na janela.');
    end

    idx = idx(ordem);

    if numel(unique(grupo(idx)))~=1
        error('Grupos inconsistentes na janela.');
    end

    tol = 1e-7*max(1,max(abs(imb(idx))));

    if max(imb(idx))-min(imb(idx))>tol
        error('Desequilibrio global inconsistente.');
    end

    % Excluir o teste antes da analise.
    if strcmp(grupo{idx(1)},'teste')
        continue;
    end

    R(end+1,1) = registro(idx(1));
    W(end+1,1) = janela(idx(1));
    Y(end+1,1) = y(idx(1));
    G{end+1,1} = grupo{idx(1)};

    RMS3(end+1,:) = rms(idx)';
    FC3(end+1,:) = fc(idx)';
    IMB(end+1,1) = imb(idx(1));

end

J = numel(Y);

if J~=395
    error('Esperadas 395 janelas de desenvolvimento.');
end

fprintf('\nJanelas de desenvolvimento: %d\n',J);
fprintf('Janelas de teste excluidas: %d\n',Jtotal-J);

% ============================================================
% 6. DECISOES E MARGENS
% ============================================================

% Margem assinada:
% positiva = lado de alarme
% negativa = lado sem alarme
%
% RMS:
%   RMS - limiar
%
% Fator de crista:
%   limiar - FC
%
% Desequilibrio:
%   IMB - limiar

MR = RMS3 - repmat(lim_rms',J,1);

MF = repmat(lim_fc',J,1) - FC3;

MI = IMB - lim_imb;

DR = MR>0;
DF = MF>0;
DI = MI>0;

% Decisao global por janela
PR = any(DR,2);
PF = any(DF,2);
PI = DI;

P = [PR PI PF];

% Margem do detector agregado:
% maximo das margens de fase

margens = [max(MR,[],2), MI, max(MF,[],2)];

nomes_det = {'RMS','Desequilibrio','Fator de crista'};

% ============================================================
% 7. METRICAS POR REGISTRO
% ============================================================

arquivo = fullfile(saida, ...
    'G3_E03_1_Metricas_Registro.csv');

fid = fopen(arquivo,'wt');

fprintf(fid, ...
    ['registro,grupo,detector,TP,FP,TN,FN,' ...
     'precisao,recall,F1,FPR\n']);

regs = unique(R);

fprintf('\n--- RESULTADOS POR REGISTRO ---\n');

for r=regs(:)'

    idxr = R==r;
    nome_grupo = G{find(idxr,1)};

    for d=1:3

        yt=Y(idxr);
        yp=P(idxr,d);

        TP=sum(yt==1 & yp==1);
        FP=sum(yt==0 & yp==1);
        TN=sum(yt==0 & yp==0);
        FN=sum(yt==1 & yp==0);

        prec=TP/max(1,TP+FP);
        rec=TP/max(1,TP+FN);
        f1=2*prec*rec/max(eps,prec+rec);
        fpr=FP/max(1,FP+TN);

        fprintf(fid, ...
            '%d,%s,%s,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f\n', ...
            r,nome_grupo,nomes_det{d}, ...
            TP,FP,TN,FN,prec,rec,f1,fpr);

        fprintf('Registro %02d | %-17s | F1=%.3f FPR=%.3f\n', ...
            r,nomes_det{d},f1,fpr);
    end
end

fclose(fid);

% ============================================================
% 8. CONTRIBUICAO INDIVIDUAL DAS FASES
% ============================================================

arquivo = fullfile(saida, ...
    'G3_E03_1_Contribuicao_Fases.csv');

fid=fopen(arquivo,'wt');

fprintf(fid, ...
    'grupo,detector,fase,n_pre,n_pos,alarmes_pre,alarmes_pos\n');

grupos={'treino','validacao'};

for g=1:2

    idxg=strcmp(G,grupos{g});

    for d=1:2

        if d==1
            D=DR;
            nome='RMS';
        else
            D=DF;
            nome='FatorCrista';
        end

        for ph=1:3

            npre=sum(idxg & Y==0);
            npos=sum(idxg & Y==1);

            apre=sum(idxg & Y==0 & D(:,ph));
            apos=sum(idxg & Y==1 & D(:,ph));

            fprintf(fid, ...
                '%s,%s,%s,%d,%d,%d,%d\n', ...
                grupos{g},nome,char(64+ph), ...
                npre,npos,apre,apos);
        end
    end
end

fclose(fid);

% ============================================================
% 9. MARGENS DE SEPARACAO
% ============================================================

arquivo = fullfile(saida, ...
    'G3_E03_1_Margens.csv');

fid=fopen(arquivo,'wt');

fprintf(fid, ...
    ['grupo,detector,regiao,n,minimo,mediana,' ...
     'maximo,media,desvio\n']);

for g=1:2

    idxg=strcmp(G,grupos{g});

    for d=1:3

        for classe=0:1

            idx=idxg & Y==classe;
            v=margens(idx,d);

            v=sort(v);
            n=numel(v);

            if n==0
                error('Grupo sem classe.');
            end

            med=median(v);

            if n>1
                dp=std(v);
            else
                dp=0;
            end

            fprintf(fid, ...
                '%s,%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
                grupos{g},nomes_det{d},classe,n, ...
                min(v),med,max(v),mean(v),dp);
        end
    end
end

fclose(fid);

% ============================================================
% 10. PRIMEIRA JANELA POS-EVENTO
% ============================================================

% Esta analise verifica a primeira janela identificada
% como pos-evento segundo o indice de janela.
%
% Nao calcula atraso fisico em segundos.

arquivo=fullfile(saida, ...
    'G3_E03_1_Primeira_Janela.csv');

fid=fopen(arquivo,'wt');

fprintf(fid, ...
    ['registro,grupo,primeira_janela_pos,' ...
     'D1_RMS,D2_Imb,D3_Crista\n']);

for r=regs(:)'

    idx=find(R==r & Y==1);

    if isempty(idx)
        error('Registro sem janela pos-evento.');
    end

    [~,ord]=sort(W(idx));
    primeiro=idx(ord(1));

    fprintf(fid,'%d,%s,%d,%d,%d,%d\n', ...
        r,G{primeiro},W(primeiro),P(primeiro,:));
end

fclose(fid);

% ============================================================
% 11. RECONSTRUIR METRICAS GLOBAIS
% ============================================================

F1_val=zeros(3,1);
FPR_val=zeros(3,1);

idxv=strcmp(G,'validacao');

for d=1:3

    yt=Y(idxv);
    yp=P(idxv,d);

    TP=sum(yt==1 & yp==1);
    FP=sum(yt==0 & yp==1);
    TN=sum(yt==0 & yp==0);
    FN=sum(yt==1 & yp==0);

    prec=TP/max(1,TP+FP);
    rec=TP/max(1,TP+FN);

    F1_val(d)=2*prec*rec/max(eps,prec+rec);
    FPR_val(d)=FP/max(1,FP+TN);
end

% ============================================================
% 12. FIGURA 1 - F1 CORRIGIDO
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 500]);

barh(1:3,F1_val,0.55);

set(gca,'ytick',1:3, ...
    'yticklabel',nomes_det, ...
    'ydir','reverse','fontsize',12);

xlim([0 1.15]);
ylim([0.4 3.6]);

xlabel('F1 na validacao');
title('G3/E03 - Desempenho dos detectores');

grid on;

for d=1:3
    text(F1_val(d)+0.025,d, ...
        sprintf('%.3f',F1_val(d)), ...
        'fontsize',12);
end

set(gcf,'paperpositionmode','auto');

drawnow();

print(fig, ...
    fullfile(saida,'G3_E03_1_Fig01_F1.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 13. FIGURA 2 - FPR CORRIGIDO
% ============================================================

fig=figure('visible','on','color','w', ...
    'position',[100 100 1100 500]);

valores=100*FPR_val;

barh(1:3,valores,0.55);

set(gca,'ytick',1:3, ...
    'yticklabel',nomes_det, ...
    'ydir','reverse','fontsize',12);

limite=max(5,1.3*max(valores)+1);

xlim([0 limite]);
ylim([0.4 3.6]);

xlabel('Taxa de falsos positivos (%)');
title('G3/E03 - Falsos alarmes na validacao');

grid on;

for d=1:3
    text(valores(d)+0.03*limite,d, ...
        sprintf('%.2f%%',valores(d)), ...
        'fontsize',12);
end

set(gcf,'paperpositionmode','auto');

drawnow();

print(fig, ...
    fullfile(saida,'G3_E03_1_Fig02_FPR.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 14. FIGURA 3 - MARGENS NA VALIDACAO
% ============================================================

% Escalas diferentes entre detectores.
% Por isso, cada detector recebe uma figura propria.

for d=1:3

    idxpre=idxv & Y==0;
    idxpos=idxv & Y==1;

    vpre=margens(idxpre,d);
    vpos=margens(idxpos,d);

    fig=figure('visible','on','color','w', ...
        'position',[100 100 1050 500]);

    % Pontos com deslocamento vertical deterministico
    % para reduzir sobreposicao visual.

    np=numel(vpre);
    nq=numel(vpos);

    yp=1+0.12*sin((1:np)'*2.39996);
    yq=2+0.12*sin((1:nq)'*2.39996);

    plot(vpre,yp,'o','markersize',5);
    hold on;

    plot(vpos,yq,'s','markersize',5);

    yl=ylim();

    plot([0 0],yl,'k--','linewidth',1.3);

    set(gca,'ytick',[1 2], ...
        'yticklabel',{'Pre-evento','Pos-evento'}, ...
        'fontsize',12);

    ylim([0.65 2.35]);

    xlabel('Margem assinada em relacao ao limiar');
    title(['G3/E03.1 - Margens: ' nomes_det{d}]);

    legend({'Pre-evento','Pos-evento','Limiar'}, ...
        'location','northeastoutside');

    grid on;
    hold off;

    set(gcf,'paperpositionmode','auto');

    drawnow();

    print(fig,fullfile(saida, ...
        sprintf('G3_E03_1_Fig%02d_Margens.png',d+2)), ...
        '-dpng','-r200');

    close(fig);
end

% ============================================================
% 15. FIGURA 6 - CONTRIBUICAO DAS FASES
% ============================================================

% Percentual de janelas pos-evento que acionam
% cada fase, por detector.

idxpos=idxv & Y==1;

taxas=zeros(3,2);

for ph=1:3

    taxas(ph,1)=100*mean(DR(idxpos,ph));
    taxas(ph,2)=100*mean(DF(idxpos,ph));

end

fig=figure('visible','on','color','w', ...
    'position',[100 100 1050 500]);

bar(taxas,'grouped');

set(gca,'xtick',1:3, ...
    'xticklabel',{'Fase A','Fase B','Fase C'}, ...
    'fontsize',12);

ylim([0 110]);

ylabel('Janelas pos-evento com alarme (%)');

title('G3/E03.1 - Contribuicao individual das fases');

legend({'RMS','Fator de crista'}, ...
    'location','northeastoutside');

grid on;

set(gcf,'paperpositionmode','auto');

drawnow();

print(fig, ...
    fullfile(saida,'G3_E03_1_Fig06_Fases.png'), ...
    '-dpng','-r200');

close(fig);

% ============================================================
% 16. RELATORIO DE AUDITORIA
% ============================================================

fid=fopen(fullfile(saida, ...
    'G3_E03_1_Protocolo.txt'),'wt');

fprintf(fid,'G3/E03.1 - Auditoria dos detectores\n\n');

fprintf(fid,'Total de janelas: %d\n',Jtotal);
fprintf(fid,'Desenvolvimento: %d\n',J);
fprintf(fid,'Teste excluido: %d\n',Jtotal-J);

fprintf(fid,'\nLimiares importados do G3/E03.\n');
fprintf(fid,'Nenhum limiar foi recalculado.\n');

fprintf(fid,'\nMargens assinadas:\n');
fprintf(fid,'RMS = RMS - limiar\n');
fprintf(fid,'Crista = limiar - fator de crista\n');
fprintf(fid,'Desequilibrio = valor - limiar\n');

fprintf(fid,'\nLimites cientificos:\n');
fprintf(fid,'1. Rotulos pre/pos sao proxies temporais.\n');
fprintf(fid,'2. Nao existe referencia saudavel comprovada.\n');
fprintf(fid,'3. Registros podem compartilhar trajetorias.\n');
fprintf(fid,'4. Nao houve teste de generalizacao fisica.\n');
fprintf(fid,'5. Nao houve avaliacao dos registros de teste.\n');
fprintf(fid,'6. Primeira janela nao equivale a atraso fisico.\n');

fprintf(fid,'\nMotor grafico: gnuplot\n');
fprintf(fid,'Resolucao de exportacao: 200 dpi\n');

fclose(fid);

% ============================================================
% 17. FINALIZACAO
% ============================================================

fprintf('\n========================================\n');
fprintf('G3/E03.1 - EXECUCAO CONCLUIDA\n');
fprintf('========================================\n');

fprintf('Janelas de desenvolvimento: %d\n',J);
fprintf('Janelas de teste reservadas: %d\n',Jtotal-J);
fprintf('Detectores auditados: 3\n');

fprintf('\nF1 na validacao:\n');

for d=1:3
    fprintf('%-18s %.4f\n',nomes_det{d},F1_val(d));
end

fprintf('\nGraficos gerados: 6\n');
fprintf('Motor: gnuplot\n');

fprintf('\nResultados salvos em:\n%s\n',saida);
