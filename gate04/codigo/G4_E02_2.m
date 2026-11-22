1;

% ============================================================
% G4_E02_2.m
% PESQUISA SRM
% GATE 4 - EXPERIMENTO 02.2
% Caracterizacao estatistica das variacoes naturais
% GNU Octave 11.3.0
% ============================================================

clear;
clc;
close all;

pkg load signal;

% ============================================================
% 1. FUNCOES AUXILIARES
% ============================================================

function y = rms_local(x)
    y = sqrt(mean(x.^2));
end

function q = percentil(x,p)

    x = sort(x(isfinite(x)));

    if isempty(x)
        q = NaN;
        return;
    end

    k = 1+(numel(x)-1)*p/100;

    a = floor(k);
    b = ceil(k);

    q = x(a)+(k-a)*(x(b)-x(a));

end

function s = resumo(x)

    x = x(isfinite(x));

    if isempty(x)
        s = [0 NaN(1,10)];
        return;
    end

    med = median(x);

    s = [
        numel(x), ...
        mean(x), ...
        std(x), ...
        med, ...
        median(abs(x-med)), ...
        min(x), ...
        percentil(x,5), ...
        percentil(x,25), ...
        percentil(x,75), ...
        percentil(x,95), ...
        max(x)
    ];

end

function exportar(fig,pasta,nome)

    set(fig,'paperpositionmode','auto');

    print(fig,fullfile(pasta,nome), ...
        '-dpng','-r150');

    close(fig);

    fprintf('Figura exportada: %s\n',nome);

end

function [t,I,E] = regularizar(S,Fs)

    ts = 1/Fs;

    to = double(S.results.time(:));

    Io = double(S.results.signals(3).values);

    Eo = double(S.results.signals(6).values(:));

    N = numel(to);

    if size(Io,1) ~= N && size(Io,2) == N
        Io = Io.';
    end

    assert(size(Io,1)==N);
    assert(size(Io,2)==3);
    assert(numel(Eo)==N);

    assert(all(isfinite(to)));
    assert(all(isfinite(Io(:))));
    assert(all(isfinite(Eo)));
    assert(all(diff(to)>0));

    Nu = round((to(end)-to(1))/ts)+1;

    t = to(1)+(0:Nu-1)'*ts;

    k = round((to-to(1))/ts)+1;

    dentro = k>=1 & k<=Nu;

    erro = inf(N,1);

    erro(dentro) = abs( ...
        to(dentro)-t(k(dentro)));

    idx = find(dentro & erro<=5e-9);

    contagem = accumarray( ...
        k(idx),1,[Nu 1],@sum,0);

    assert(all(contagem>0), ...
        'Grade temporal incompleta.');

    ultimo = accumarray( ...
        k(idx),idx,[Nu 1],@max,0);

    I = Io(ultimo,:);

    E = Eo(ultimo);

    assert(all(ultimo>0));
    assert(Nu==110000);

end

function [T,R] = periodo_acf(x,Fs,Tref)

    x = x(:)-mean(x);

    lmin = round(0.85*Tref*Fs);
    lmax = round(1.15*Tref*Fs);

    if numel(x)<=2*lmax || sum(x.^2)<=eps
        T = NaN;
        R = NaN;
        return;
    end

    [a,lags] = xcorr(x,lmax,'coeff');

    m = lags>=lmin & lags<=lmax;

    a = a(m);
    lags = lags(m);

    [R,j] = max(a);

    lag = lags(j);

    % Refinamento parabolico subamostral.

    if j>1 && j<numel(a)

        den = a(j-1)-2*a(j)+a(j+1);

        if abs(den)>eps

            delta = 0.5*(a(j-1)-a(j+1))/den;

            if abs(delta)<=1
                lag = lag+delta;
            end

        end

    end

    T = lag/Fs;

end

function [lag_s,R] = atraso_fases(x,y,Fs,Tref)

    x = x(:)-mean(x);
    y = y(:)-mean(y);

    if sum(x.^2)<=eps || sum(y.^2)<=eps
        lag_s = NaN;
        R = NaN;
        return;
    end

    lim = round(0.5*Tref*Fs);

    [a,lags] = xcorr(y,x,lim,'coeff');

    [R,j] = max(a);

    lag_s = lags(j)/Fs;

end

function [larguras,intervalos,picos] = pulsos( ...
    x,Fs,limiar,largura_min)

    x = x(:);

    ativo = x>=limiar;

    d = diff([false;ativo;false]);

    ini = find(d==1);
    fim = find(d==-1)-1;

    % Excluir pulsos cortados pelas fronteiras.

    valido = ini>1 & fim<numel(x);

    ini = ini(valido);
    fim = fim(valido);

    valido = (fim-ini+1)>=largura_min;

    ini = ini(valido);
    fim = fim(valido);

    larguras = (fim-ini+1)/Fs;

    intervalos = diff(ini)/Fs;

    picos = zeros(numel(ini),1);

    for j = 1:numel(ini)
        picos(j) = max(abs(x(ini(j):fim(j))));
    end

end

function P = bandas_psd(x,Fs)

    x = x(:);

    L = min(2048,numel(x));

    [Pxx,f] = pwelch( ...
        x,hamming(L),0.5,4096,Fs);

    B = [
        0 5000;
        5000 20000;
        20000 50000;
        50000 100000
    ];

    P = zeros(1,4);

    df = f(2)-f(1);

    for k = 1:4

        if k<4
            m = f>=B(k,1) & f<B(k,2);
        else
            m = f>=B(k,1) & f<=B(k,2);
        end

        P(k) = sum(Pxx(m))*df;

    end

end

function [dur,intervalos,frac] = retencao(x,tol)

    x = x(:);

    mudanca = find(abs(diff(x))>tol)+1;

    bordas = [1;mudanca;numel(x)+1];

    dur = diff(bordas);

    % Excluir retencoes incompletas nas extremidades.

    if numel(dur)>2
        dur = dur(2:end-1);
    else
        dur = [];
    end

    intervalos = diff(mudanca);

    frac = mean(abs(diff(x))<=tol);

end

function gravar_linha(fid,x)

    for k = 1:numel(x)

        if k>1
            fprintf(fid,',');
        end

        fprintf(fid,'%.12g',x(k));

    end

    fprintf(fid,'\n');

end

% ============================================================
% 2. CONFIGURACAO
% ============================================================

ROOT = [
    'PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
    'TesteNo1_31OCT2026/Data and Scripts'
];

DATA_ROOT = fullfile(ROOT,'Results_1_to_12');

REG_TREINO = [1 2 4 5 7 8 10 11];

REG_VALID = [3 9];

REGISTROS = sort([REG_TREINO REG_VALID]);

REG_TESTE = [6 12];

assert(~any(ismember(REGISTROS,REG_TESTE)));

FS = 200000;
TS = 1/FS;

TREF = 0.01308;

NREF = round(TREF*FS);

TRANSITORIO = 0.200;

FASES = {'A','B','C'};

REGIOES = {
    'pre';
    'transitorio';
    'tardio'
};

TEVENTOS = [
    1.0017;
    1.0019;
    1.0021;
    1.0024;
    1.0026;
    1.0030;
    1.0040;
    1.0050;
    1.0060;
    1.0070;
    1.0075;
    1.0080
];

% Limiar operacional de atividade:
% 10% do pico pre-evento mediano dos registros
% de treinamento, calculado posteriormente.

FRAC_LIMIAR = 0.10;

% Duracao minima de atividade: 200 us.

PULSO_MIN = round(200e-6*FS);

% Tolerancias para investigacao de retencao.

TOLS_RETENCAO = [0 1e-8 1e-6 1e-4 1e-3];

% ============================================================
% 3. DIRETORIO DE SAIDA
% ============================================================

run_id = 1;

while true

    OUT = fullfile(ROOT,sprintf( ...
        'Gate4_Exp02_2_Run%02d',run_id));

    if exist(OUT,'dir')~=7
        break;
    end

    run_id = run_id+1;

end

mkdir(OUT);

FIG = fullfile(OUT,'Figuras');

mkdir(FIG);

graphics_toolkit('gnuplot');

set(0,'defaultfigurevisible','off');
set(0,'defaultaxesfontsize',11);
set(0,'defaulttextfontsize',11);

fprintf('\n==============================================\n');
fprintf(' PESQUISA SRM\n');
fprintf(' GATE 4 - EXPERIMENTO 02.2\n');
fprintf(' Caracterizacao estatistica\n');
fprintf('==============================================\n');

fprintf('Fs: %.0f Hz\n',FS);
fprintf('Periodo de referencia: %.6f ms\n',TREF*1000);
fprintf('Amostras por janela: %d\n',NREF);
fprintf('Registros de treino: %d\n',numel(REG_TREINO));
fprintf('Registros de validacao: %d\n',numel(REG_VALID));
fprintf('Registros 06 e 12: RESERVADOS\n');
fprintf('Saida: %s\n\n',OUT);

% ============================================================
% 4. CALIBRACAO EXCLUSIVA COM TREINAMENTO
% ============================================================

PICOS_TREINO = zeros(numel(REG_TREINO),3);

for j = 1:numel(REG_TREINO)

    r = REG_TREINO(j);

    arquivo = fullfile(DATA_ROOT, ...
        sprintf('results_%d.mat',r));

    S = load(arquivo,'results');

    [t,I,E] = regularizar(S,FS);

    pre = t<TEVENTOS(r)-1e-10;

    assert(any(pre));

    PICOS_TREINO(j,:) = max(abs(I(pre,:)),[],1);

    clear S t I E;

end

LIMIARES = FRAC_LIMIAR*median(PICOS_TREINO,1);

fprintf('LIMIARES OPERACIONAIS DE ATIVIDADE\n');

for p = 1:3

    fprintf('Fase %s: %.9f A\n', ...
        FASES{p},LIMIARES(p));

end

% ============================================================
% 5. ABERTURA DOS ARQUIVOS
% ============================================================

fw = fopen(fullfile(OUT, ...
    'G4_E02_2_Janelas.csv'),'w');

fp = fopen(fullfile(OUT, ...
    'G4_E02_2_Periodicidade.csv'),'w');

fu = fopen(fullfile(OUT, ...
    'G4_E02_2_Pulsos.csv'),'w');

fr = fopen(fullfile(OUT, ...
    'G4_E02_2_Retencao.csv'),'w');

fc = fopen(fullfile(OUT, ...
    'G4_E02_2_Correlacoes.csv'),'w');

fe = fopen(fullfile(OUT, ...
    'G4_E02_2_Marcadores.csv'),'w');

ft = fopen(fullfile(OUT, ...
    'G4_E02_2_Evolucao_Temporal.csv'),'w');

assert(all([fw fp fu fr fc fe ft]>=0));

fprintf(fw, ...
    ['registro,grupo,regiao,janela,fase,' ...
     'tempo_relativo_centro_ms,rms_A,media_A,' ...
     'desvio_A,pico_A,pico_pico_A,' ...
     'energia_A2s,fator_crista,' ...
     'rms_derivada_A_s,atividade_pct,' ...
     'psd_0_5k,psd_5_20k,' ...
     'psd_20_50k,psd_50_100k,' ...
     'rms_trifasico_A,desequilibrio_pct\n']);

fprintf(fp, ...
    ['registro,grupo,regiao,fase,' ...
     'periodo_acf_ms,frequencia_Hz,acf_max,' ...
     'periodo_pulsos_mediano_ms,' ...
     'periodo_pulsos_mad_ms,' ...
     'n_intervalos_pulsos\n']);

fprintf(fu, ...
    ['registro,grupo,regiao,fase,pulso,' ...
     'largura_ms,pico_A\n']);

fprintf(fr, ...
    ['registro,grupo,regiao,fase,tolerancia_A,' ...
     'fracao_incrementos_nulos,' ...
     'n_retencoes,moda_amostras,' ...
     'mediana_amostras,p95_amostras,' ...
     'frac_retencoes_8,' ...
     'frac_retencoes_multiplos_8,' ...
     'frac_intervalos_mudanca_8,' ...
     'frac_intervalos_mudanca_multiplos_8\n']);

fprintf(fc, ...
    ['registro,grupo,regiao,' ...
     'corr_AB,corr_AC,corr_BC,' ...
     'lag_AB_ms,lag_AC_ms,lag_BC_ms,' ...
     'xcorr_AB,xcorr_AC,xcorr_BC\n']);

fprintf(fe, ...
    ['registro,grupo,t_evento_nominal_s,' ...
     't_primeira_mudanca_marcador_s,' ...
     'erro_evento_us,n_transicoes,' ...
     'marcador_pre,marcador_no_evento,' ...
     'marcador_final,status\n']);

fprintf(ft, ...
    ['registro,grupo,regiao,janela,' ...
     'tempo_relativo_ms,rms_A,rms_B,rms_C,' ...
     'rms_trifasico,desequilibrio_pct\n']);

% ============================================================
% 6. ESTRUTURAS DE RESULTADOS
% ============================================================

% W:
% 1 registro
% 2 grupo: 1 treino / 2 validacao
% 3 regiao
% 4 janela
% 5 fase
% 6 tempo relativo central ms
% 7:21 caracteristicas

W = [];

% TRI:
% 1 registro / 2 grupo / 3 regiao
% 4 janela / 5 tempo relativo
% 6:10 RMS A/B/C, RMS trifasico, desequilibrio

TRI = [];

PER = [];

PUL = [];

RET = [];

COR = [];

EVENT = [];

% Vetor trifasico por janela:
% 13 caracteristicas por fase + 2 globais.

X = [];

META = [];

NOMES_BASE = {
    'RMS';
    'Media';
    'Desvio';
    'Pico';
    'PicoPico';
    'Energia';
    'Crista';
    'RMS_Derivada';
    'Atividade';
    'PSD_0_5k';
    'PSD_5_20k';
    'PSD_20_50k';
    'PSD_50_100k'
};

NOMES_X = {};

for p = 1:3
    for j = 1:numel(NOMES_BASE)

        NOMES_X{end+1} = sprintf( ...
            '%s_%s',FASES{p},NOMES_BASE{j});

    end
end

NOMES_X{end+1} = 'RMS_Trifasico';

NOMES_X{end+1} = 'Desequilibrio';

DEMO = struct();

% ============================================================
% 7. PROCESSAMENTO DOS REGISTROS
% ============================================================

for r = REGISTROS

    if any(REG_TREINO==r)

        grupo = 'treino';
        g = 1;

    else

        grupo = 'validacao';
        g = 2;

    end

    fprintf('\n----------------------------------------------\n');
    fprintf('Registro %02d | %s\n',r,grupo);
    fprintf('----------------------------------------------\n');

    arquivo = fullfile(DATA_ROOT, ...
        sprintf('results_%d.mat',r));

    S = load(arquivo,'results');

    [t,I,E] = regularizar(S,FS);

    clear S;

    te = TEVENTOS(r);

    [erro_te,ke] = min(abs(t-te));

    assert(erro_te<1e-9);
    assert(ke>1);

    % --------------------------------------------------------
    % 7.1. Auditoria do marcador discreto
    % --------------------------------------------------------

    Epre = E(ke-1);

    Eevento = E(ke);

    Efinal = E(end);

    mudancas = find(diff(E)~=0)+1;

    primeira = find(E~=E(1),1,'first');

    tf = NaN;
    erro_f = NaN;

    if ~isempty(primeira)

        tf = t(primeira);

        erro_f = (tf-te)*1e6;

    end

    status = 'OK';

    if isempty(primeira) || ...
        abs(erro_f)>TS*1e6+1e-6 || ...
        Eevento==Epre || ...
        Eevento~=Efinal || ...
        numel(mudancas)~=1

        status = 'REVISAR';

    end

    fprintf(fe, ...
        '%d,%s,%.12g,%.12g,%.12g,%d,%.12g,%.12g,%.12g,%s\n', ...
        r,grupo,te,tf,erro_f,numel(mudancas), ...
        Epre,Eevento,Efinal,status);

    fprintf('Marcador no evento: %s\n',status);

    fprintf('Erro temporal do marcador: %.9g us\n', ...
        erro_f);

    if ~strcmp(status,'OK')

        error(['Marcador inconsistente no registro %02d. ' ...
               'Interrompendo antes da caracterizacao.'],r);

    end

    EVENT = [
        EVENT;
        r g te tf erro_f numel(mudancas)
    ];

    % --------------------------------------------------------
    % 7.2. Definicao das regioes temporais
    % --------------------------------------------------------

    kt = ke+round(TRANSITORIO*FS);

    MASK = {
        (1:ke-1)';
        (ke:min(kt-1,numel(t)))';
        (kt:numel(t))'
    };

    for q = 1:3

        idx = MASK{q};

        assert(numel(idx)>NREF);

        Y = I(idx,:);

        fprintf('\nRegiao: %s\n',REGIOES{q});

        fprintf('Amostras: %d\n',numel(idx));

        % ----------------------------------------------------
        % 7.3. Correlacoes e defasagens
        % ----------------------------------------------------

        C = corrcoef(Y);

        [lab,rab] = atraso_fases( ...
            Y(:,1),Y(:,2),FS,TREF);

        [lac,rac] = atraso_fases( ...
            Y(:,1),Y(:,3),FS,TREF);

        [lbc,rbc] = atraso_fases( ...
            Y(:,2),Y(:,3),FS,TREF);

        cor_linha = [
            C(1,2),C(1,3),C(2,3), ...
            lab*1000,lac*1000,lbc*1000, ...
            rab,rac,rbc
        ];

        fprintf(fc,'%d,%s,%s,', ...
            r,grupo,REGIOES{q});

        gravar_linha(fc,cor_linha);

        COR = [
            COR;
            r g q cor_linha
        ];

        fprintf('Correlacao AB/AC/BC: %.5f %.5f %.5f\n', ...
            C(1,2),C(1,3),C(2,3));

        fprintf('Atrasos AB/AC/BC: %.5f %.5f %.5f ms\n', ...
            lab*1000,lac*1000,lbc*1000);

        % ----------------------------------------------------
        % 7.4. Periodicidade, pulsos e retencao
        % ----------------------------------------------------

        for p = 1:3

            x = Y(:,p);

            [Tp,Rp] = periodo_acf(x,FS,TREF);

            [larg,inter,picos] = pulsos( ...
                x,FS,LIMIARES(p),PULSO_MIN);

            inter_ms = inter*1000;

            if isempty(inter_ms)

                med_inter = NaN;
                mad_inter = NaN;

            else

                med_inter = median(inter_ms);

                mad_inter = median(abs( ...
                    inter_ms-med_inter));

            end

            fprintf(fp, ...
                '%d,%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%d\n', ...
                r,grupo,REGIOES{q},FASES{p}, ...
                Tp*1000,1/Tp,Rp,med_inter, ...
                mad_inter,numel(inter));

            PER = [
                PER;
                r g q p Tp*1000 Rp ...
                med_inter mad_inter numel(inter)
            ];

            for j = 1:numel(larg)

                fprintf(fu, ...
                    '%d,%s,%s,%s,%d,%.12g,%.12g\n', ...
                    r,grupo,REGIOES{q},FASES{p}, ...
                    j,larg(j)*1000,picos(j));

                PUL = [
                    PUL;
                    r g q p j larg(j)*1000 picos(j)
                ];

            end

            fprintf(['Fase %s | T_ACF: %.6f ms | ' ...
                     'ACF: %.6f | pulsos: %d\n'], ...
                FASES{p},Tp*1000,Rp,numel(larg));

            for h = 1:numel(TOLS_RETENCAO)

                tol = TOLS_RETENCAO(h);

                [dur,inter_mud,frac] = retencao(x,tol);

                md = NaN;
                med = NaN;
                p95 = NaN;
                f8 = NaN;
                fm8 = NaN;
                fi8 = NaN;
                fim8 = NaN;

                if ~isempty(dur)

                    md = mode(dur);

                    med = median(dur);

                    p95 = percentil(dur,95);

                    f8 = mean(dur==8);

                    fm8 = mean(mod(dur,8)==0);

                end

                if ~isempty(inter_mud)

                    fi8 = mean(inter_mud==8);

                    fim8 = mean(mod(inter_mud,8)==0);

                end

                z = [
                    tol frac numel(dur) ...
                    md med p95 f8 fm8 fi8 fim8
                ];

                fprintf(fr,'%d,%s,%s,%s,', ...
                    r,grupo,REGIOES{q},FASES{p});

                gravar_linha(fr,z);

                RET = [
                    RET;
                    r g q p h z
                ];

            end

        end

        % ----------------------------------------------------
        % 7.5. Janelas trifasicas de referencia
        % ----------------------------------------------------
        %
        % Janelas nao sobrepostas.
        % Nenhuma janela atravessa uma fronteira de regiao.
        % O pre-evento e ancorado no instante do evento.
        % Transitorio e tardio comecam em suas fronteiras.
        % As janelas sao de periodo nominal, e nao
        % necessariamente ciclos eletricos completos.

        nw = floor(numel(idx)/NREF);

        sobra = mod(numel(idx),NREF);

        if q==1
            inicio = sobra+1;
        else
            inicio = 1;
        end

        fprintf('Janelas completas: %d\n',nw);

        fprintf('Amostras residuais excluidas: %d\n',sobra);

        for w = 1:nw

            a = inicio+(w-1)*NREF;

            b = a+NREF-1;

            ids = idx(a:b);

            Z = I(ids,:);

            R = sqrt(mean(Z.^2,1));

            Rtri = sqrt(mean(R.^2));

            deseq = 100*std(R,0)/max(mean(R),eps);

            tc = (mean(t(ids))-te)*1000;

            TRI = [
                TRI;
                r g q w tc R Rtri deseq
            ];

            fprintf(ft,'%d,%s,%s,%d,', ...
                r,grupo,REGIOES{q},w);

            gravar_linha(ft,[tc R Rtri deseq]);

            xv = [];

            for p = 1:3

                x = Z(:,p);

                rms_x = rms_local(x);

                media_x = mean(x);

                desvio_x = std(x);

                pico_x = max(abs(x));

                pp_x = max(x)-min(x);

                energia_x = sum(x.^2)*TS;

                crista_x = pico_x/max(rms_x,eps);

                derivada_x = rms_local(diff(x)/TS);

                atividade_x = 100*mean(x>=LIMIARES(p));

                psd_x = bandas_psd(x,FS);

                f = [
                    rms_x media_x desvio_x ...
                    pico_x pp_x energia_x ...
                    crista_x derivada_x atividade_x ...
                    psd_x
                ];

                assert(all(isfinite(f)));

                xv = [xv f];

                W = [
                    W;
                    r g q w p tc f Rtri deseq
                ];

                fprintf(fw,'%d,%s,%s,%d,%s,', ...
                    r,grupo,REGIOES{q},w,FASES{p});

                gravar_linha(fw,[tc f Rtri deseq]);

            end

            X = [
                X;
                xv Rtri deseq
            ];

            META = [
                META;
                r g q w tc
            ];

        end

    end

    % --------------------------------------------------------
    % 7.6. Exemplo local para figuras
    % --------------------------------------------------------

    if r==1

        DEMO.t = t;
        DEMO.I = I;
        DEMO.te = te;

        DEMO.ke = ke;

        DEMO.indice_pre = MASK{1};

    end

    fprintf('\nRegistro %02d concluido.\n',r);

    clear t I E Y Z;

end

fclose(fw);
fclose(fp);
fclose(fu);
fclose(fr);
fclose(fc);
fclose(fe);
fclose(ft);

assert(all(isfinite(X(:))));
assert(size(X,2)==41);
assert(size(X,1)==size(META,1));

assert(~any(ismember(META(:,1),REG_TESTE)));

% ============================================================
% 8. ESTATISTICAS DESCRITIVAS
% ============================================================

fd = fopen(fullfile(OUT, ...
    'G4_E02_2_Estatisticas.csv'),'w');

assert(fd>=0);

fprintf(fd, ...
    ['grupo,regiao,fase,caracteristica,' ...
     'n,media,dp,mediana,mad,minimo,' ...
     'p05,p25,p75,p95,maximo\n']);

GRUPOS = {'treino','validacao'};

nomes_f = [
    NOMES_BASE;
    {'RMS_Trifasico'};
    {'Desequilibrio'}
];

fprintf('\n==============================================\n');
fprintf(' ESTATISTICAS POR REGIME E FASE\n');
fprintf('==============================================\n');

for g = 1:2

    for q = 1:3

        fprintf('\n%s | %s\n', ...
            GRUPOS{g},REGIOES{q});

        for p = 1:3

            m = W(:,2)==g & ...
                W(:,3)==q & ...
                W(:,5)==p;

            A = W(m,7:21);

            fprintf(['Fase %s | n=%d | ' ...
                     'RMS=%.6f +/- %.6f A | ' ...
                     'pico=%.6f A\n'], ...
                FASES{p},size(A,1), ...
                mean(A(:,1)),std(A(:,1)), ...
                mean(A(:,4)));

            for j = 1:numel(nomes_f)

                s = resumo(A(:,j));

                fprintf(fd,'%s,%s,%s,%s,', ...
                    GRUPOS{g},REGIOES{q}, ...
                    FASES{p},nomes_f{j});

                gravar_linha(fd,s);

            end

        end

        % Globais sem triplicacao por fase.

        m = TRI(:,2)==g & TRI(:,3)==q;

        fprintf('RMS trifasico: %.6f +/- %.6f A\n', ...
            mean(TRI(m,9)),std(TRI(m,9)));

        fprintf('Desequilibrio: %.6f +/- %.6f %%\n', ...
            mean(TRI(m,10)),std(TRI(m,10)));

    end

end

fclose(fd);

% ============================================================
% 9. ESTATISTICAS INTRA E ENTRE REGISTROS
% ============================================================

fv = fopen(fullfile(OUT, ...
    'G4_E02_2_Variabilidade_Intra_Entre.csv'),'w');

assert(fv>=0);

fprintf(fv, ...
    ['regiao,caracteristica,n_registros,' ...
     'media_das_medias,dp_entre_medias,' ...
     'dp_intra_pooled,dp_total_janelas,' ...
     'cv_entre_pct,cv_intra_pct\n']);

for q = 1:3

    for j = 1:size(X,2)

        medias = [];
        ss_intra = 0;
        gl_intra = 0;

        for r = REG_TREINO

            m = META(:,1)==r & META(:,3)==q;

            x = X(m,j);

            if isempty(x)
                continue;
            end

            medias(end+1) = mean(x);

            ss_intra = ss_intra+sum( ...
                (x-mean(x)).^2);

            gl_intra = gl_intra+numel(x)-1;

        end

        m = META(:,2)==1 & META(:,3)==q;

        x = X(m,j);

        media = mean(medias);

        dp_entre = std(medias);

        dp_intra = sqrt( ...
            ss_intra/max(gl_intra,1));

        cv_entre = NaN;
        cv_intra = NaN;

        if abs(media)>1e-12

            cv_entre = 100*dp_entre/abs(media);

            cv_intra = 100*dp_intra/abs(media);

        end

        fprintf(fv,'%s,%s,', ...
            REGIOES{q},NOMES_X{j});

        gravar_linha(fv,[
            numel(medias) media dp_entre ...
            dp_intra std(x) cv_entre cv_intra
        ]);

    end

end

fclose(fv);

% ============================================================
% 10. MATRIZES DE COVARIANCIA E CORRELACAO
% ============================================================
%
% Ajuste exclusivamente nos registros de treinamento.
% Matrizes descritivas, nao utilizadas diretamente
% para gerar variantes nesta etapa.

COVS = cell(3,1);

CORRS = cell(3,1);

for q = 1:3

    m = META(:,2)==1 & META(:,3)==q;

    A = X(m,:);

    mu = mean(A,1);
    sd = std(A,0,1);

    ativos = sd>1e-12;

    Z = zeros(size(A));

    Z(:,ativos) = ...
        (A(:,ativos)-mu(ativos))./sd(ativos);

    CV = cov(A);

    CR = NaN(size(CV));

    CR(ativos,ativos) = corrcoef(Z(:,ativos));

    COVS{q} = CV;
    CORRS{q} = CR;

    arquivos = {
        sprintf('G4_E02_2_Cov_%s.csv',REGIOES{q});
        sprintf('G4_E02_2_Corr_%s.csv',REGIOES{q})
    };

    matrizes = {CV;CR};

    for h = 1:2

        fid = fopen(fullfile(OUT,arquivos{h}),'w');

        assert(fid>=0);

        fprintf(fid,'caracteristica');

        for j = 1:numel(NOMES_X)
            fprintf(fid,',%s',NOMES_X{j});
        end

        fprintf(fid,'\n');

        M = matrizes{h};

        for j = 1:size(M,1)

            fprintf(fid,'%s,',NOMES_X{j});

            gravar_linha(fid,M(j,:));

        end

        fclose(fid);

    end

    fprintf('\nCOVARIANCIA: %s\n',REGIOES{q});

    fprintf('Janelas de treino: %d\n',size(A,1));

    fprintf('Caracteristicas ativas: %d/%d\n', ...
        sum(ativos),numel(ativos));

    fprintf('Posto da matriz padronizada: %d\n', ...
        rank(Z));

    fprintf('Dimensao: %d\n',size(Z,2));

end

% ============================================================
% 11. ENVOLTORIAS EMPIRICAS DE TREINAMENTO
% ============================================================
%
% Nao constituem limites fisicos certificados.
% Nao assumir distribuicao gaussiana.
% Validacao nao participa do ajuste.

fl = fopen(fullfile(OUT, ...
    'G4_E02_2_Envoltoria_Treino.csv'),'w');

assert(fl>=0);

fprintf(fl, ...
    ['regiao,caracteristica,mediana,mad,' ...
     'p01,p05,p25,p75,p95,p99,' ...
     'minimo,maximo,n_treino\n']);

LIMITES = cell(3,1);

for q = 1:3

    m = META(:,2)==1 & META(:,3)==q;

    A = X(m,:);

    L = zeros(size(A,2),4);

    for j = 1:size(A,2)

        x = A(:,j);

        med = median(x);

        mad_x = median(abs(x-med));

        p01 = percentil(x,1);
        p05 = percentil(x,5);
        p25 = percentil(x,25);
        p75 = percentil(x,75);
        p95 = percentil(x,95);
        p99 = percentil(x,99);

        fprintf(fl,'%s,%s,', ...
            REGIOES{q},NOMES_X{j});

        gravar_linha(fl,[
            med mad_x p01 p05 p25 p75 ...
            p95 p99 min(x) max(x) numel(x)
        ]);

        L(j,:) = [p01 p99 min(x) max(x)];

    end

    LIMITES{q} = L;

end

fclose(fl);

% ============================================================
% 12. AUDITORIA DA VALIDACAO
% ============================================================

fa = fopen(fullfile(OUT, ...
    'G4_E02_2_Cobertura_Validacao.csv'),'w');

assert(fa>=0);

fprintf(fa, ...
    ['regiao,caracteristica,n_validacao,' ...
     'fracao_dentro_p01_p99,' ...
     'fracao_dentro_min_max,' ...
     'n_abaixo_min,n_acima_max\n']);

COB = [];

for q = 1:3

    L = LIMITES{q};

    m = META(:,2)==2 & META(:,3)==q;

    A = X(m,:);

    for j = 1:size(A,2)

        x = A(:,j);

        tol = 1e-10*max(1,max(abs(L(j,:))));

        dentro_p = x>=L(j,1)-tol & ...
                   x<=L(j,2)+tol;

        dentro_m = x>=L(j,3)-tol & ...
                   x<=L(j,4)+tol;

        z = [
            numel(x),mean(dentro_p), ...
            mean(dentro_m), ...
            sum(x<L(j,3)-tol), ...
            sum(x>L(j,4)+tol)
        ];

        fprintf(fa,'%s,%s,', ...
            REGIOES{q},NOMES_X{j});

        gravar_linha(fa,z);

        COB = [
            COB;
            q j z
        ];

    end

    z = COB(COB(:,1)==q,:);

    fprintf('\nCOBERTURA DA VALIDACAO: %s\n', ...
        REGIOES{q});

    fprintf('Cobertura media P01-P99: %.4f %%\n', ...
        100*mean(z(:,4)));

    fprintf('Cobertura media minimo-maximo: %.4f %%\n', ...
        100*mean(z(:,5)));

end

fclose(fa);

% ============================================================
% 13. RESUMO DA PERIODICIDADE
% ============================================================

fprintf('\n==============================================\n');
fprintf(' PERIODICIDADE - TREINAMENTO\n');
fprintf('==============================================\n');

for q = 1:3

    fprintf('\nRegiao: %s\n',REGIOES{q});

    for p = 1:3

        m = PER(:,2)==1 & ...
            PER(:,3)==q & ...
            PER(:,4)==p;

        z = PER(m,:);

        fprintf(['Fase %s | T_ACF: %.9f +/- %.9f ms | ' ...
                 'ACF media: %.6f\n'], ...
            FASES{p},mean(z(:,5)), ...
            std(z(:,5)),mean(z(:,6)));

        fprintf('Periodo por pulsos mediano: %.9f ms\n', ...
            median(z(isfinite(z(:,7)),7)));

    end

end

% ============================================================
% 14. RESUMO DA RETENCAO
% ============================================================

fprintf('\n==============================================\n');
fprintf(' INVESTIGACAO DE RETENCAO TEMPORAL\n');
fprintf('==============================================\n');

for q = 1:3

    fprintf('\nRegiao: %s\n',REGIOES{q});

    for h = 1:numel(TOLS_RETENCAO)

        m = RET(:,2)==1 & ...
            RET(:,3)==q & ...
            RET(:,5)==h;

        z = RET(m,:);

        pesos = z(:,8);

        total = sum(pesos);

        valid = pesos>0 & isfinite(z(:,12));

        if any(valid)

            f8 = sum(pesos(valid).*z(valid,12))/ ...
                sum(pesos(valid));

            fm8 = sum(pesos(valid).*z(valid,13))/ ...
                sum(pesos(valid));

        else

            f8 = NaN;
            fm8 = NaN;

        end

        fprintf(['Tol=%.1e A | n=%d | ' ...
                 'retencao 8=%.4f%% | ' ...
                 'multiplos de 8=%.4f%%\n'], ...
            TOLS_RETENCAO(h),total, ...
            100*f8,100*fm8);

    end

end

% ============================================================
% 15. FIGURA 01 - RMS POR REGIME
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    M = zeros(3,2);
    D = zeros(3,2);

    for p = 1:3

        for g = 1:2

            m = W(:,2)==g & ...
                W(:,3)==q & ...
                W(:,5)==p;

            M(p,g) = mean(W(m,7));

            D(p,g) = std(W(m,7));

        end

    end

    bar(M,'grouped');

    hold on;

    for g = 1:2

        offset = (g-1.5)*0.28;

        errorbar((1:3)+offset, ...
            M(:,g),D(:,g),'k.');

    end

    grid on;

    set(gca,'xtick',1:3,'xticklabel',FASES);

    ylabel('RMS (A)');

    title(sprintf('Regiao: %s',REGIOES{q}));

    if q==1
        legend({'Treino','Validacao'}, ...
            'location','northwest');
    end

end

exportar(fig,FIG,'G4_E02_2_01_RMS_Regimes.png');

% ============================================================
% 16. FIGURA 02 - EVOLUCAO TEMPORAL
% ============================================================

fig = figure('position',[100 100 1300 1000]);

for p = 1:3

    subplot(3,1,p);

    hold on;

    for r = REG_TREINO

        m = TRI(:,1)==r;

        A = sortrows(TRI(m,:),5);

        plot(A(:,5),A(:,5+p), ...
            'linewidth',1.1);

    end

    yl = ylim();

    plot([0 0],yl,'k--','linewidth',1.2);

    plot([200 200],yl,'k:','linewidth',1.2);

    grid on;

    ylabel('RMS (A)');

    title(sprintf('Fase %s - Treinamento',FASES{p}));

    if p==3
        xlabel('Tempo relativo ao evento (ms)');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_02_Evolucao_RMS.png');

% ============================================================
% 17. FIGURA 03 - DESEQUILIBRIO
% ============================================================

fig = figure('position',[100 100 1200 650]);

hold on;

for r = REG_TREINO

    m = TRI(:,1)==r;

    A = sortrows(TRI(m,:),5);

    plot(A(:,5),A(:,10),'linewidth',1.2);

end

yl = ylim();

plot([0 0],yl,'k--');

plot([200 200],yl,'k:');

grid on;

xlabel('Tempo relativo ao evento (ms)');

ylabel('Desequilibrio RMS (%)');

title('Evolucao do desequilibrio trifasico');

exportar(fig,FIG, ...
    'G4_E02_2_03_Desequilibrio.png');

% ============================================================
% 18. FIGURA 04 - PERIODICIDADE
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    M = NaN(numel(REG_TREINO),3);

    for j = 1:numel(REG_TREINO)

        for p = 1:3

            m = PER(:,1)==REG_TREINO(j) & ...
                PER(:,3)==q & PER(:,4)==p;

            M(j,p) = PER(m,5);

        end

    end

    plot(REG_TREINO,M(:,1),'o-');

    hold on;

    plot(REG_TREINO,M(:,2),'s-');

    plot(REG_TREINO,M(:,3),'^-');

    plot([min(REG_TREINO) max(REG_TREINO)], ...
        [TREF TREF]*1000,'k--');

    grid on;

    ylabel('Periodo (ms)');

    title(REGIOES{q});

    set(gca,'xtick',REG_TREINO);

    if q==1

        legend({'Fase A','Fase B','Fase C', ...
                'Referencia'}, ...
            'location','northeast');

    end

    if q==3
        xlabel('Registro');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_04_Periodicidade.png');

% ============================================================
% 19. FIGURA 05 - LARGURAS DE ATIVIDADE
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    hold on;

    for p = 1:3

        m = PUL(:,2)==1 & ...
            PUL(:,3)==q & ...
            PUL(:,4)==p;

        x = PUL(m,6);

        if isempty(x)
            continue;
        end

        plot(p*ones(size(x)),x,'.', ...
            'markersize',9);

        plot(p,median(x),'kd', ...
            'markersize',7, ...
            'markerfacecolor','k');

    end

    set(gca,'xtick',1:3,'xticklabel',FASES);

    xlim([0.5 3.5]);

    grid on;

    ylabel('Largura (ms)');

    title(sprintf('Atividade acima do limiar - %s', ...
        REGIOES{q}));

end

exportar(fig,FIG, ...
    'G4_E02_2_05_Largura_Atividade.png');

% ============================================================
% 20. FIGURA 06 - CORRELACOES ENTRE FASES
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    m = COR(:,2)==1 & COR(:,3)==q;

    z = COR(m,:);

    bar(z(:,1),z(:,4:6),'grouped');

    grid on;

    ylim([-1 1]);

    ylabel('Correlacao');

    title(REGIOES{q});

    set(gca,'xtick',REG_TREINO);

    if q==1
        legend({'AB','AC','BC'}, ...
            'location','northeast');
    end

    if q==3
        xlabel('Registro');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_06_Correlacao_Fases.png');

% ============================================================
% 21. FIGURA 07 - DEFASAGENS
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    m = COR(:,2)==1 & COR(:,3)==q;

    z = COR(m,:);

    bar(z(:,1),z(:,7:9),'grouped');

    hold on;

    plot([min(REG_TREINO)-0.5 max(REG_TREINO)+0.5], ...
        [TREF/3 TREF/3]*1000,'k--');

    plot([min(REG_TREINO)-0.5 max(REG_TREINO)+0.5], ...
        [-TREF/3 -TREF/3]*1000,'k--');

    grid on;

    ylabel('Atraso (ms)');

    title(REGIOES{q});

    set(gca,'xtick',REG_TREINO);

    if q==1
        legend({'AB','AC','BC'}, ...
            'location','northeast');
    end

    if q==3
        xlabel('Registro');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_07_Defasagens.png');

% ============================================================
% 22. FIGURA 08 - RETENCOES DE OITO AMOSTRAS
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    M = NaN(numel(TOLS_RETENCAO),3);

    for h = 1:numel(TOLS_RETENCAO)

        for p = 1:3

            m = RET(:,2)==1 & ...
                RET(:,3)==q & ...
                RET(:,4)==p & ...
                RET(:,5)==h;

            z = RET(m,:);

            valid = z(:,8)>0 & ...
                    isfinite(z(:,12));

            if any(valid)

                pesos = z(valid,8);

                M(h,p) = 100*sum( ...
                    pesos.*z(valid,12))/sum(pesos);

            end

        end

    end

    bar(M,'grouped');

    grid on;

    ylim([0 100]);

    ylabel('Retencoes de 8 amostras (%)');

    title(REGIOES{q});

    set(gca,'xtick',1:numel(TOLS_RETENCAO), ...
        'xticklabel',{'0','1e-8','1e-6','1e-4','1e-3'});

    if q==1
        legend(FASES,'location','northwest');
    end

    if q==3
        xlabel('Tolerancia de corrente (A)');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_08_Retencao_8_Amostras.png');

% ============================================================
% 23. FIGURA 09 - HISTOGRAMA DE RETENCOES
% ============================================================

if isfield(DEMO,'I')

    fig = figure('position',[100 100 1200 950]);

    for p = 1:3

        subplot(3,1,p);

        x = DEMO.I(DEMO.indice_pre,p);

        [dur,~,~] = retencao(x,1e-6);

        cont = zeros(33,1);

        for k = 1:32
            cont(k) = sum(dur==k);
        end

        cont(33) = sum(dur>32);

        cont = 100*cont/max(numel(dur),1);

        bar(1:33,cont);

        grid on;

        xlim([0 34]);

        set(gca,'xtick',[1 4 8 12 16 24 32 33], ...
            'xticklabel', ...
            {'1','4','8','12','16','24','32','>32'});

        ylabel('Ocorrencias (%)');

        title(sprintf('Registro 01 - Pre - Fase %s', ...
            FASES{p}));

        if p==3
            xlabel('Duracao da retencao (amostras)');
        end

    end

    exportar(fig,FIG, ...
        'G4_E02_2_09_Histograma_Retencoes.png');

end

% ============================================================
% 24. FIGURA 10 - AUTOCORRELACAO DOS INCREMENTOS
% ============================================================

if isfield(DEMO,'I')

    fig = figure('position',[100 100 1200 950]);

    for p = 1:3

        subplot(3,1,p);

        x = DEMO.I(DEMO.indice_pre,p);

        dx = diff(x);

        dx = dx-mean(dx);

        [a,l] = xcorr(dx,80,'coeff');

        m = l>0;

        stem(l(m),a(m),'filled');

        grid on;

        hold on;

        yl = ylim();

        for k = 8:8:80

            plot([k k],yl,'r--', ...
                'linewidth',0.6);

        end

        ylabel('Autocorrelacao');

        title(sprintf('Incrementos - Fase %s', ...
            FASES{p}));

        if p==3
            xlabel('Atraso (amostras)');
        end

    end

    exportar(fig,FIG, ...
        'G4_E02_2_10_ACF_Incrementos.png');

end

% ============================================================
% 25. FIGURAS 11-13 - MATRIZES DE CORRELACAO
% ============================================================

for q = 1:3

    fig = figure('position',[100 100 1100 950]);

    C = CORRS{q};

    D = C;

    D(~isfinite(D)) = 0;

    imagesc(D,[-1 1]);

    axis square;

    colorbar;

    colormap(jet(256));

    hold on;

    for k = [13.5 26.5 39.5]

        plot([k k],[0.5 41.5],'k-', ...
            'linewidth',1.5);

        plot([0.5 41.5],[k k],'k-', ...
            'linewidth',1.5);

    end

    set(gca, ...
        'xtick',[7 20 33 40.5], ...
        'xticklabel',{'Fase A','Fase B','Fase C','Globais'}, ...
        'ytick',[7 20 33 40.5], ...
        'yticklabel',{'Fase A','Fase B','Fase C','Globais'});

    title(sprintf( ...
        'Correlacao das caracteristicas - %s', ...
        REGIOES{q}));

    exportar(fig,FIG,sprintf( ...
        'G4_E02_2_%02d_Correlacao_%s.png', ...
        10+q,REGIOES{q}));

end

% ============================================================
% 26. FIGURA 14 - VARIABILIDADE DE AMPLITUDE
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    hold on;

    for p = 1:3

        m = W(:,2)==1 & ...
            W(:,3)==q & ...
            W(:,5)==p;

        x = W(m,7);

        y = W(m,10);

        plot(x,y,'.','markersize',10);

    end

    grid on;

    ylabel('Pico (A)');

    title(REGIOES{q});

    if q==1
        legend(FASES,'location','northeast');
    end

    if q==3
        xlabel('RMS (A)');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_14_RMS_vs_Pico.png');

% ============================================================
% 27. FIGURA 15 - COBERTURA DA VALIDACAO
% ============================================================

fig = figure('position',[100 100 1300 950]);

for q = 1:3

    subplot(3,1,q);

    m = COB(:,1)==q;

    z = COB(m,:);

    bar(z(:,2),100*z(:,4:5),'grouped');

    grid on;

    ylim([0 105]);

    xlim([0.5 41.5]);

    ylabel('Cobertura (%)');

    title(REGIOES{q});

    if q==1
        legend({'P01-P99','Min-Max'}, ...
            'location','southwest');
    end

    if q==3
        xlabel('Indice da caracteristica');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_15_Cobertura_Validacao.png');

% ============================================================
% 28. FIGURA 16 - DISTRIBUICAO DA ENERGIA
% ============================================================

fig = figure('position',[100 100 1200 950]);

for q = 1:3

    subplot(3,1,q);

    hold on;

    for p = 1:3

        m = W(:,2)==1 & ...
            W(:,3)==q & ...
            W(:,5)==p;

        x = sort(W(m,12));

        if isempty(x)
            continue;
        end

        Femp = (1:numel(x))'/numel(x);

        stairs(x,Femp,'linewidth',1.3);

    end

    grid on;

    ylabel('Probabilidade acumulada');

    title(sprintf('Energia por janela - %s', ...
        REGIOES{q}));

    if q==1
        legend(FASES,'location','southeast');
    end

    if q==3
        xlabel('Energia de corrente (A^2 s)');
    end

end

exportar(fig,FIG, ...
    'G4_E02_2_16_ECDF_Energia.png');

% ============================================================
% 29. FIGURA 17 - ENERGIA ESPECTRAL POR REGIME
% ============================================================

fig = figure('position',[100 100 1300 950]);

for q = 1:3

    subplot(3,1,q);

    A = zeros(3,4);

    for p = 1:3

        m = W(:,2)==1 & ...
            W(:,3)==q & ...
            W(:,5)==p;

        A(p,:) = median(W(m,16:19),1);

    end

    bar(10*log10(max(A,realmin)),'grouped');

    grid on;

    set(gca,'xtick',1:3,'xticklabel',FASES);

    ylabel('Potencia de corrente (dB re 1 A^2)');

    title(REGIOES{q});

    if q==1

        legend({'0-5 kHz','5-20 kHz', ...
                '20-50 kHz','50-100 kHz'}, ...
            'location','northeast');

    end

end

exportar(fig,FIG, ...
    'G4_E02_2_17_Energia_Espectral.png');

% ============================================================
% 30. RESUMO CONSOLIDADO
% ============================================================

fprintf('\n==============================================\n');
fprintf(' RESUMO CONSOLIDADO\n');
fprintf('==============================================\n');

fprintf('Registros processados: %d\n', ...
    numel(unique(META(:,1))));

fprintf('Janelas trifasicas totais: %d\n', ...
    size(X,1));

fprintf('Janelas de treinamento: %d\n', ...
    sum(META(:,2)==1));

fprintf('Janelas de validacao: %d\n', ...
    sum(META(:,2)==2));

fprintf('Linhas individuais por fase: %d\n', ...
    size(W,1));

fprintf('Caracteristicas trifasicas: %d\n', ...
    size(X,2));

fprintf('Pulsos operacionais identificados: %d\n', ...
    size(PUL,1));

fprintf('Linhas de periodicidade: %d\n', ...
    size(PER,1));

fprintf('Linhas de retencao: %d\n', ...
    size(RET,1));

fprintf('NaN nas caracteristicas: %d\n', ...
    sum(isnan(X(:))));

fprintf('Inf nas caracteristicas: %d\n', ...
    sum(isinf(X(:))));

fprintf('\nDISTRIBUICAO DAS JANELAS\n');

for g = 1:2

    for q = 1:3

        fprintf('%s | %s: %d\n', ...
            GRUPOS{g},REGIOES{q}, ...
            sum(META(:,2)==g & META(:,3)==q));

    end

end

fprintf('\nAUDITORIA DOS MARCADORES\n');

fprintf('Marcadores aprovados: %d/%d\n', ...
    size(EVENT,1),numel(REGISTROS));

fprintf('Erro temporal maximo: %.12g us\n', ...
    max(abs(EVENT(:,5))));

% ============================================================
% 31. PROTOCOLO
% ============================================================

fid = fopen(fullfile(OUT, ...
    'G4_E02_2_Protocolo.txt'),'w');

assert(fid>=0);

fprintf(fid,'GATE 4 - EXPERIMENTO 02.2\n\n');

fprintf(fid,'Fs nominal: %.0f Hz\n',FS);

fprintf(fid,'Ts: %.12g s\n',TS);

fprintf(fid,'Periodo de referencia: %.12g s\n',TREF);

fprintf(fid,'Amostras por janela: %d\n',NREF);

fprintf(fid,'Regiao transitoria: %.12g s\n', ...
    TRANSITORIO);

fprintf(fid,'Treinamento: ');
fprintf(fid,'%02d ',REG_TREINO);

fprintf(fid,'\nValidacao: ');
fprintf(fid,'%02d ',REG_VALID);

fprintf(fid,'\nTeste reservado: 06 12\n');

fprintf(fid,'\nRegularizacao: ultima amostra coincidente.\n');

fprintf(fid,'Tolerancia temporal: 5 ns.\n');

fprintf(fid,'Interpolacao: nenhuma.\n');

fprintf(fid,'Marcadores discretos auditados.\n');

fprintf(fid,'Janelas nao sobrepostas.\n');

fprintf(fid,'Nenhuma janela cruza fronteiras de regiao.\n');

fprintf(fid,['Janelas com duracao nominal, nao ' ...
             'necessariamente ciclos eletricos completos.\n']);

fprintf(fid,['Busca ACF restrita a 85-115%% ' ...
             'do periodo de referencia.\n']);

fprintf(fid,['Periodos ACF com baixa correlacao ' ...
             'devem ser interpretados com cautela.\n']);

fprintf(fid,['Atrasos calculados por maximo da ' ...
             'correlacao cruzada em +/- Tref/2.\n']);

fprintf(fid,['Atrasos pos-evento nao equivalem ' ...
             'necessariamente a defasagem eletrica.\n']);

fprintf(fid,['Larguras de atividade por limiar ' ...
             'na corrente, nao por sinal de gate.\n']);

fprintf(fid,'Limiar de atividade: %.4f%%\n', ...
    FRAC_LIMIAR*100);

fprintf(fid,'Limiares A/B/C: %.12g %.12g %.12g A\n', ...
    LIMIARES);

fprintf(fid,'Duracao minima de atividade: %.12g s\n', ...
    PULSO_MIN/FS);

fprintf(fid,['Limiares ajustados exclusivamente ' ...
             'nos registros de treinamento.\n']);

fprintf(fid,['Covariancia e envoltorias empiricas ' ...
             'calculadas exclusivamente no treinamento.\n']);

fprintf(fid,['Caracteristicas globais repetidas no ' ...
             'CSV por fase, mas nao na matriz trifasica.\n']);

fprintf(fid,['Percentis nao constituem limites ' ...
             'fisicos certificados.\n']);

fprintf(fid,['Variabilidade entre janelas nao ' ...
             'implica independencia estatistica.\n']);

fprintf(fid,['Variabilidade intra e entre registros ' ...
             'avaliada separadamente.\n']);

fprintf(fid,['Retencoes de oito amostras investigadas ' ...
             'em cinco tolerancias.\n']);

fprintf(fid,['Frequencia de atualizacao de 25 kHz ' ...
             'constitui hipotese, nao conclusao.\n']);

fprintf(fid,['Energia de corrente expressa em A^2 s, ' ...
             'nao energia eletrica em joules.\n']);

fprintf(fid,['Nenhuma referencia saudavel ' ...
             'independente disponivel.\n']);

fprintf(fid,['Pre e pos-evento sao rotulos temporais ' ...
             'operacionais.\n']);

fprintf(fid,'Nenhuma variante sintetica gerada.\n');

fprintf(fid,'Nenhum modelo treinado.\n');

fprintf(fid,'Nenhum arquivo MAT derivado salvo.\n');

fprintf(fid,'Nenhum arquivo original modificado.\n');

fprintf(fid,'Nenhum registro reservado acessado.\n');

fclose(fid);

% ============================================================
% 32. LIMPEZA E ENCERRAMENTO
% ============================================================

clear DEMO PICOS_TREINO;

set(0,'defaultfigurevisible','on');

fprintf('\n==============================================\n');
fprintf(' G4/E02.2 - EXECUCAO CONCLUIDA\n');
fprintf('==============================================\n');

fprintf('Caracterizacao estatistica: CONCLUIDA\n');

fprintf('Regularizacao temporal: CONCLUIDA\n');

fprintf('Auditoria dos marcadores: CONCLUIDA\n');

fprintf('Periodicidade: ANALISADA\n');

fprintf('Retencao temporal: ANALISADA\n');

fprintf('Correlacoes e covariancias: ANALISADAS\n');

fprintf('Registros 06 e 12: PRESERVADOS\n');

fprintf('Arquivos originais: PRESERVADOS\n');

fprintf('Variantes sinteticas geradas: ZERO\n');

fprintf('Figuras previstas: 17\n');

fprintf('\nDiretorio dos resultados:\n%s\n',OUT);

fprintf('\nAguardando interpretacao cientifica.\n');

fprintf('==============================================\n');
