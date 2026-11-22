
% ============================================================
% G4_E02_4.m
% GATE 4 - EXPERIMENTO 02.4
% SIMILARIDADE, REFERENCIAS PERIODICAS E ANALISE DE RESIDUOS
%
% GNU Octave 11.3.0
% Representacao canonica: 25 kHz
%
% Nenhuma variante definitiva sera gerada.
% Nenhum modelo de IA sera treinado.
% Registros 06 e 12 permanecem reservados.
% ============================================================

1;
clear;
clc;
close all;

pkg load signal;

% ============================================================
% FUNCOES AUXILIARES
% ============================================================

function v = rms_local(x)
  v = sqrt(mean(x.^2,1));
endfunction


function v = percentil(x,p)

  x = sort(x(isfinite(x)));
  n = numel(x);

  if n == 0
    v = NaN;
    return;
  endif

  k = 1+(n-1)*p/100;

  a = floor(k);
  b = ceil(k);

  v = x(a)+(k-a)*(x(b)-x(a));

endfunction


function s = estatisticas(x)

  x = x(isfinite(x));

  if isempty(x)
    s = NaN(1,12);
    return;
  endif

  m = median(x);

  s = [numel(x), ...
       mean(x), ...
       std(x), ...
       m, ...
       median(abs(x-m)), ...
       rms_local(x), ...
       min(x), ...
       percentil(x,1), ...
       percentil(x,5), ...
       percentil(x,95), ...
       percentil(x,99), ...
       max(x)];

endfunction


function exportar(fig,pasta,nome)

  arquivo = fullfile(pasta,[nome ".png"]);

  set(fig,"paperpositionmode","auto");

  print(fig,arquivo,"-dpng","-r150");

  fprintf("Figura exportada: %s.png\n",nome);

  close(fig);

endfunction


function fig = figura(w,h)

  fig = figure("visible","off", ...
               "position",[50 50 w h], ...
               "color",[1 1 1]);

endfunction


function linha_csv(fid,v)

  fprintf(fid,"%.12g",v(1));

  for k = 2:numel(v)
    fprintf(fid,",%.12g",v(k));
  endfor

  fprintf(fid,"\n");

endfunction


function C = corr_segura(X)

  C = corrcoef(X);

endfunction


function [tu,Xu,Eu,audit] = regularizar(t,X,E,fs)

  t = t(:);
  E = E(:);

  assert(rows(X)==numel(t));
  assert(columns(X)==3);
  assert(numel(E)==numel(t));

  assert(all(isfinite(t)));
  assert(all(isfinite(X(:))));
  assert(all(isfinite(E)));
  assert(all(diff(t)>0));

  Ts = 1/fs;

  Nu = round((t(end)-t(1))/Ts)+1;

  tu = t(1)+(0:Nu-1)'*Ts;

  k = round((t-t(1))/Ts)+1;

  candidatos = find(k>=1 & k<=Nu);

  candidatos = candidatos( ...
      abs(t(candidatos)-tu(k(candidatos))) ...
      <= 0.001*Ts);

  contagem = accumarray( ...
      k(candidatos), ...
      ones(numel(candidatos),1), ...
      [Nu 1],@sum,0);

  assert(all(contagem>0), ...
         "Existem pontos ausentes na grade nominal.");

  idx = accumarray( ...
      k(candidatos),candidatos, ...
      [Nu 1],@max,0);

  assert(all(idx>0));

  Xu = X(idx,:);
  Eu = E(idx);

  audit = [numel(t),Nu, ...
           numel(t)-Nu, ...
           sum(contagem==0), ...
           sum(contagem>1)];

  assert(Nu==110000);

endfunction


function [t25,X25,err] = reduzir_zoh(t,X)

  assert(mod(rows(X),8)==0);

  idx = 1:8:rows(X);

  t25 = t(idx);

  X25 = X(idx,:);

  Xrec = repelem(X25,8,1);

  D = X-Xrec;

  err = max(abs(D(:)));

  assert(err==0, ...
         "Reconstrucao ZOH nao foi exata.");

endfunction


function [P,R,borda] = estimar_periodo(X,fs,Tref)

  N0 = round(fs*Tref);

  lo = round(0.85*N0);
  hi = round(1.15*N0);

  P = NaN(1,3);
  R = NaN(1,3);
  borda = zeros(1,3);

  for p = 1:3

    x = X(:,p)-mean(X(:,p));

    if rms_local(x)<=eps || numel(x)<=hi
      continue;
    endif

    [c,l] = xcorr(x,hi,"coeff");

    sel = find(l>=lo & l<=hi);

    [R(p),j] = max(c(sel));

    k = sel(j);

    lag = l(k);

    if j==1 || j==numel(sel)

      borda(p)=1;

    elseif k>1 && k<numel(c)

      a = c(k-1);
      b = c(k);
      d = c(k+1);

      den = a-2*b+d;

      if abs(den)>eps
        delta = 0.5*(a-d)/den;
        lag = lag+max(-0.5,min(0.5,delta));
      endif

    endif

    P(p)=lag;

  endfor

endfunction


function Y = ciclo_normalizado(X,inicio,P,Nref)

  pos = inicio+(0:Nref-1)'*P/Nref;

  assert(min(pos)>=1);
  assert(max(pos)<=rows(X));

  Y = interp1((1:rows(X))', ...
              X,pos,"linear");

endfunction


function [Y,desloc,erro] = alinhar_ciclo(X,ref,L)

  N = rows(ref);

  assert(rows(X)==N);
  assert(columns(X)==3);

  escala = sqrt(mean(ref.^2,1));

  escala = max(escala,1e-6);

  melhor = Inf;

  Y = X;
  desloc = 0;

  for k = -L:L

    Z = circshift(X,k,1);

    D = bsxfun(@rdivide,Z-ref,escala);

    erro_k = mean(D(:).^2);

    if erro_k<melhor

      melhor=erro_k;

      Y=Z;

      desloc=k;

    endif

  endfor

  erro = sqrt(melhor);

endfunction


function ref = referencia_balanceada(C,ids,Nref,L)

  assert(!isempty(ids));

  % Cada registro recebe o mesmo peso.
  % Ciclos de um registro nao sao confundidos
  % com registros independentes.

  M = zeros(Nref,3,numel(ids));

  inicial = C{ids(1)}(:,:,1);

  for iter = 1:3

    for j = 1:numel(ids)

      A = C{ids(j)};

      S = zeros(Nref,3);

      for k = 1:size(A,3)

        Y = alinhar_ciclo(A(:,:,k),inicial,L);

        S = S+Y;

      endfor

      M(:,:,j)=S/size(A,3);

    endfor

    inicial = median(M,3);

  endfor

  ref = inicial;

endfunction


function [acf,lags,tau,lag05] = dependencia(x,L,fs)

  x = x(:)-mean(x);

  tau=NaN;
  lag05=NaN;

  if rms_local(x)<1e-12 || numel(x)<=L

    acf=NaN(L+1,1);
    lags=(0:L)'/fs;

    return;

  endif

  [c,l]=xcorr(x,L,"coeff");

  acf=c(l>=0);
  acf=acf(:);

  lags=(0:L)'/fs;

  j=find(acf(2:end)<=exp(-1),1);

  if !isempty(j)
    tau=j/fs;
  endif

  j=find(abs(acf(2:end))<=0.05,1);

  if !isempty(j)
    lag05=j/fs;
  endif

endfunction


function b = psd_bandas(x,fs)

  x=x(:)-mean(x);

  N=numel(x);

  bordas=[0 100 500 1000 5000 fs/2];

  b=zeros(1,5);

  if N<32
    b(:)=NaN;
    return;
  endif

  nw=min(512,N);

  nf=max(1024,2^nextpow2(nw));

  [P,f]=pwelch(x,hamming(nw),0.5,nf,fs);

  df=f(2)-f(1);

  for k=1:5

    if k<5
      sel=f>=bordas(k) & f<bordas(k+1);
    else
      sel=f>=bordas(k) & f<=bordas(k+1);
    endif

    b(k)=sum(P(sel))*df;

  endfor

endfunction


function c = cobertura(x,lo,hi)

  if isempty(x) || !isfinite(lo) || !isfinite(hi)
    c=NaN;
    return;
  endif

  tol=1e-10*max([1 abs(lo) abs(hi)]);

  c=100*mean(x>=lo-tol & x<=hi+tol);

endfunction


function r = regime_nome(q)

  nomes={"pre","transitorio","tardio"};

  r=nomes{q};

endfunction


% ============================================================
% CONFIGURACAO
% ============================================================

ROOT = ["PASTA_DO_USUARIO/Documents/" ...
        "Pesquisa_SRM/TesteNo1_31OCT2026/" ...
        "Data and Scripts"];

RAW = fullfile(ROOT,"Results_1_to_12");

assert(exist(RAW,"dir")==7, ...
       sprintf("Pasta inexistente: %s",RAW));

TREINO = [1 2 4 5 7 8 10 11];

VALIDACAO = [3 9];

TESTE = [6 12];

REGISTROS = sort([TREINO VALIDACAO]);

assert(isempty(intersect(REGISTROS,TESTE)));

EVENTOS = [1.0017 1.0019 1.0021 1.0024 ...
           1.0026 1.0030 1.0040 1.0050 ...
           1.0060 1.0070 1.0075 1.0080];

FS_RAW = 200000;

FS = 25000;

TS = 1/FS;

TREF = 0.01308;

NREF = round(TREF*FS);

TRANSITORIO = 0.200;

NTRANS = round(TRANSITORIO*FS);

LALIGN = round(0.05*NREF);

ACF_L = 3*NREF;

ACF_MIN = 0.70;

FASES = {"A","B","C"};

CORES = [0.10 0.35 0.80;
         0.85 0.20 0.15;
         0.10 0.60 0.30];

rand("state",42);
randn("state",42);

run=1;

while true

  OUT=fullfile(ROOT, ...
      sprintf("Gate4_Exp02_4_Run%02d",run));

  if exist(OUT,"dir")!=7
    break;
  endif

  run=run+1;

endwhile

mkdir(OUT);

FIG=fullfile(OUT,"Figuras");

mkdir(FIG);

graphics_toolkit("gnuplot");

set(0,"defaultfigurevisible","off");

diary(fullfile(OUT,"G4_E02_4_Console.txt"));

fprintf("\n========================================\n");
fprintf(" GATE 4 - EXPERIMENTO 02.4\n");
fprintf(" SIMILARIDADE E ANALISE DE RESIDUOS\n");
fprintf("========================================\n");

fprintf("Fs original: %d Hz\n",FS_RAW);
fprintf("Fs canonica: %d Hz\n",FS);
fprintf("Janela de referencia: %d amostras\n",NREF);
fprintf("Alinhamento maximo: %d amostras\n",LALIGN);
fprintf("Registros de treino: %d\n",numel(TREINO));
fprintf("Registros de validacao: %d\n",numel(VALIDACAO));
fprintf("Registros 06 e 12: RESERVADOS\n");
fprintf("Saida: %s\n",OUT);


% ============================================================
% E02.4.0 - CARREGAMENTO E AUDITORIA
% ============================================================

D=cell(12,1);

CICLOS=cell(12,3);

RES=cell(12,3);

REFS=cell(12,3);

ALINHAMENTOS=cell(12,3);

REGIOES=cell(12,3);

PERIODOS=NaN(12,3,3);

QUALIDADE=NaN(12,3,3);

BORDAS=NaN(12,3,3);

ERROS=NaN(12,3,3);

NRMSE=NaN(12,3,3);

fid=fopen(fullfile(OUT, ...
         "G4_E02_4_Integridade.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,n_original,n_regular," ...
  "n_removidas,n_ausentes,n_colisoes,n_25k," ...
  "erro_zoh_A,evento_nominal_s,evento_marcador_s," ...
  "primeiro_update_pos_s,offset_update_us\n"]);

for r=REGISTROS

  assert(!ismember(r,TESTE));

  arquivo=fullfile(RAW, ...
                   sprintf("results_%d.mat",r));

  assert(exist(arquivo,"file")==2, ...
         sprintf("Arquivo nao encontrado: %s",arquivo));

  S=load(arquivo,"results");

  assert(isfield(S,"results"));

  t=S.results.time(:);

  X=S.results.signals(3).values;

  E=S.results.signals(6).values(:);

  clear S;

  if rows(X)==3 && columns(X)==numel(t)
    X=X';
  endif

  [tu,Xu,Eu,audit]=regularizar(t,X,E,FS_RAW);

  trans=find(diff(Eu)!=0)+1;

  assert(numel(trans)==1, ...
         "Marcador possui numero inesperado de transicoes.");

  te=tu(trans(1));

  assert(abs(te-EVENTOS(r))<=1/FS_RAW+1e-10);

  [t25,X25,ez]=reduzir_zoh(tu,Xu);

  ke=find(t25>=te-1e-10,1,"first");

  assert(!isempty(ke));

  Epre=Eu(1);
  Epos=Eu(end);

  assert(Epre!=Epos);
  assert(Eu(trans(1))==Epos);

  kt=find(t25>=te+TRANSITORIO-1e-10, ...
          1,"first");

  assert(!isempty(kt));
  assert(kt>ke);

  D{r}=struct("t",t25, ...
              "X",X25, ...
              "te",te, ...
              "ke",ke, ...
              "kt",kt, ...
              "grupo",1+ismember(r,VALIDACAO));

  REGIOES{r,1}=X25(1:ke-1,:);
  REGIOES{r,2}=X25(ke:kt-1,:);
  REGIOES{r,3}=X25(kt:end,:);

  linha_csv(fid,[r,D{r}.grupo,audit, ...
                 rows(X25),ez,EVENTOS(r),te, ...
                 t25(ke),1e6*(t25(ke)-te)]);

  fprintf("\nR%02d | grupo %d\n",r,D{r}.grupo);

  fprintf("Amostras: %d -> %d -> %d\n", ...
          audit(1),audit(2),rows(X25));

  fprintf("Erro ZOH: %.12g A\n",ez);

  fprintf("Offset evento/update: %.6f us\n", ...
          1e6*(t25(ke)-te));

  clear t X E tu Xu Eu X25 t25;

endfor

fclose(fid);


% ============================================================
% E02.4.1 - SIMILARIDADE PRE-EVENTO
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.1 - SIMILARIDADE PRE-EVENTO\n");
fprintf("========================================\n");

SIM=NaN(12,12,3);

SIMMAX=NaN(12,12,3);

EXATO=NaN(12,12,3);

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Similaridade_Pre.csv"),"w");

fprintf(fid, ...
 ["registro_i,registro_j,grupo_i,grupo_j,fase," ...
  "n_comum,rmse_A,nrmse_pct,mae_A,max_A," ...
  "fracao_igual,correlacao\n"]);

for a=1:numel(REGISTROS)

  i=REGISTROS(a);

  for b=a:numel(REGISTROS)

    j=REGISTROS(b);

    assert(abs(D{i}.t(1)-D{j}.t(1))<1e-10);

    N=min(rows(REGIOES{i,1}), ...
          rows(REGIOES{j,1}));

    Xi=REGIOES{i,1}(1:N,:);

    Xj=REGIOES{j,1}(1:N,:);

    for p=1:3

      x=Xi(:,p);
      y=Xj(:,p);

      e=x-y;

      rm=rms_local(e);

      nr=100*rm/max(rms_local(x),eps);

      ma=mean(abs(e));

      mx=max(abs(e));

      eq=mean(e==0);

      c=corrcoef(x,y);

      cc=c(1,2);

      SIM(i,j,p)=rm;
      SIM(j,i,p)=rm;

      SIMMAX(i,j,p)=mx;
      SIMMAX(j,i,p)=mx;

      EXATO(i,j,p)=eq;
      EXATO(j,i,p)=eq;

      linha_csv(fid,[i,j,D{i}.grupo,D{j}.grupo, ...
                     p,N,rm,nr,ma,mx,eq,cc]);

    endfor

  endfor

endfor

fclose(fid);

for p=1:3

  M=SIM(TREINO,TREINO,p);

  Q=EXATO(TREINO,TREINO,p);

  sel=triu(true(numel(TREINO)),1);

  fprintf("\nFase %s - pares de treino\n",FASES{p});

  fprintf("RMSE mediano: %.12g A\n", ...
          median(M(sel)));

  fprintf("RMSE maximo: %.12g A\n", ...
          max(M(sel)));

  fprintf("Fracao identica mediana: %.10f %%\n", ...
          100*median(Q(sel)));

endfor

% Numero de trajetorias pre-evento distintas
% considerando o suporte temporal comum.

NCOMUM=Inf;

for r=TREINO

  NCOMUM=min(NCOMUM,rows(REGIOES{r,1}));

endfor

REPRESENTANTES=[];

GRUPOS_EXATOS=zeros(12,1);

for r=TREINO

  encontrado=false;

  Xr=REGIOES{r,1}(1:NCOMUM,:);

  for j=1:numel(REPRESENTANTES)

    s=REPRESENTANTES(j);

    Xs=REGIOES{s,1}(1:NCOMUM,:);

    if isequal(Xr,Xs)

      GRUPOS_EXATOS(r)=j;

      encontrado=true;

      break;

    endif

  endfor

  if !encontrado

    REPRESENTANTES(end+1)=r;

    GRUPOS_EXATOS(r)=numel(REPRESENTANTES);

  endif

endfor

fprintf("\nAmostras no suporte comum do treino: %d\n", ...
        NCOMUM);

fprintf("Trajetorias pre-evento exatamente distintas: %d/%d\n", ...
        numel(REPRESENTANTES),numel(TREINO));

fprintf("Nao confundir trajetorias distintas com independencia fisica.\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Grupos_Pre_Exatos.csv"),"w");

fprintf(fid,"registro,grupo_trajetoria,n_comum\n");

for r=TREINO
  linha_csv(fid,[r,GRUPOS_EXATOS(r),NCOMUM]);
endfor

fclose(fid);


% ============================================================
% E02.4.2 - PERIODICIDADE POR REGIME
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.2 - PERIODICIDADE\n");
fprintf("========================================\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Periodicidade.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,regiao,fase,periodo_amostras," ...
  "periodo_ms,acf,borda,aceito\n"]);

for r=REGISTROS

  for q=1:3

    X=REGIOES{r,q};

    [P,R,B]=estimar_periodo(X,FS,TREF);

    PERIODOS(r,q,:)=reshape(P,1,1,3);

    QUALIDADE(r,q,:)=reshape(R,1,1,3);

    BORDAS(r,q,:)=reshape(B,1,1,3);

    for p=1:3

      aceito=isfinite(P(p)) && ...
             !B(p) && R(p)>=ACF_MIN && q!=2;

      linha_csv(fid,[r,D{r}.grupo,q,p,P(p), ...
                     1000*P(p)/FS,R(p),B(p),aceito]);

    endfor

  endfor

endfor

fclose(fid);

% Periodo comum trifasico estimado por registro.
% Somente pre-evento e tardio.
% O transitorio nao recebe modelo estacionario.

PERIODO_COMUM=NaN(12,3);

for r=REGISTROS

  for q=[1 3]

    p=reshape(PERIODOS(r,q,:),1,3);

    a=reshape(QUALIDADE(r,q,:),1,3);

    b=reshape(BORDAS(r,q,:),1,3);

    ok=isfinite(p) & a>=ACF_MIN & b==0;

    if sum(ok)<2

      fprintf("R%02d | %s: periodicidade rejeitada\n", ...
              r,regime_nome(q));

      continue;

    endif

    PERIODO_COMUM(r,q)=median(p(ok));

    fprintf("R%02d | %s | T comum: %.8f ms\n", ...
            r,regime_nome(q), ...
            1000*PERIODO_COMUM(r,q)/FS);

  endfor

endfor


% ============================================================
% E02.4.3 - EXTRACAO DOS CICLOS
%
% Normalizacao de fase empregada exclusivamente
% para diagnostico de residuos.
% Nao substitui os sinais canonicos originais.
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.3 - EXTRACAO DOS CICLOS\n");
fprintf("========================================\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Ciclos.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,regiao,periodo_amostras," ...
  "n_ciclos,n_pontos_normalizados\n"]);

for r=REGISTROS

  for q=[1 3]

    P=PERIODO_COMUM(r,q);

    if !isfinite(P)
      continue;
    endif

    X=REGIOES{r,q};

    N=rows(X);

    K=floor((N-1)/P);

    if K<2
      continue;
    endif

    A=zeros(NREF,3,K);

    % Todos os canais usam a mesma coordenada de fase.

    for k=1:K

      inicio=1+(k-1)*P;

      A(:,:,k)=ciclo_normalizado(X,inicio,P,NREF);

    endfor

    CICLOS{r,q}=A;

    linha_csv(fid,[r,D{r}.grupo,q,P,K,NREF]);

    fprintf("R%02d | %s | %d ciclos completos\n", ...
            r,regime_nome(q),K);

  endfor

endfor

fclose(fid);


% ============================================================
% E02.4.4 - REFERENCIAS PERIODICAS CROSS-FIT
%
% Treino: leave-one-record-out.
% Pre-evento: excluir tambem trajetorias exatamente
% identicas ao registro investigado.
%
% Validacao: referencia exclusivamente de treino.
%
% Caso nao existam doadores independentes em termos
% de trajetoria, o resultado sera omitido.
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.4 - REFERENCIAS E RESIDUOS\n");
fprintf("========================================\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Alinhamento.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,regiao,ciclo,n_doadores," ...
  "desloc_amostras,desloc_us,erro_normalizado," ...
  "rmse_A_A,rmse_B_A,rmse_C_A\n"]);

REF_GLOBAL=cell(1,3);


% Referencias globais ajustadas somente no treino.

for q=[1 3]

  disponiveis=[];

  for s=TREINO

    if !isempty(CICLOS{s,q})
      disponiveis(end+1)=s;
    endif

  endfor

  if isempty(disponiveis)
    continue;
  endif

  REF_GLOBAL{q}=referencia_balanceada( ...
      CICLOS(:,q),disponiveis,NREF,LALIGN);

endfor

for r=REGISTROS

  for q=[1 3]

    A=CICLOS{r,q};

    if isempty(A)
      continue;
    endif

    doadores=[];

    for s=TREINO

      if s==r || isempty(CICLOS{s,q})
        continue;
      endif

      % Evita que uma copia exata do trecho pre-evento
      % seja usada como referencia cross-fit.

      if q==1 && ismember(r,TREINO)

        if GRUPOS_EXATOS(s)==GRUPOS_EXATOS(r)
          continue;
        endif

      endif

      doadores(end+1)=s;

    endfor

    if isempty(doadores)

      fprintf(["R%02d | %s | sem doadores elegiveis; " ...
               "residuo cross-fit omitido\n"], ...
               r,regime_nome(q));

      continue;

    endif

    ref=referencia_balanceada( ...
        CICLOS(:,q),doadores,NREF,LALIGN);

    REFS{r,q}=ref;

    K=size(A,3);

    R=zeros(NREF,3,K);

    shifts=zeros(K,1);

    for k=1:K

      [Y,d,en]=alinhar_ciclo( ...
          A(:,:,k),ref,LALIGN);

      R(:,:,k)=Y-ref;

      shifts(k)=d;

      rm=rms_local(R(:,:,k));

      linha_csv(fid,[r,D{r}.grupo,q,k, ...
                     numel(doadores),d,d*TS*1e6,en,rm]);

    endfor

    RES{r,q}=R;

    ALINHAMENTOS{r,q}=shifts;

    % Preserva a separacao entre ciclos para calculo
    % de autocorrelacao e estatisticas temporais.

    M=reshape(permute(R,[1 3 2]),[],3);

    EA=rms_local(M);

    SX=reshape(permute(A,[1 3 2]),[],3);

    NA=100*EA./max(rms_local(SX),eps);

    ERROS(r,q,:)=reshape(EA,1,1,3);

    NRMSE(r,q,:)=reshape(NA,1,1,3);

    fprintf("\nR%02d | %s | doadores: %d\n", ...
            r,regime_nome(q),numel(doadores));

    fprintf("RMSE residual [A]: %.8f %.8f %.8f\n",EA);

    fprintf("NRMSE residual [%%]: %.8f %.8f %.8f\n",NA);

    fprintf("Deslocamento medio absoluto [us]: %.4f\n", ...
            mean(abs(shifts))*TS*1e6);

    fprintf("Alinhamentos no limite: %d/%d\n", ...
            sum(abs(shifts)==LALIGN),K);

  endfor

endfor

fclose(fid);


% ============================================================
% E02.4.5 - REFERENCIA TRANSITORIA
%
% Coordenada temporal relativa ao evento.
% Sem alinhamento independente das fases.
% Sem ajuste utilizando validacao.
%
% Diferencas de fase no instante da falha permanecem
% como parte da variabilidade entre trajetorias.
% Nao interpretar esses residuos como ruido branco.
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.5 - REFERENCIA TRANSITORIA\n");
fprintf("========================================\n");

% Grade comum iniciada 40 us apos o evento.
% Evita extrapolacao quando o evento ocorre entre
% dois instantes de atualizacao.

tau=(1:NTRANS-1)'*TS;

NT=numel(tau);

TRANS_ALINHADO=cell(12,1);

for r=REGISTROS

  t=D{r}.t-D{r}.te;

  X=D{r}.X;

  Z=interp1(t,X,tau,"previous");

  assert(all(isfinite(Z(:))));

  TRANS_ALINHADO{r}=Z;

endfor

M=zeros(NT,3,numel(TREINO));

for j=1:numel(TREINO)

  M(:,:,j)=TRANS_ALINHADO{TREINO(j)};

endfor

REF_GLOBAL{2}=median(M,3);

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Residuos_Transitorio.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,fase,n_doadores,rmse_A," ...
  "nrmse_pct,media_A,desvio_A,pico_abs_A\n"]);

for r=REGISTROS

  doadores=TREINO(TREINO!=r);

  assert(!isempty(doadores));

  M=zeros(NT,3,numel(doadores));

  for j=1:numel(doadores)

    M(:,:,j)=TRANS_ALINHADO{doadores(j)};

  endfor

  ref=median(M,3);

  R=TRANS_ALINHADO{r}-ref;

  REFS{r,2}=ref;

  RES{r,2}=R;

  ea=rms_local(R);

  na=100*ea./max(rms_local(TRANS_ALINHADO{r}),eps);

  ERROS(r,2,:)=reshape(ea,1,1,3);

  NRMSE(r,2,:)=reshape(na,1,1,3);

  for p=1:3

    linha_csv(fid,[r,D{r}.grupo,p,numel(doadores), ...
                   ea(p),na(p),mean(R(:,p)), ...
                   std(R(:,p)),max(abs(R(:,p)))]);

  endfor

  fprintf("R%02d | RMSE [A]: %.8f %.8f %.8f\n", ...
          r,ea);

endfor

fclose(fid);


% ============================================================
% E02.4.6 - ESTATISTICAS E DEPENDENCIA DOS RESIDUOS
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.6 - ESTATISTICAS DOS RESIDUOS\n");
fprintf("========================================\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Estatisticas_Residuos.csv"),"w");

fprintf(fid, ...
 ["registro,grupo,regiao,fase,n,media_A,dp_A," ...
  "mediana_A,mad_A,rms_A,min_A,p01_A,p05_A," ...
  "p95_A,p99_A,max_A,nrmse_pct\n"]);

fc=fopen(fullfile(OUT, ...
          "G4_E02_4_Correlacoes_Residuos.csv"),"w");

fprintf(fc, ...
 "registro,grupo,regiao,corr_AB,corr_AC,corr_BC\n");

fa=fopen(fullfile(OUT, ...
          "G4_E02_4_ACF_Residuos.csv"),"w");

fprintf(fa, ...
 ["registro,grupo,regiao,fase,segmento," ...
  "tau_e_ms,lag_005_ms,acf_lag1\n"]);

fp=fopen(fullfile(OUT, ...
          "G4_E02_4_PSD_Residuos.csv"),"w");

fprintf(fp, ...
 ["registro,grupo,regiao,fase,segmento," ...
  "psd_0_100,psd_100_500,psd_500_1000," ...
  "psd_1000_5000,psd_5000_12500\n"]);

POOL=cell(2,3,3);

TAUS=cell(3,3);

ACF_MEDIA=cell(3,3);

for r=REGISTROS

  for q=1:3

    R=RES{r,q};

    if isempty(R)
      continue;
    endif

    nr=size(R,3);

    M=reshape(permute(R,[1 3 2]),[],3);

    g=D{r}.grupo;

    C=corr_segura(M);

    linha_csv(fc,[r,g,q,C(1,2),C(1,3),C(2,3)]);

    fprintf("\nR%02d | %s\n",r,regime_nome(q));

    fprintf("Correlacoes residuais: %.6f %.6f %.6f\n", ...
            C(1,2),C(1,3),C(2,3));

    for p=1:3

      x=M(:,p);

      POOL{g,q,p}=[POOL{g,q,p};x];

      st=estatisticas(x);

      linha_csv(fid,[r,g,q,p,st,NRMSE(r,q,p)]);

      for k=1:nr

        z=R(:,p,k);

        LL=min(ACF_L,numel(z)-2);

        [ac,lag,ta,l05]=dependencia(z,LL,FS);

        linha_csv(fa,[r,g,q,p,k, ...
                     1000*ta,1000*l05,ac(2)]);

        bb=psd_bandas(z,FS);

        linha_csv(fp,[r,g,q,p,k,bb]);

        if g==1 && isfinite(ta)

          TAUS{q,p}=[TAUS{q,p};ta];

        endif

        if g==1

          % ACF limitada a um ciclo, sem unir
          % artificialmente ciclos consecutivos.

          L=min(NREF-2,numel(ac)-1);

          aa=ac(1:L+1);

          if all(isfinite(aa))

            ACF_MEDIA{q,p}=[ACF_MEDIA{q,p},aa];

          endif

        endif

      endfor

    endfor

  endfor

endfor

fclose(fid);
fclose(fc);
fclose(fa);
fclose(fp);


% ============================================================
% E02.4.7 - ENVOLTORIA DE TREINO E AUDITORIA
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.7 - ENVOLTORIAS EMPIRICAS\n");
fprintf("========================================\n");

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Envoltoria_Residuos.csv"),"w");

fprintf(fid, ...
 ["regiao,fase,n_treino,n_registros_treino," ...
  "n_validacao,media_A,dp_A,mediana_A,mad_A," ...
  "p01_A,p99_A,min_A,max_A," ...
  "cobertura_val_p01_p99_pct," ...
  "cobertura_val_minmax_pct,tau_mediana_ms\n"]);

ENV=NaN(3,3,4);

for q=1:3

  for p=1:3

    x=POOL{1,q,p};

    y=POOL{2,q,p};

    if isempty(x)

      fprintf("%s | Fase %s | sem residuos cross-fit\n", ...
              regime_nome(q),FASES{p});

      continue;

    endif

    L=percentil(x,1);
    U=percentil(x,99);

    mn=min(x);
    mx=max(x);

    cv=cobertura(y,L,U);

    cm=cobertura(y,mn,mx);

    tau_med=NaN;

    if !isempty(TAUS{q,p})
      tau_med=median(TAUS{q,p});
    endif

    nreg=0;

    for r=TREINO

      if !isempty(RES{r,q})
        nreg=nreg+1;
      endif

    endfor

    ENV(q,p,:)=reshape([L U mn mx],1,1,4);

    linha_csv(fid,[q,p,numel(x),nreg,numel(y), ...
                   mean(x),std(x),median(x), ...
                   median(abs(x-median(x))), ...
                   L,U,mn,mx,cv,cm,1000*tau_med]);

    fprintf("\n%s | Fase %s\n", ...
            regime_nome(q),FASES{p});

    fprintf("Residuos de treino: %d\n",numel(x));

    fprintf("Registros contribuintes: %d\n",nreg);

    fprintf("P01/P99: %.8f / %.8f A\n",L,U);

    fprintf("Min/max: %.8f / %.8f A\n",mn,mx);

    fprintf("Cobertura validacao P01-P99: %.4f %%\n",cv);

    fprintf("Cobertura validacao min-max: %.4f %%\n",cm);

    fprintf("Tempo de correlacao mediano: %.6f ms\n", ...
            1000*tau_med);

  endfor

endfor

fclose(fid);


% ============================================================
% E02.4.8 - VARIABILIDADE INTER-REGISTROS
%
% Limites empiricos para futuras perturbacoes.
% Nenhum limite derivado da validacao.
% Nenhuma perturbacao aplicada neste experimento.
% ============================================================

fprintf("\n========================================\n");
fprintf(" E02.4.8 - LIMITES DE PERTURBACAO\n");
fprintf("========================================\n");

AMP=cell(3,3);

PER=cell(1,3);

PULSE=cell(3,3);

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Variabilidade_Registros.csv"),"w");

fprintf(fid, ...
 ["registro,regiao,fase,rms_A,pico_A," ...
  "energia_A2s,media_A,periodo_ms\n"]);

for r=TREINO

  for q=1:3

    X=REGIOES{r,q};

    P=PERIODO_COMUM(r,q);

    for p=1:3

      x=X(:,p);

      a=rms_local(x);

      pk=max(abs(x));

      energia=sum(x.^2)*TS;

      AMP{q,p}=[AMP{q,p};a];

      PULSE{q,p}=[PULSE{q,p};pk];

      linha_csv(fid,[r,q,p,a,pk,energia, ...
                     mean(x),1000*P/FS]);

    endfor

    if isfinite(P)

      PER{q}=[PER{q};1000*P/FS];

    endif

  endfor

endfor

fclose(fid);

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Limites_Candidatos.csv"),"w");

fprintf(fid, ...
 ["regiao,fase,n_registros,rms_mediano_A," ...
  "rms_min_A,rms_max_A,rms_cv_pct," ...
  "ganho_relativo_min,ganho_relativo_max," ...
  "pico_min_A,pico_max_A,periodo_mediano_ms," ...
  "periodo_min_ms,periodo_max_ms\n"]);

for q=1:3

  for p=1:3

    x=AMP{q,p};

    pk=PULSE{q,p};

    m=median(x);

    cv=100*std(x)/max(mean(x),eps);

    gmin=min(x)/max(m,eps);

    gmax=max(x)/max(m,eps);

    pp=[NaN NaN NaN];

    if !isempty(PER{q})

      pp=[median(PER{q}),min(PER{q}),max(PER{q})];

    endif

    linha_csv(fid,[q,p,numel(x),m,min(x),max(x), ...
                   cv,gmin,gmax,min(pk),max(pk),pp]);

    fprintf("\n%s | Fase %s\n", ...
            regime_nome(q),FASES{p});

    fprintf("CV RMS entre registros: %.6f %%\n",cv);

    fprintf("Ganho relativo empirico: %.8f a %.8f\n", ...
            gmin,gmax);

    fprintf("Periodo mediano: %.8f ms\n",pp(1));

  endfor

endfor

fclose(fid);

% ------------------------------------------------------------
% Auditoria da variabilidade temporal dentro do transitorio.
% Segmentos de 20 ms.
% Mantem a dependencia com o tempo relativo ao evento.
% ------------------------------------------------------------

PASSO=round(0.020*FS);

NB=floor(NT/PASSO);

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Envoltoria_Transitorio_Temporal.csv"),"w");

fprintf(fid, ...
 ["segmento,inicio_ms,fim_ms,fase,n_registros," ...
  "rms_residual_mediano_A,rms_residual_min_A," ...
  "rms_residual_max_A,p01_A,p99_A\n"]);

for b=1:NB

  idx=(b-1)*PASSO+(1:PASSO);

  for p=1:3

    a=[];

    z=[];

    for r=TREINO

      x=RES{r,2}(idx,p);

      a=[a;rms_local(x)];

      z=[z;x];

    endfor

    linha_csv(fid,[b,1000*tau(idx(1)), ...
                   1000*tau(idx(end)),p,numel(a), ...
                   median(a),min(a),max(a), ...
                   percentil(z,1),percentil(z,99)]);

  endfor

endfor

fclose(fid);


% ============================================================
% FIGURAS
% ============================================================

fprintf("\n========================================\n");
fprintf(" EXPORTACAO DAS FIGURAS\n");
fprintf("========================================\n");


% ------------------------------------------------------------
% FIGURA 01 - Similaridade pre-evento
% ------------------------------------------------------------

f=figura(1450,520);

for p=1:3

  subplot(1,3,p);

  M=SIM(REGISTROS,REGISTROS,p);

  imagesc(M);

  axis image;

  colorbar;

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS, ...
          "ytick",1:numel(REGISTROS), ...
          "yticklabel",REGISTROS);

  xlabel("Registro");
  ylabel("Registro");

  title(sprintf("RMSE pre-evento - fase %s [A]", ...
                FASES{p}));

endfor

colormap(jet(128));

exportar(f,FIG,"G4_E02_4_01_Similaridade_Pre");


% ------------------------------------------------------------
% FIGURA 02 - Fracao de amostras identicas
% ------------------------------------------------------------

f=figura(1450,520);

for p=1:3

  subplot(1,3,p);

  imagesc(100*EXATO(REGISTROS,REGISTROS,p));

  axis image;

  caxis([0 100]);

  colorbar;

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS, ...
          "ytick",1:numel(REGISTROS), ...
          "yticklabel",REGISTROS);

  xlabel("Registro");
  ylabel("Registro");

  title(sprintf("Amostras identicas - fase %s [%%]", ...
                FASES{p}));

endfor

exportar(f,FIG,"G4_E02_4_02_Identidade_Pre");


% ------------------------------------------------------------
% FIGURA 03 - Comparacao direta das trajetorias
% ------------------------------------------------------------

f=figura(1300,950);

N=min(NCOMUM,4*NREF);

tt=(0:N-1)'*TS*1000;

for p=1:3

  subplot(3,1,p);

  hold on;

  for r=TREINO

    plot(tt,REGIOES{r,1}(1:N,p), ...
         "linewidth",0.8);

  endfor

  ylabel("Corrente [A]");

  title(sprintf("Trajetorias de treino - fase %s", ...
                FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo desde inicio do registro [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_03_Trajetorias_Pre");


% ------------------------------------------------------------
% FIGURA 04 - Diferencas amostra a amostra
% ------------------------------------------------------------

f=figura(1300,950);

r0=TREINO(1);

for p=1:3

  subplot(3,1,p);

  hold on;

  for r=TREINO(2:end)

    x=REGIOES{r,1}(1:NCOMUM,p) ...
      -REGIOES{r0,1}(1:NCOMUM,p);

    plot((0:NCOMUM-1)*TS*1000,x, ...
         "linewidth",0.8);

  endfor

  ylabel("Diferenca [A]");

  title(sprintf("Diferencas em relacao a R%02d - fase %s", ...
                r0,FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo desde inicio do registro [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_04_Diferencas_Pre");


% ------------------------------------------------------------
% FIGURA 05 - Referencias periodicas
% ------------------------------------------------------------

f=figura(1300,850);

for j=1:2

  q=[1 3](j);

  subplot(2,1,j);

  hold on;

  if !isempty(REF_GLOBAL{q})

    for p=1:3

      plot((0:NREF-1)/NREF*360, ...
           REF_GLOBAL{q}(:,p), ...
           "color",CORES(p,:), ...
           "linewidth",1.6);

    endfor

    legend(FASES,"location","northeastoutside");

  endif

  title(sprintf("Referencia global de treino - %s", ...
                regime_nome(q)));

  ylabel("Corrente [A]");

  grid on;

endfor

xlabel("Coordenada normalizada do ciclo [graus]");

exportar(f,FIG,"G4_E02_4_05_Referencias_Periodicas");


% ------------------------------------------------------------
% FIGURA 06 - Ciclos alinhados, regime tardio
% ------------------------------------------------------------

f=figura(1300,950);

r0=TREINO(1);

q=3;

for p=1:3

  subplot(3,1,p);

  hold on;

  A=CICLOS{r0,q};

  ref=REFS{r0,q};

  if !isempty(A) && !isempty(ref)

    for k=1:size(A,3)

      Y=alinhar_ciclo(A(:,:,k),ref,LALIGN);

      plot((0:NREF-1)/NREF*360,Y(:,p), ...
           "color",[0.65 0.70 0.78], ...
           "linewidth",0.6);

    endfor

    plot((0:NREF-1)/NREF*360,ref(:,p), ...
         "color",CORES(p,:), ...
         "linewidth",2.0);

  endif

  title(sprintf("R%02d - ciclos tardios - fase %s", ...
                r0,FASES{p}));

  ylabel("Corrente [A]");

  grid on;

  if p==3
    xlabel("Coordenada normalizada do ciclo [graus]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_06_Ciclos_Alinhados");


% ------------------------------------------------------------
% FIGURA 07 - Residuo de um ciclo tardio
% ------------------------------------------------------------

f=figura(1300,950);

for p=1:3

  subplot(3,1,p);

  R=RES{r0,3};

  if !isempty(R)

    plot((0:NREF-1)*TS*1000,R(:,p,1), ...
         "color",CORES(p,:), ...
         "linewidth",1.3);

  endif

  ylabel("Residuo [A]");

  title(sprintf("R%02d - residuo tardio - fase %s", ...
                r0,FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo equivalente no ciclo [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_07_Residuo_Periodico");


% ------------------------------------------------------------
% FIGURA 08 - Referencia transiente e trajetorias
% ------------------------------------------------------------

f=figura(1350,1050);

for p=1:3

  subplot(3,1,p);

  hold on;

  for r=TREINO

    plot(tau*1000,TRANS_ALINHADO{r}(:,p), ...
         "color",[0.70 0.74 0.80], ...
         "linewidth",0.7);

  endfor

  plot(tau*1000,REF_GLOBAL{2}(:,p), ...
       "color",CORES(p,:), ...
       "linewidth",1.8);

  ylabel("Corrente [A]");

  title(sprintf("Trajetorias transitorias - fase %s", ...
                FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo relativo ao evento [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_08_Referencia_Transitoria");


% ------------------------------------------------------------
% FIGURA 09 - Residuos transitorios cross-fit
% ------------------------------------------------------------

f=figura(1350,1050);

for p=1:3

  subplot(3,1,p);

  hold on;

  for r=TREINO

    plot(tau*1000,RES{r,2}(:,p), ...
         "linewidth",0.8);

  endfor

  ylabel("Residuo [A]");

  title(sprintf("Residuos transitorios cross-fit - fase %s", ...
                FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo relativo ao evento [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_09_Residuos_Transitorios");


% ------------------------------------------------------------
% FIGURA 10 - RMSE por registro e regime
% ------------------------------------------------------------

f=figura(1400,1000);

for q=1:3

  subplot(3,1,q);

  Y=reshape(ERROS(REGISTROS,q,:), ...
            numel(REGISTROS),3);

  bar(Y,"grouped");

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS);

  title(sprintf("RMSE residual - %s",regime_nome(q)));

  ylabel("RMSE [A]");

  grid on;

  legend(FASES,"location","northeastoutside");

  if q==3
    xlabel("Registro");
  endif

endfor

exportar(f,FIG,"G4_E02_4_10_RMSE_Residuos");


% ------------------------------------------------------------
% FIGURA 11 - NRMSE por registro e regime
% ------------------------------------------------------------

f=figura(1400,1000);

for q=1:3

  subplot(3,1,q);

  Y=reshape(NRMSE(REGISTROS,q,:), ...
            numel(REGISTROS),3);

  bar(Y,"grouped");

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS);

  title(sprintf("NRMSE residual - %s",regime_nome(q)));

  ylabel("NRMSE [%]");

  grid on;

  legend(FASES,"location","northeastoutside");

  if q==3
    xlabel("Registro");
  endif

endfor

exportar(f,FIG,"G4_E02_4_11_NRMSE_Residuos");


% ------------------------------------------------------------
% FIGURAS 12-14 - Distribuicao dos residuos
% ------------------------------------------------------------

for q=1:3

  f=figura(1400,520);

  for p=1:3

    subplot(1,3,p);

    x=POOL{1,q,p};

    if !isempty(x)

      [h,c]=hist(x,100);

      h=h/sum(h);

      bar(c,h,1, ...
          "facecolor",CORES(p,:), ...
          "edgecolor","none");

    else

      text(0.5,0.5,"Sem residuos cross-fit", ...
           "units","normalized", ...
           "horizontalalignment","center");

    endif

    xlabel("Residuo [A]");

    ylabel("Frequencia relativa");

    title(sprintf("%s - fase %s", ...
                  regime_nome(q),FASES{p}));

    grid on;

  endfor

  exportar(f,FIG, ...
           sprintf("G4_E02_4_%02d_Histograma_%s", ...
                   11+q,regime_nome(q)));

endfor


% ------------------------------------------------------------
% FIGURA 15 - ACF dos residuos
% ------------------------------------------------------------

f=figura(1350,1000);

for q=1:3

  subplot(3,1,q);

  hold on;

  lg={};

  for p=1:3

    M=ACF_MEDIA{q,p};

    if !isempty(M)

      a=mean(M,2);

      plot((0:numel(a)-1)*TS*1000,a, ...
           "color",CORES(p,:), ...
           "linewidth",1.4);

      lg{end+1}=FASES{p};

    endif

  endfor

  if !isempty(lg)

    legend(lg,"location","northeastoutside");

  endif

  xlim([0 1000*(NREF-2)*TS]);

  ylim([-1 1]);

  title(sprintf("ACF media dos residuos - %s", ...
                regime_nome(q)));

  ylabel("Autocorrelacao");

  grid on;

  if q==3
    xlabel("Atraso [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_15_ACF_Residuos");


% ------------------------------------------------------------
% FIGURA 16 - Correlacoes trifasicas residuais
% ------------------------------------------------------------

f=figura(1450,520);

for q=1:3

  subplot(1,3,q);

  M=[];

  for r=TREINO

    R=RES{r,q};

    if !isempty(R)

      M=[M;reshape(permute(R,[1 3 2]),[],3)];

    endif

  endfor

  if !isempty(M)

    C=corrcoef(M);

    imagesc(C);

    axis image;

    caxis([-1 1]);

    colorbar;

    set(gca,"xtick",1:3,"xticklabel",FASES, ...
            "ytick",1:3,"yticklabel",FASES);

    for i=1:3

      for j=1:3

        text(j,i,sprintf("%.3f",C(i,j)), ...
             "horizontalalignment","center", ...
             "color","w", ...
             "fontweight","bold");

      endfor

    endfor

  endif

  title(sprintf("Correlacao residual - %s", ...
                regime_nome(q)));

endfor

exportar(f,FIG,"G4_E02_4_16_Correlacoes_Residuos");


% ------------------------------------------------------------
% FIGURA 17 - Cobertura treino/validacao
% ------------------------------------------------------------

f=figura(1350,1000);

for q=1:3

  subplot(3,1,q);

  B=NaN(3,2);

  for p=1:3

    x=POOL{1,q,p};

    y=POOL{2,q,p};

    if !isempty(x) && !isempty(y)

      B(p,1)=cobertura(y,percentil(x,1), ...
                         percentil(x,99));

      B(p,2)=cobertura(y,min(x),max(x));

    endif

  endfor

  bar(B,"grouped");

  ylim([0 105]);

  set(gca,"xtick",1:3,"xticklabel",FASES);

  ylabel("Cobertura [%]");

  title(sprintf("Envoltoria de treino - %s", ...
                regime_nome(q)));

  legend({"P01-P99","Min-max"}, ...
         "location","northeastoutside");

  grid on;

endfor

exportar(f,FIG,"G4_E02_4_17_Cobertura_Validacao");


% ------------------------------------------------------------
% FIGURA 18 - Variabilidade temporal dos residuos
% ------------------------------------------------------------

f=figura(1350,1000);

for p=1:3

  subplot(3,1,p);

  A=zeros(NB,numel(TREINO));

  for b=1:NB

    idx=(b-1)*PASSO+(1:PASSO);

    for j=1:numel(TREINO)

      A(b,j)=rms_local(RES{TREINO(j),2}(idx,p));

    endfor

  endfor

  tt=((0:NB-1)+0.5)*PASSO*TS*1000;

  hold on;

  fill([tt fliplr(tt)], ...
       [min(A,[],2)' fliplr(max(A,[],2)')], ...
       [0.78 0.84 0.92], ...
       "edgecolor","none");

  plot(tt,median(A,2), ...
       "color",CORES(p,:), ...
       "linewidth",1.8);

  ylabel("RMS residual [A]");

  title(sprintf("Variabilidade transitoria - fase %s", ...
                FASES{p}));

  grid on;

  if p==3
    xlabel("Tempo relativo ao evento [ms]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_18_Variabilidade_Transitoria");


% ------------------------------------------------------------
% FIGURA 19 - Variabilidade RMS entre registros de treino
% ------------------------------------------------------------

f=figura(1350,1000);

for q=1:3

  subplot(3,1,q);

  A=zeros(numel(TREINO),3);

  for p=1:3

    A(:,p)=AMP{q,p};

  endfor

  bar(A,"grouped");

  set(gca,"xtick",1:numel(TREINO), ...
          "xticklabel",TREINO);

  ylabel("RMS [A]");

  title(sprintf("Variabilidade entre registros - %s", ...
                regime_nome(q)));

  legend(FASES,"location","northeastoutside");

  grid on;

  if q==3
    xlabel("Registro de treino");
  endif

endfor

exportar(f,FIG,"G4_E02_4_19_Variabilidade_RMS");


% ============================================================
% PROTOCOLO
% ============================================================

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_Protocolo.txt"),"w");

fprintf(fid,"GATE 4 - EXPERIMENTO 02.4\n\n");

fprintf(fid,"Fs original: %d Hz\n",FS_RAW);

fprintf(fid,"Fs canonica: %d Hz\n",FS);

fprintf(fid,"Periodo de referencia: %.8f s\n",TREF);

fprintf(fid,"Amostras por ciclo normalizado: %d\n",NREF);

fprintf(fid,"Limite alinhamento: %d amostras\n",LALIGN);

fprintf(fid,"ACF minima: %.4f\n",ACF_MIN);

fprintf(fid,"Treino: ");
fprintf(fid,"%02d ",TREINO);
fprintf(fid,"\n");

fprintf(fid,"Validacao: ");
fprintf(fid,"%02d ",VALIDACAO);
fprintf(fid,"\n");

fprintf(fid,"Teste reservado: ");
fprintf(fid,"%02d ",TESTE);
fprintf(fid,"\n\n");

fprintf(fid, ...
 ["Regularizacao por ultima ocorrencia da grade nominal.\n" ...
  "Reconstrucao ZOH exata exigida em todos os registros.\n" ...
  "Representacao canonica extraida a 25 kHz.\n" ...
  "Similaridade pre-evento em suporte temporal comum.\n" ...
  "Trajetorias exatamente iguais agrupadas no treino.\n" ...
  "Referencias periodicas balanceadas por registro.\n" ...
  "Cross-fit por exclusao de registro.\n" ...
  "Exclusao adicional de trajetorias pre exatamente iguais.\n" ...
  "Ausencia de doadores implica omissao do residuo.\n" ...
  "Validacao nunca utilizada no ajuste das referencias.\n" ...
  "Alinhamento conjunto das tres fases.\n" ...
  "Normalizacao de ciclos apenas para diagnostico.\n" ...
  "Transitorio modelado em tempo relativo ao evento.\n" ...
  "Residuos transitorios nao interpretados como ruido.\n" ...
  "ACF calculada sem concatenar artificialmente ciclos.\n" ...
  "Envoltorias empiricas estimadas somente no treino.\n" ...
  "Limites candidatos nao representam validacao fisica.\n" ...
  "Nenhuma variante definitiva produzida.\n" ...
  "Nenhum modelo de IA treinado.\n" ...
  "Arquivos originais preservados.\n" ...
  "Registros 06 e 12 nao carregados.\n"]);

fclose(fid);


% ============================================================
% CONSOLIDACAO
% ============================================================

fprintf("\n========================================\n");
fprintf(" RESULTADOS CONSOLIDADOS\n");
fprintf("========================================\n");

fprintf("Registros processados: %d\n",numel(REGISTROS));

fprintf("Trajetorias pre-evento distintas no treino: %d/%d\n", ...
        numel(REPRESENTANTES),numel(TREINO));

fprintf("Suporte temporal comum: %d amostras\n",NCOMUM);

fprintf("Representacao canonica: %d Hz\n",FS);

for q=1:3

  fprintf("\nREGIME: %s\n",regime_nome(q));

  for p=1:3

    fprintf("\nFase %s\n",FASES{p});

    for g=1:2

      if g==1

        ids=TREINO;

        nome="Treino cross-fit";

      else

        ids=VALIDACAO;

        nome="Validacao";

      endif

      a=reshape(ERROS(ids,q,p),[],1);

      b=reshape(NRMSE(ids,q,p),[],1);

      a=a(isfinite(a));

      b=b(isfinite(b));

      if isempty(a)

        fprintf("%s: sem residuos elegiveis\n",nome);

      else

        fprintf(["%s | n=%d | RMSE medio=%.8f A | " ...
                 "NRMSE medio=%.8f %%\n"], ...
                 nome,numel(a),mean(a),mean(b));

      endif

    endfor

    if !isempty(POOL{1,q,p})

      x=POOL{1,q,p};

      fprintf("MAD residual: %.8f A\n", ...
              median(abs(x-median(x))));

      fprintf("P01-P99: %.8f a %.8f A\n", ...
              percentil(x,1),percentil(x,99));

      if !isempty(POOL{2,q,p})

        fprintf("Cobertura validacao: %.4f %%\n", ...
                cobertura(POOL{2,q,p}, ...
                          percentil(x,1),percentil(x,99)));

      endif

    endif

  endfor

endfor

fprintf("\n========================================\n");
fprintf(" G4/E02.4 - EXECUCAO CONCLUIDA\n");
fprintf("========================================\n");

fprintf("Similaridade pre-evento: ANALISADA\n");
fprintf("Referencias periodicas: ANALISADAS\n");
fprintf("Referencias transitorias: ANALISADAS\n");
fprintf("Residuos cross-fit: ANALISADOS\n");
fprintf("Dependencia temporal: ANALISADA\n");
fprintf("Correlacoes trifasicas: ANALISADAS\n");
fprintf("Limites empiricos: PRODUZIDOS\n");
fprintf("Figuras: EXPORTADAS\n");
fprintf("Registros 06 e 12: PRESERVADOS\n");
fprintf("Arquivos originais: PRESERVADOS\n");
fprintf("Variantes definitivas: ZERO\n");
fprintf("Treinamento de IA: NENHUM\n");

fprintf("\nDiretorio:\n%s\n",OUT);

fprintf("\nAguardando interpretacao cientifica.\n");

fprintf("========================================\n");

diary off;
