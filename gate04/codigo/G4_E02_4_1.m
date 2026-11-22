
% ============================================================
% G4_E02_4_1.m
% GATE 4 - EXPERIMENTO CORRETIVO 02.4.1
%
% ALINHAMENTO CIRCULAR GLOBAL E AUDITORIA DOS RESIDUOS
%
% GNU Octave 11.3.0
%
% Objetivos:
% 1. Comparar alinhamento limitado e alinhamento global.
% 2. Separar variabilidade intraciclo e inter-registros.
% 3. Recalcular referencias periodicas.
% 4. Caracterizar residuos apos alinhamento adequado.
% 5. Comparar RMS em ciclos completos equivalentes.
% 6. Avaliar viabilidade do bootstrap multivariado.
%
% Nenhuma variante definitiva sera produzida.
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

function r = rms_local(x)
  r = sqrt(mean(x.^2,1));
endfunction


function v = pct(x,p)

  x = sort(x(isfinite(x)));

  if isempty(x)
    v = NaN;
    return;
  endif

  k = 1+(numel(x)-1)*p/100;

  a = floor(k);
  b = ceil(k);

  v = x(a)+(k-a)*(x(b)-x(a));

endfunction


function linha(fid,v)

  fprintf(fid,"%.12g",v(1));

  for k=2:numel(v)
    fprintf(fid,",%.12g",v(k));
  endfor

  fprintf(fid,"\n");

endfunction


function f = abrir_csv(pasta,nome,cabecalho)

  f = fopen(fullfile(pasta,nome),"w");

  assert(f>0);

  fprintf(f,"%s\n",cabecalho);

endfunction


function f = nova_figura(w,h)

  f = figure("visible","off", ...
             "position",[40 40 w h], ...
             "color",[1 1 1]);

endfunction


function exportar(f,pasta,nome)

  set(f,"paperpositionmode","auto");

  print(f,fullfile(pasta,[nome ".png"]), ...
        "-dpng","-r150");

  close(f);

  fprintf("Figura exportada: %s.png\n",nome);

endfunction


function nome = nome_regime(q)

  if q==1
    nome="pre";
  else
    nome="tardio";
  endif

endfunction


function [t25,X25,te,ke,kt,erro] = ...
    carregar_registro(arquivo,evento,fs0,fs,TTRANS)

  assert(exist(arquivo,"file")==2, ...
         sprintf("Arquivo inexistente: %s",arquivo));

  S=load(arquivo,"results");

  assert(isfield(S,"results"));

  t=S.results.time(:);

  X=S.results.signals(3).values;

  E=S.results.signals(6).values(:);

  clear S;

  if rows(X)==3 && columns(X)==numel(t)
    X=X';
  endif

  assert(columns(X)==3);
  assert(rows(X)==numel(t));
  assert(numel(E)==numel(t));

  assert(all(isfinite(t)));
  assert(all(isfinite(X(:))));
  assert(all(isfinite(E)));
  assert(all(diff(t)>0));

  Ts0=1/fs0;

  N=round((t(end)-t(1))/Ts0)+1;

  tu=t(1)+(0:N-1)'*Ts0;

  k=round((t-t(1))/Ts0)+1;

  candidatos=find(k>=1 & k<=N);

  candidatos=candidatos( ...
      abs(t(candidatos)-tu(k(candidatos))) ...
      <=0.001*Ts0);

  contagem=accumarray( ...
      k(candidatos), ...
      ones(numel(candidatos),1), ...
      [N 1],@sum,0);

  assert(all(contagem>0), ...
         "Grade nominal incompleta.");

  idx=accumarray(k(candidatos),candidatos, ...
                 [N 1],@max,0);

  assert(all(idx>0));

  Xu=X(idx,:);
  Eu=E(idx);

  assert(N==110000);

  ev=find(diff(Eu)!=0)+1;

  assert(numel(ev)==1);

  te=tu(ev);

  assert(abs(te-evento)<=Ts0+1e-10);

  assert(Eu(1)!=Eu(end));
  assert(Eu(ev)==Eu(end));

  F=round(fs0/fs);

  assert(F==8);
  assert(mod(N,F)==0);

  t25=tu(1:F:end);
  X25=Xu(1:F:end,:);

  Xrec=repelem(X25,F,1);

  erro=max(abs(Xrec(:)-Xu(:)));

  assert(erro==0, ...
         "Reconstrucao ZOH nao exata.");

  ke=find(t25>=te-1e-10,1,"first");

  kt=find(t25>=te+TTRANS-1e-10,1,"first");

  assert(!isempty(ke));
  assert(!isempty(kt));
  assert(kt>ke);

endfunction


function [P,qualidade] = periodo_comum(X,fs,TREF)

  N0=round(fs*TREF);

  lo=round(0.85*N0);
  hi=round(1.15*N0);

  pp=NaN(1,3);
  rr=NaN(1,3);

  for p=1:3

    x=X(:,p)-mean(X(:,p));

    if numel(x)<=hi || std(x)<1e-12
      continue;
    endif

    [c,l]=xcorr(x,hi,"coeff");

    sel=find(l>=lo & l<=hi);

    [v,j]=max(c(sel));

    k=sel(j);

    if j==1 || j==numel(sel) || v<0.70
      continue;
    endif

    lag=l(k);

    a=c(k-1);
    b=c(k);
    d=c(k+1);

    den=a-2*b+d;

    if abs(den)>eps

      delta=0.5*(a-d)/den;

      lag=lag+max(-0.5,min(0.5,delta));

    endif

    pp(p)=lag;
    rr(p)=v;

  endfor

  ok=isfinite(pp);

  if sum(ok)<2

    P=NaN;
    qualidade=NaN;

  else

    P=median(pp(ok));
    qualidade=median(rr(ok));

  endif

endfunction


function [A,INICIOS,RMS_NATIVE] = ...
    extrair_ciclos(X,P,NREF)

  K=floor((rows(X)-1)/P);

  A=zeros(NREF,3,K);

  INICIOS=zeros(K,1);

  RMS_NATIVE=zeros(K,3);

  for k=1:K

    inicio=1+(k-1)*P;

    fim=inicio+P;

    pos=inicio+(0:NREF-1)'*P/NREF;

    assert(min(pos)>=1);
    assert(max(pos)<=rows(X));

    A(:,:,k)=interp1((1:rows(X))', ...
                     X,pos,"linear");

    % RMS calculado tambem na grade original de 25 kHz.
    % Evita confundir interpolacao com variacao fisica.

    ia=max(1,ceil(inicio));

    ib=min(rows(X),ceil(fim)-1);

    RMS_NATIVE(k,:)=rms_local(X(ia:ib,:));

    INICIOS(k)=inicio;

  endfor

endfunction


% ============================================================
% ALINHAMENTO CIRCULAR
%
% O mesmo deslocamento e aplicado a A, B e C.
%
% Nenhuma fase recebe deslocamento independente.
% Nenhum ganho e ajustado durante o alinhamento.
%
% A funcao custo normaliza as contribuicoes das fases,
% impedindo que a fase B domine apenas por amplitude.
%
% A busca global utiliza todas as posicoes circulares.
% ============================================================


function [Y,d,erro,curva] = alinhar(X,ref,L)

  N=rows(ref);

  assert(rows(X)==N);
  assert(columns(X)==3);
  assert(columns(ref)==3);

  % Preserva a normalizacao original das fases.

  escala=max(rms_local(ref),1e-6);

  Xn=bsxfun(@rdivide,X,escala);
  Rn=bsxfun(@rdivide,ref,escala);

  L=min(L,floor(N/2));

  shifts=-L:L;

  if numel(shifts)>N
    shifts=shifts(1:N);
  endif

  % Correlacao circular conjunta via FFT.
  %
  % C(d) = soma de Rn .* circshift(Xn,d)
  %
  % A energia dos sinais permanece constante
  % sob deslocamentos circulares.

  FX=fft(Xn,[],1);
  FR=fft(Rn,[],1);

  C=real(ifft(sum(FR.*conj(FX),2)));

  energia=sum(Xn(:).^2)+sum(Rn(:).^2);

  indices=mod(shifts,N)+1;

  custo=(energia-2*C(indices))/(3*N);

  custo=max(custo(:),0);

  % Identificacao do minimo global ou limitado.

  vmin=min(custo);

  % Reavaliacao direta dos candidatos praticamente
  % empatados para controlar erros numericos da FFT.

  tolerancia=1e-12*max(1,energia/(3*N));

  candidatos=find(custo<=vmin+tolerancia);

  melhor=Inf;
  jmelhor=candidatos(1);

  for k=1:numel(candidatos)

    j=candidatos(k);

    Z=circshift(Xn,shifts(j),1);

    D=Z-Rn;

    J=mean(D(:).^2);

    if J<melhor

      melhor=J;
      jmelhor=j;

    endif

  endfor

  d=shifts(jmelhor);

  Y=circshift(X,d,1);

  % Erro final calculado diretamente, sem
  % cancelamento numerico na formula espectral.

  D=bsxfun(@rdivide,Y-ref,escala);

  erro=sqrt(mean(D(:).^2));

  custo(jmelhor)=erro^2;

  curva=[shifts(:),sqrt(custo)];

endfunction



function J = transformar_escala(X,ref)

  escala=max(rms_local(ref),1e-6);

  D=bsxfun(@rdivide,X-ref,escala);

  J=mean(D(:).^2);

endfunction


% ============================================================
% REFERENCIA BALANCEADA POR REGISTRO
%
% Cada registro tem o mesmo peso.
%
% O refinamento global pode alterar a posicao circular
% da referencia, mas uma ancora fixa evita deriva.
% ============================================================

function [ref,H] = construir_ref(C,ids,NREF,L,MAXIT)

  assert(!isempty(ids));

  ancora=C{ids(1)}(:,:,1);

  ref=ancora;

  H=NaN(MAXIT,3);

  for it=1:MAXIT

    M=zeros(NREF,3,numel(ids));

    custos=zeros(numel(ids),1);

    for j=1:numel(ids)

      A=C{ids(j)};

      K=size(A,3);

      S=zeros(NREF,3);

      J=0;

      for k=1:K

        [Y,d,e]=alinhar(A(:,:,k),ref,L);

        S=S+Y;

        J=J+e^2;

      endfor

      M(:,:,j)=S/K;

      custos(j)=J/K;

    endfor

    nova=median(M,3);

    % Ancora apenas para o algoritmo global.
    % A versao limitada reproduz a metodologia anterior.

    if L>=floor(NREF/2)

      nova=alinhar(nova,ancora,floor(NREF/2));

    endif

    delta=rms_local(nova-ref);

    H(it,:)=[it,mean(custos),mean(delta)];

    ref=nova;

    if it>=3 && mean(delta)<1e-6

      H=H(1:it,:);

      break;

    endif

  endfor

endfunction


function [ref,H] = construir_intra(A,k_excluido,NREF,L)

  ids=setdiff(1:size(A,3),k_excluido);

  assert(!isempty(ids));

  C=cell(1,1);

  C{1}=A(:,:,ids);

  [ref,H]=construir_ref(C,1,NREF,L,5);

endfunction


function M = juntar_ciclos(R)

  M=reshape(permute(R,[1 3 2]),[],3);

endfunction


% ============================================================
% ACF DOS RESIDUOS
%
% ACF estimada separadamente para cada ciclo.
% Nao concatenar ciclos artificialmente.
%
% As unidades temporais dos ciclos normalizados sao
% equivalentes e nao necessariamente tempos fisicos exatos.
% ============================================================

function [a,tau,t05,lag1] = acf_residuo(x,L,Ts)

  x=x(:)-mean(x);

  L=min(L,numel(x)-2);

  a=NaN(L+1,1);

  tau=NaN;
  t05=NaN;
  lag1=NaN;

  if L<1 || rms_local(x)<1e-12
    return;
  endif

  [c,l]=xcorr(x,L,"coeff");

  a=c(l>=0);
  a=a(:);

  lag1=a(2);

  j=find(a(2:end)<=exp(-1),1);

  if !isempty(j)
    tau=j*Ts;
  endif

  j=find(abs(a(2:end))<=0.05,1);

  if !isempty(j)
    t05=j*Ts;
  endif

endfunction


function b = psd_bandas(x,fs)

  x=x(:)-mean(x);

  bordas=[0 100 500 1000 5000 fs/2];

  N=numel(x);

  nw=min(256,N);

  nf=max(1024,2^nextpow2(nw));

  [P,f]=pwelch(x,hamming(nw),0.5,nf,fs);

  df=f(2)-f(1);

  b=zeros(1,5);

  for k=1:5

    if k<5
      sel=f>=bordas(k) & f<bordas(k+1);
    else
      sel=f>=bordas(k) & f<=bordas(k+1);
    endif

    b(k)=sum(P(sel))*df;

  endfor

endfunction


function v = cobertura(x,lo,hi)

  if isempty(x) || !isfinite(lo) || !isfinite(hi)

    v=NaN;

    return;

  endif

  tol=1e-10*max([1 abs(lo) abs(hi)]);

  v=100*mean(x>=lo-tol & x<=hi+tol);

endfunction


function v = cv_local(x)

  v=100*std(x)/max(abs(mean(x)),eps);

endfunction


% ============================================================
% CONFIGURACAO
% ============================================================

ROOT=["PASTA_DO_USUARIO/Documents/" ...
      "Pesquisa_SRM/TesteNo1_31OCT2026/" ...
      "Data and Scripts"];

RAW=fullfile(ROOT,"Results_1_to_12");

assert(exist(RAW,"dir")==7, ...
       sprintf("Pasta inexistente: %s",RAW));

TREINO=[1 2 4 5 7 8 10 11];

VALIDACAO=[3 9];

TESTE=[6 12];

REGISTROS=sort([TREINO VALIDACAO]);

assert(isempty(intersect(REGISTROS,TESTE)));

EVENTOS=[1.0017 1.0019 1.0021 1.0024 ...
         1.0026 1.0030 1.0040 1.0050 ...
         1.0060 1.0070 1.0075 1.0080];

FS0=200000;

FS=25000;

TS=1/FS;

TREF=0.01308;

NREF=round(FS*TREF);

LLOCAL=16;

LGLOBAL=floor(NREF/2);

MAXIT=8;

TTRANS=0.200;

FASES={"A","B","C"};

METODOS={"Local16", ...
         "GlobalRefLocal", ...
         "GlobalRefGlobal", ...
         "IntraLOCO"};

CORES=[0.10 0.35 0.80;
       0.85 0.20 0.15;
       0.10 0.60 0.30];

% Criterios exploratorios definidos antes da execucao.
% Nao representam limites fisicos certificados.
%
% Servem apenas para decidir se vale a pena investigar
% uma futura expansao baseada em residuos.

LIM_NRMSE_MEDIANA=5;

LIM_NRMSE_MAX=10;

LIM_REDUCAO=50;

rand("state",42);
randn("state",42);

run=1;

while true

  OUT=fullfile(ROOT, ...
      sprintf("Gate4_Exp02_4_1_Run%02d",run));

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

diary(fullfile(OUT,"G4_E02_4_1_Console.txt"));

fprintf("\n========================================\n");
fprintf(" GATE 4 - EXPERIMENTO 02.4.1\n");
fprintf(" ALINHAMENTO CIRCULAR GLOBAL\n");
fprintf("========================================\n");

fprintf("Fs original: %d Hz\n",FS0);
fprintf("Fs canonica: %d Hz\n",FS);
fprintf("Amostras por ciclo: %d\n",NREF);

fprintf("Busca limitada: %d posicoes\n", ...
        2*LLOCAL+1);

fprintf("Busca global: %d posicoes\n", ...
        2*LGLOBAL+1);

fprintf("Registros 06 e 12: RESERVADOS\n");

fprintf("Saida: %s\n",OUT);


% ============================================================
% E01 - AUDITORIA E EXTRACAO DOS CICLOS
% ============================================================

D=cell(12,1);

C=cell(12,2);

REG=cell(12,2);

INICIOS=cell(12,2);

NATIVE_RMS=cell(12,2);

PER=NaN(12,2);

QUAL=NaN(12,2);

GRUPO=zeros(12,1);

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Integridade.csv", ...
    ["registro,grupo,n_25k,erro_zoh_A," ...
     "evento_s,offset_update_us"]);

fc=abrir_csv(OUT, ...
    "G4_E02_4_1_Ciclos.csv", ...
    ["registro,grupo,regiao,periodo_amostras," ...
     "periodo_ms,acf,n_ciclos"]);

fprintf("\n========================================\n");
fprintf(" EXTRAINDO CICLOS\n");
fprintf("========================================\n");

for r=REGISTROS

  assert(!ismember(r,TESTE));

  arquivo=fullfile(RAW, ...
                   sprintf("results_%d.mat",r));

  [t,X,te,ke,kt,ez]=carregar_registro( ...
      arquivo,EVENTOS(r),FS0,FS,TTRANS);

  GRUPO(r)=1+ismember(r,VALIDACAO);

  D{r}=struct("t",t,"X",X,"te",te, ...
              "ke",ke,"kt",kt);

  REG{r,1}=X(1:ke-1,:);

  REG{r,2}=X(kt:end,:);

  linha(fid,[r,GRUPO(r),rows(X),ez,te, ...
             1e6*(t(ke)-te)]);

  fprintf("\nR%02d | grupo %d | ZOH: %.12g A\n", ...
          r,GRUPO(r),ez);

  for q=1:2

    [P,Q]=periodo_comum(REG{r,q},FS,TREF);

    assert(isfinite(P), ...
           sprintf("Periodicidade invalida: R%02d",r));

    PER(r,q)=P;

    QUAL(r,q)=Q;

    [A,I,RN]=extrair_ciclos(REG{r,q},P,NREF);

    assert(size(A,3)>=3);

    C{r,q}=A;

    INICIOS{r,q}=I;

    NATIVE_RMS{r,q}=RN;

    linha(fc,[r,GRUPO(r),q,P, ...
              P/FS*1000,Q,size(A,3)]);

    fprintf("%s | T=%.8f ms | ACF=%.6f | ciclos=%d\n", ...
            nome_regime(q),P/FS*1000,Q,size(A,3));

  endfor

endfor

fclose(fid);
fclose(fc);


% ============================================================
% E02 - RECONFIRMACAO DA IDENTIDADE PRE-EVENTO
% ============================================================

fprintf("\n========================================\n");
fprintf(" IDENTIDADE PRE-EVENTO\n");
fprintf("========================================\n");

NCOMUM=Inf;

for r=TREINO

  NCOMUM=min(NCOMUM,rows(REG{r,1}));

endfor

r0=TREINO(1);

X0=REG{r0,1}(1:NCOMUM,:);

REP=[];

GEX=zeros(12,1);

for r=TREINO

  encontrado=false;

  Xr=REG{r,1}(1:NCOMUM,:);

  for j=1:numel(REP)

    s=REP(j);

    if isequal(Xr,REG{s,1}(1:NCOMUM,:))

      GEX(r)=j;

      encontrado=true;

      break;

    endif

  endfor

  if !encontrado

    REP(end+1)=r;

    GEX(r)=numel(REP);

  endif

endfor

fprintf("Suporte comum: %d amostras\n",NCOMUM);

fprintf("Trajetorias distintas: %d/%d\n", ...
        numel(REP),numel(TREINO));

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Identidade_Pre.csv", ...
    ["registro,grupo,n_comum,grupo_exato," ...
     "rmse_A,rmse_B,rmse_C," ...
     "identidade_A_pct,identidade_B_pct," ...
     "identidade_C_pct"]);

for r=REGISTROS

  N=min(NCOMUM,rows(REG{r,1}));

  Xr=REG{r,1}(1:N,:);

  E=Xr-X0(1:N,:);

  identidade=100*mean(E==0,1);

  rr=rms_local(E);

  linha(fid,[r,GRUPO(r),N,GEX(r),rr,identidade]);

  fprintf("R%02d | RMSE comum: %.10g %.10g %.10g A\n", ...
          r,rr);

  fprintf("Identidade: %.8f %.8f %.8f %%\n", ...
          identidade);

endfor

fclose(fid);


% ============================================================
% E03 - RMS EM INTERVALOS EQUIVALENTES
%
% Comparacao:
% A. RMS de todo o regime.
% B. RMS de ciclos completos normalizados.
% C. RMS de ciclos completos na grade nativa.
% D. RMS no mesmo suporte absoluto pre-evento.
% ============================================================

fprintf("\n========================================\n");
fprintf(" VARIABILIDADE RMS EM SUPORTE EQUIVALENTE\n");
fprintf("========================================\n");

RMS_FULL=NaN(12,2,3);

RMS_CICLOS=NaN(12,2,3);

RMS_NATIVO=NaN(12,2,3);

RMS_COMUM=NaN(12,3);

KCOMUM=zeros(1,2);

for q=1:2

  kk=[];

  for r=TREINO
    kk(end+1)=size(C{r,q},3);
  endfor

  KCOMUM(q)=min(kk);

endfor

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_RMS_Equivalente.csv", ...
    ["registro,grupo,regiao,fase,n_ciclos," ...
     "rms_regime_A,rms_ciclos_A,rms_nativo_A," ...
     "rms_suporte_comum_A"]);

for r=REGISTROS

  RMS_COMUM(r,:)=rms_local( ...
      REG{r,1}(1:NCOMUM,:));

  for q=1:2

    A=C{r,q};

    K=min(KCOMUM(q),size(A,3));

    M=juntar_ciclos(A(:,:,1:K));

    a=rms_local(REG{r,q});

    b=rms_local(M);

    RN=NATIVE_RMS{r,q}(1:K,:);

    % RMS medio quadraticamente entre ciclos.
    % Cada ciclo recebe o mesmo peso.

    c=sqrt(mean(RN.^2,1));

    RMS_FULL(r,q,:)=reshape(a,1,1,3);

    RMS_CICLOS(r,q,:)=reshape(b,1,1,3);

    RMS_NATIVO(r,q,:)=reshape(c,1,1,3);

    for p=1:3

      comum=NaN;

      if q==1

        comum=RMS_COMUM(r,p);

      endif

      linha(fid,[r,GRUPO(r),q,p,K, ...
                 a(p),b(p),c(p),comum]);

    endfor

  endfor

endfor

fclose(fid);

CV=NaN(2,3,4);

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_CV_RMS.csv", ...
    ["regiao,fase,cv_regime_pct," ...
     "cv_ciclos_pct,cv_nativo_pct," ...
     "cv_suporte_comum_pct"]);

for q=1:2

  for p=1:3

    a=reshape(RMS_FULL(TREINO,q,p),[],1);

    b=reshape(RMS_CICLOS(TREINO,q,p),[],1);

    c=reshape(RMS_NATIVO(TREINO,q,p),[],1);

    cv0=cv_local(a);

    cv1=cv_local(b);

    cv2=cv_local(c);

    cv3=NaN;

    if q==1

      cv3=cv_local(RMS_COMUM(TREINO,p));

    endif

    CV(q,p,:)=reshape([cv0 cv1 cv2 cv3],1,1,4);

    linha(fid,[q,p,cv0,cv1,cv2,cv3]);

    fprintf("\n%s | fase %s\n", ...
            nome_regime(q),FASES{p});

    fprintf("CV regime completo: %.10f %%\n",cv0);

    fprintf("CV ciclos normalizados: %.10f %%\n",cv1);

    fprintf("CV ciclos nativos: %.10f %%\n",cv2);

    fprintf("CV suporte comum: %.10f %%\n",cv3);

  endfor

endfor

fclose(fid);


% ============================================================
% E04 - REFERENCIAS GLOBAIS DE TREINAMENTO
%
% Referencias completas usadas apenas para visualizacao.
% Metricas inter-registros utilizam cross-fit.
% ============================================================

fprintf("\n========================================\n");
fprintf(" CONSTRUCAO DAS REFERENCIAS GLOBAIS\n");
fprintf("========================================\n");

REF_LOCAL=cell(1,2);

REF_GLOBAL=cell(1,2);

H_LOCAL=cell(1,2);

H_GLOBAL=cell(1,2);

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Convergencia.csv", ...
    ["regiao,metodo,iteracao," ...
     "funcao_custo_anterior," ...
     "variacao_referencia_A"]);

for q=1:2

  % Para o pre-evento, referencias globais descritivas
  % utilizam um representante por trajetoria exata.

  ids=TREINO;

  if q==1
    ids=REP;
  endif

  [REF_LOCAL{q},H_LOCAL{q}]=construir_ref( ...
      C(:,q),ids,NREF,LLOCAL,3);

  [REF_GLOBAL{q},H_GLOBAL{q}]=construir_ref( ...
      C(:,q),ids,NREF,LGLOBAL,MAXIT);

  H=H_LOCAL{q};

  for j=1:rows(H)
    linha(fid,[q,1,H(j,:)]);
  endfor

  H=H_GLOBAL{q};

  for j=1:rows(H)
    linha(fid,[q,3,H(j,:)]);
  endfor

  fprintf("%s | referencia limitada: %d iteracoes\n", ...
          nome_regime(q),rows(H_LOCAL{q}));

  fprintf("%s | referencia global: %d iteracoes\n", ...
          nome_regime(q),rows(H_GLOBAL{q}));

endfor

fclose(fid);


% ============================================================
% E05 - COMPARACAO CROSS-FIT DOS METODOS
%
% Metodo 1:
% Referencia limitada + alinhamento limitado.
%
% Metodo 2:
% Mesma referencia limitada + alinhamento global.
% Isola o efeito da ampliacao da busca.
%
% Metodo 3:
% Referencia refinada globalmente + alinhamento global.
%
% Metodo 4:
% Referencia intraregistro leave-one-cycle-out.
% Descritiva, nao evidencia de generalizacao.
%
% Pre-evento:
% Excluir doadores exatamente iguais ao registro alvo
% no suporte comum.
%
% Validacao nunca empregada para ajustar referencias
% dos metodos inter-registros.
% ============================================================

fprintf("\n========================================\n");
fprintf(" COMPARACAO DOS METODOS\n");
fprintf("========================================\n");

RES=cell(12,2,4);

ALIGNED=cell(12,2,4);

REF_CROSSFIT=cell(12,2,4);

SHIFTS=cell(12,2,4);

RMS_ERRO=NaN(12,2,4,3);

NRMSE=NaN(12,2,4,3);

CUSTO=NaN(12,2,4);

SAT_LOCAL=NaN(12,2);

NDOADORES=zeros(12,2);

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Erros_Ciclos.csv", ...
    ["registro,grupo,regiao,metodo,ciclo," ...
     "n_doadores,shift_amostras,shift_graus," ...
     "shift_us_equivalente,custo_normalizado," ...
     "rmse_A,rmse_B,rmse_C," ...
     "nrmse_A_pct,nrmse_B_pct,nrmse_C_pct"]);

fs=abrir_csv(OUT, ...
    "G4_E02_4_1_Comparacao_Registros.csv", ...
    ["registro,grupo,regiao,metodo,n_ciclos," ...
     "n_doadores,rmse_A,rmse_B,rmse_C," ...
     "nrmse_A_pct,nrmse_B_pct,nrmse_C_pct," ...
     "custo_normalizado,saturacao_local_pct"]);

for r=REGISTROS

  for q=1:2

    A=C{r,q};

    K=size(A,3);

    doadores=[];

    for s=TREINO

      if s==r
        continue;
      endif

      if q==1

        xr=REG{r,1}(1:NCOMUM,:);

        xs=REG{s,1}(1:NCOMUM,:);

        if isequal(xr,xs)
          continue;
        endif

        % Evita multiplicar o peso de trajetorias
        % duplicadas na referencia pre-evento.

        repetido=false;

        for jj=1:numel(doadores)

          if GEX(s)==GEX(doadores(jj))

            repetido=true;

            break;

          endif

        endfor

        if repetido
          continue;
        endif

      endif

      doadores(end+1)=s;

    endfor

    NDOADORES(r,q)=numel(doadores);

    fprintf("\nR%02d | %s | doadores=%d\n", ...
            r,nome_regime(q),numel(doadores));

    refL=[];
    refG=[];

    if !isempty(doadores)

      refL=construir_ref( ...
          C(:,q),doadores,NREF,LLOCAL,3);

      refG=construir_ref( ...
          C(:,q),doadores,NREF,LGLOBAL,MAXIT);

    endif

    for m=1:4

      if m<=3 && isempty(doadores)

        fprintf("%s: sem doadores elegiveis\n", ...
                METODOS{m});

        continue;

      endif

      R=zeros(NREF,3,K);

      YALL=zeros(NREF,3,K);

      RF=zeros(NREF,3,K);

      shifts=zeros(K,1);

      custos=zeros(K,1);

      for k=1:K

        if m==1

          ref=refL;
          L=LLOCAL;
          nd=numel(doadores);

        elseif m==2

          ref=refL;
          L=LGLOBAL;
          nd=numel(doadores);

        elseif m==3

          ref=refG;
          L=LGLOBAL;
          nd=numel(doadores);

        else

          ref=construir_intra(A,k,NREF,LGLOBAL);

          L=LGLOBAL;

          nd=K-1;

        endif

        [Y,d,e]=alinhar(A(:,:,k),ref,L);

        R(:,:,k)=Y-ref;

        YALL(:,:,k)=Y;

        RF(:,:,k)=ref;

        shifts(k)=d;

        custos(k)=transformar_escala(Y,ref);

        rr=rms_local(R(:,:,k));

        nn=100*rr./max(rms_local(A(:,:,k)),eps);

        linha(fid,[r,GRUPO(r),q,m,k,nd, ...
                   d,360*d/NREF, ...
                   d*PER(r,q)/NREF/FS*1e6, ...
                   e,rr,nn]);

      endfor

      RES{r,q,m}=R;

      ALIGNED{r,q,m}=YALL;

      REF_CROSSFIT{r,q,m}=RF;

      SHIFTS{r,q,m}=shifts;

      Z=juntar_ciclos(R);

      SX=juntar_ciclos(A);

      rr=rms_local(Z);

      nn=100*rr./max(rms_local(SX),eps);

      RMS_ERRO(r,q,m,:)=reshape(rr,1,1,1,3);

      NRMSE(r,q,m,:)=reshape(nn,1,1,1,3);

      CUSTO(r,q,m)=sqrt(mean(custos));

      sat=NaN;

      if m==1

        sat=100*mean(abs(shifts)==LLOCAL);

        SAT_LOCAL(r,q)=sat;

      endif

      linha(fs,[r,GRUPO(r),q,m,K,nd, ...
                rr,nn,CUSTO(r,q,m),sat]);

      fprintf("%s\n",METODOS{m});

      fprintf("  RMSE [A]: %.8f %.8f %.8f\n",rr);

      fprintf("  NRMSE [%%]: %.8f %.8f %.8f\n",nn);

      fprintf("  Custo normalizado: %.8f\n", ...
              CUSTO(r,q,m));

      fprintf("  Shift mediano absoluto [graus]: %.6f\n", ...
              median(abs(shifts))*360/NREF);

      if m==1

        fprintf("  Saturacao local: %.4f %%\n",sat);

      endif

    endfor

    if !isempty(doadores)

      a=CUSTO(r,q,1);

      b=CUSTO(r,q,2);

      c=CUSTO(r,q,3);

      fprintf("Reducao custo local -> global: %.6f %%\n", ...
              100*(a-b)/max(a,eps));

      fprintf("Reducao custo local -> global refinado: %.6f %%\n", ...
              100*(a-c)/max(a,eps));

      % A busca global contem todas as solucoes locais.
      % Portanto, para a mesma referencia, o custo
      % nao pode aumentar, exceto por arredondamento.

      assert(b<=a+1e-10, ...
             "Inconsistencia na busca global.");

    endif

  endfor

endfor

fclose(fid);
fclose(fs);


% ============================================================
% E06 - CONSOLIDACAO DOS ERROS
% ============================================================

fprintf("\n========================================\n");
fprintf(" RESUMO DOS ERROS\n");
fprintf("========================================\n");

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Resumo_Metodos.csv", ...
    ["grupo,regiao,metodo,fase,n_registros," ...
     "rmse_media_A,rmse_mediana_A,rmse_max_A," ...
     "nrmse_media_pct,nrmse_mediana_pct," ...
     "nrmse_max_pct"]);

for g=1:2

  if g==1
    ids=TREINO;
  else
    ids=VALIDACAO;
  endif

  for q=1:2

    for m=1:4

      for p=1:3

        a=reshape(RMS_ERRO(ids,q,m,p),[],1);

        b=reshape(NRMSE(ids,q,m,p),[],1);

        ok=isfinite(a) & isfinite(b);

        a=a(ok);
        b=b(ok);

        if isempty(a)
          continue;
        endif

        linha(fid,[g,q,m,p,numel(a), ...
                   mean(a),median(a),max(a), ...
                   mean(b),median(b),max(b)]);

        fprintf(["Grupo %d | %s | %s | fase %s | " ...
                 "RMSE=%.6f A | NRMSE=%.6f %%\n"], ...
                g,nome_regime(q),METODOS{m}, ...
                FASES{p},mean(a),mean(b));

      endfor

    endfor

  endfor

endfor

fclose(fid);


% ============================================================
% E07 - DECOMPOSICAO DA REDUCAO DO ERRO
% ============================================================

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Reducao_Erro.csv", ...
    ["registro,grupo,regiao,fase," ...
     "nrmse_local_pct,nrmse_global_pct," ...
     "nrmse_global_ref_pct,nrmse_intra_pct," ...
     "reducao_busca_pct,reducao_total_pct," ...
     "reducao_custo_busca_pct," ...
     "reducao_custo_total_pct"]);

REDUCAO=NaN(12,2,3);

for r=REGISTROS

  for q=1:2

    if !isfinite(CUSTO(r,q,1))
      continue;
    endif

    j0=CUSTO(r,q,1);

    j1=CUSTO(r,q,2);

    j2=CUSTO(r,q,3);

    jc1=100*(j0-j1)/max(j0,eps);

    jc2=100*(j0-j2)/max(j0,eps);

    for p=1:3

      a=NRMSE(r,q,1,p);

      b=NRMSE(r,q,2,p);

      c=NRMSE(r,q,3,p);

      d=NRMSE(r,q,4,p);

      reb=100*(a-b)/max(a,eps);

      ret=100*(a-c)/max(a,eps);

      REDUCAO(r,q,p)=ret;

      linha(fid,[r,GRUPO(r),q,p, ...
                 a,b,c,d,reb,ret,jc1,jc2]);

    endfor

  endfor

endfor

fclose(fid);

fprintf("\n========================================\n");
fprintf(" REDUCAO DO ERRO TARDIO - TREINAMENTO\n");
fprintf("========================================\n");

for p=1:3

  a=reshape(NRMSE(TREINO,2,1,p),[],1);

  b=reshape(NRMSE(TREINO,2,2,p),[],1);

  c=reshape(NRMSE(TREINO,2,3,p),[],1);

  d=reshape(NRMSE(TREINO,2,4,p),[],1);

  fprintf("\nFase %s\n",FASES{p});

  fprintf("Local: %.8f %%\n",mean(a));

  fprintf("Busca global: %.8f %%\n",mean(b));

  fprintf("Referencia global: %.8f %%\n",mean(c));

  fprintf("Intraregistro LOCO: %.8f %%\n",mean(d));

  fprintf("Reducao relativa da media: %.8f %%\n", ...
          100*(mean(a)-mean(c))/max(mean(a),eps));

endfor


% ============================================================
% E08 - ESTATISTICAS DOS NOVOS RESIDUOS
%
% Metodo 3: residuos inter-registros globalmente alinhados.
% Metodo 4: residuos intraregistro.
%
% As duas populacoes permanecem separadas.
% ============================================================

fprintf("\n========================================\n");
fprintf(" ESTATISTICAS RESIDUAIS\n");
fprintf("========================================\n");

POOL=cell(2,2,4,3);

TAU=cell(2,4,3);

ACFM=cell(2,4,3);

PSDPOOL=cell(2,4,3);

fc=abrir_csv(OUT, ...
    "G4_E02_4_1_Correlacoes.csv", ...
    ["registro,grupo,regiao,metodo," ...
     "corr_AB,corr_AC,corr_BC"]);

fa=abrir_csv(OUT, ...
    "G4_E02_4_1_ACF.csv", ...
    ["registro,grupo,regiao,metodo,fase,ciclo," ...
     "tau_e_ms,tau_005_ms,acf_lag1"]);

fp=abrir_csv(OUT, ...
    "G4_E02_4_1_PSD.csv", ...
    ["registro,grupo,regiao,metodo,fase,ciclo," ...
     "psd_0_100,psd_100_500,psd_500_1000," ...
     "psd_1000_5000,psd_5000_12500"]);

fr=abrir_csv(OUT, ...
    "G4_E02_4_1_Estatisticas_Residuos.csv", ...
    ["registro,grupo,regiao,metodo,fase,n," ...
     "media_A,dp_A,rms_A,mediana_A,mad_A," ...
     "p01_A,p05_A,p95_A,p99_A,min_A,max_A"]);

for r=REGISTROS

  g=GRUPO(r);

  for q=1:2

    for m=[3 4]

      R=RES{r,q,m};

      if isempty(R)
        continue;
      endif

      M=juntar_ciclos(R);

      CR=corrcoef(M);

      linha(fc,[r,g,q,m, ...
                CR(1,2),CR(1,3),CR(2,3)]);

      for p=1:3

        x=M(:,p);

        POOL{g,q,m,p}=[POOL{g,q,m,p};x];

        md=median(x);

        linha(fr,[r,g,q,m,p,numel(x), ...
                  mean(x),std(x),rms_local(x),md, ...
                  median(abs(x-md)), ...
                  pct(x,1),pct(x,5), ...
                  pct(x,95),pct(x,99), ...
                  min(x),max(x)]);

        for k=1:size(R,3)

          z=R(:,p,k);

          % Tempo equivalente do ciclo normalizado.

          [ac,te,t05,l1]=acf_residuo( ...
              z,NREF-2,TS);

          linha(fa,[r,g,q,m,p,k, ...
                    1000*te,1000*t05,l1]);

          bb=psd_bandas(z,FS);

          linha(fp,[r,g,q,m,p,k,bb]);

          if g==1

            if isfinite(te)

              TAU{q,m,p}=[TAU{q,m,p};te];

            endif

            if all(isfinite(ac))

              ACFM{q,m,p}=[ACFM{q,m,p},ac];

            endif

            PSDPOOL{q,m,p}=[PSDPOOL{q,m,p};bb];

          endif

        endfor

      endfor

    endfor

  endfor

endfor

fclose(fc);
fclose(fa);
fclose(fp);
fclose(fr);


% ============================================================
% E09 - ENVOLTORIAS E COBERTURA
%
% Somente residuos inter-registros podem ser utilizados
% para avaliar cobertura externa por referencia.
%
% Residuos intraregistro sao descritivos.
% Nao equivalem a uma populacao independente.
% ============================================================

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Envoltoria.csv", ...
    ["regiao,metodo,fase,n_treino,n_validacao," ...
     "n_registros_treino,media_A,dp_A,rms_A," ...
     "mediana_A,mad_A,p01_A,p05_A,p95_A,p99_A," ...
     "min_A,max_A,cobertura_val_pct," ...
     "cobertura_minmax_val_pct," ...
     "tau_mediana_ms"]);

ENV=NaN(2,4,3,4);

fprintf("\n========================================\n");
fprintf(" NOVAS ENVOLTORIAS RESIDUAIS\n");
fprintf("========================================\n");

for q=1:2

  for m=[3 4]

    for p=1:3

      x=POOL{1,q,m,p};

      y=POOL{2,q,m,p};

      if isempty(x)

        fprintf("%s | %s | %s: sem residuos elegiveis\n", ...
                nome_regime(q),METODOS{m},FASES{p});

        continue;

      endif

      l=pct(x,1);

      u=pct(x,99);

      mn=min(x);

      mx=max(x);

      md=median(x);

      mad0=median(abs(x-md));

      cv=cobertura(y,l,u);

      cm=cobertura(y,mn,mx);

      tt=TAU{q,m,p};

      tau=NaN;

      if !isempty(tt)
        tau=median(tt);
      endif

      nreg=0;

      for r=TREINO

        if !isempty(RES{r,q,m})
          nreg=nreg+1;
        endif

      endfor

      ENV(q,m,p,:)=reshape([l u mn mx],1,1,1,4);

      linha(fid,[q,m,p,numel(x),numel(y),nreg, ...
                 mean(x),std(x),rms_local(x),md, ...
                 mad0,l,pct(x,5),pct(x,95),u, ...
                 mn,mx,cv,cm,1000*tau]);

      fprintf("\n%s | %s | fase %s\n", ...
              nome_regime(q),METODOS{m},FASES{p});

      fprintf("Registros de treino contribuintes: %d\n",nreg);

      fprintf("Amostras residuais: %d\n",numel(x));

      fprintf("RMS residual: %.8f A\n",rms_local(x));

      fprintf("MAD residual: %.8f A\n",mad0);

      fprintf("P01/P99: %.8f / %.8f A\n",l,u);

      fprintf("Tau mediana: %.8f ms\n",1000*tau);

      fprintf("Cobertura validacao: %.6f %%\n",cv);

      if m==4

        fprintf(["ATENCAO: cobertura intraregistro " ...
                 "apenas descritiva.\n"]);

      endif

    endfor

  endfor

endfor

fclose(fid);


% ============================================================
% E10 - DEPENDENCIA DOS RESIDUOS COM A POSICAO NO CICLO
%
% Investiga se os residuos sao homogeneos em fase ou
% se concentram em regioes de comutacao.
%
% Se os residuos dependerem fortemente da coordenada
% do ciclo, bootstrap estacionario indiscriminado
% nao sera considerado adequado.
% ============================================================

NBINS=12;

PHASE_RMS=NaN(2,4,NBINS,3);

PHASE_BIAS=NaN(2,4,NBINS,3);

PHASE_DP=NaN(2,4,NBINS,3);

HETERO=NaN(2,4,3);

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Residuos_Posicao_Ciclo.csv", ...
    ["regiao,metodo,fase,bin," ...
     "inicio_graus,fim_graus,n," ...
     "media_A,dp_A,rms_A,p01_A,p99_A"]);

for q=1:2

  for m=[3 4]

    for p=1:3

      for b=1:NBINS

        ia=floor((b-1)*NREF/NBINS)+1;

        ib=floor(b*NREF/NBINS);

        x=[];

        for r=TREINO

          R=RES{r,q,m};

          if isempty(R)
            continue;
          endif

          % A coordenada circular de todas as referencias
          % cross-fit e intraregistro deve ser padronizada
          % antes de combinar residuos de registros.

          RR=REF_CROSSFIT{r,q,m};

          for kk=1:size(R,3)

            ref=RR(:,:,kk);

            alvo=REF_GLOBAL{q};

            [Y,d]=alinhar(ref,alvo,LGLOBAL);

            Z=circshift(R(:,:,kk),d,1);

            x=[x;Z(ia:ib,p)];

          endfor

        endfor

        if isempty(x)
          continue;
        endif

        a=rms_local(x);

        PHASE_RMS(q,m,b,p)=a;

        PHASE_BIAS(q,m,b,p)=mean(x);

        PHASE_DP(q,m,b,p)=std(x);

        linha(fid,[q,m,p,b, ...
                   360*(b-1)/NBINS,360*b/NBINS, ...
                   numel(x),mean(x),std(x),a, ...
                   pct(x,1),pct(x,99)]);

      endfor

      a=reshape(PHASE_RMS(q,m,:,p),[],1);

      a=a(isfinite(a));

      if !isempty(a)

        HETERO(q,m,p)=max(a)/max(median(a),eps);

      endif

    endfor

  endfor

endfor

fclose(fid);

fprintf("\n========================================\n");
fprintf(" HETEROGENEIDADE DOS RESIDUOS NO CICLO\n");
fprintf("========================================\n");

for q=1:2

  for m=[3 4]

    a=reshape(HETERO(q,m,:),1,3);

    fprintf("%s | %s | max RMS / mediana: ", ...
            nome_regime(q),METODOS{m});

    fprintf("%.6f %.6f %.6f\n",a);

  endfor

endfor


% ============================================================
% E11 - COMPARACAO DA ESTRUTURA DOS PULSOS
%
% Mesmo apos alinhamento, os residuos podem representar
% diferencas de largura, pico ou forma dos pulsos.
%
% Caracterizacao em ciclos nativos sem modificacao
% das correntes originais.
% ============================================================

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Forma_Pulsos.csv", ...
    ["registro,grupo,regiao,fase,ciclo," ...
     "rms_A,pico_A,media_A,fator_crista," ...
     "atividade_pct,rms_derivada_A_s"]);

LIMIAR=zeros(1,3);

for p=1:3

  pk=[];

  for r=TREINO

    pk(end+1)=max(abs(REG{r,1}(:,p)));

  endfor

  LIMIAR(p)=0.10*median(pk);

endfor

for r=REGISTROS

  for q=1:2

    X=REG{r,q};

    P=PER(r,q);

    for k=1:size(C{r,q},3)

      inicio=INICIOS{r,q}(k);

      ia=max(1,ceil(inicio));

      ib=min(rows(X),ceil(inicio+P)-1);

      for p=1:3

        x=X(ia:ib,p);

        rr=rms_local(x);

        pk=max(abs(x));

        atividade=100*mean(x>=LIMIAR(p));

        linha(fid,[r,GRUPO(r),q,p,k,rr,pk, ...
                   mean(x),pk/max(rr,eps), ...
                   atividade,rms_local(diff(x)/TS)]);

      endfor

    endfor

  endfor

endfor

fclose(fid);


% ============================================================
% E12 - DIAGNOSTICO EXPLORATORIO DO BOOTSTRAP
%
% Nenhuma decisao de aprovacao fisica e automatizada.
%
% Criterios quantitativos apenas exploratorios:
% - NRMSE mediana <= 5%.
% - NRMSE maxima <= 10%.
% - Reducao relativa do erro medio >= 50%.
%
% Avaliacao exclusivamente em treinamento.
% A validacao permanece como auditoria externa.
%
% Mesmo que todos esses criterios sejam atendidos,
% a ausencia de diversidade fisica permanece.
% ============================================================

fprintf("\n========================================\n");
fprintf(" TRIAGEM PARA BOOTSTRAP MULTIVARIADO\n");
fprintf("========================================\n");

fid=abrir_csv(OUT, ...
    "G4_E02_4_1_Triagem_Bootstrap.csv", ...
    ["regiao,fase,nrmse_mediana_global_pct," ...
     "nrmse_max_global_pct,reducao_media_pct," ...
     "nrmse_mediana_intra_pct," ...
     "nrmse_max_intra_pct," ...
     "heterogeneidade_global," ...
     "heterogeneidade_intra," ...
     "criterio_mediana,criterio_maximo," ...
     "criterio_reducao,triagem_quantitativa"]);

TRIAGEM=zeros(2,3);

for q=1:2

  for p=1:3

    a=reshape(NRMSE(TREINO,q,1,p),[],1);

    b=reshape(NRMSE(TREINO,q,3,p),[],1);

    d=reshape(NRMSE(TREINO,q,4,p),[],1);

    a=a(isfinite(a));

    b=b(isfinite(b));

    d=d(isfinite(d));

    med=NaN;
    mx=NaN;
    red=NaN;

    mi=NaN;
    mxi=NaN;

    if !isempty(d)

      mi=median(d);
      mxi=max(d);

    endif

    c1=0;
    c2=0;
    c3=0;

    if !isempty(b) && !isempty(a)

      med=median(b);

      mx=max(b);

      red=100*(mean(a)-mean(b))/max(mean(a),eps);

      c1=med<=LIM_NRMSE_MEDIANA;

      c2=mx<=LIM_NRMSE_MAX;

      c3=red>=LIM_REDUCAO;

    endif

    aprovado=c1 && c2 && c3;

    TRIAGEM(q,p)=aprovado;

    linha(fid,[q,p,med,mx,red,mi,mxi, ...
               HETERO(q,3,p),HETERO(q,4,p), ...
               c1,c2,c3,aprovado]);

    fprintf("\n%s | fase %s\n",nome_regime(q),FASES{p});

    fprintf("NRMSE mediana inter-registros: %.8f %%\n",med);

    fprintf("NRMSE maxima inter-registros: %.8f %%\n",mx);

    fprintf("Reducao relativa media: %.8f %%\n",red);

    fprintf("NRMSE mediana intraregistro: %.8f %%\n",mi);

    fprintf("NRMSE maxima intraregistro: %.8f %%\n",mxi);

    if isempty(b)

      fprintf("STATUS: SEM BASE INTER-REGISTROS ELEGIVEL\n");

    elseif aprovado

      fprintf("STATUS: CANDIDATO A INVESTIGACAO POSTERIOR\n");

    else

      fprintf("STATUS: CRITERIOS EXPLORATORIOS NAO ATENDIDOS\n");

    endif

  endfor

endfor

fclose(fid);

fprintf("\nNenhum bootstrap foi executado.\n");

fprintf(["O transitorio permanece excluido do bootstrap " ...
         "estacionario.\n"]);

fprintf(["A elegibilidade definitiva depende da " ...
         "interpretacao cientifica.\n"]);


% ============================================================
% EXPORTACAO DAS FIGURAS
% ============================================================

fprintf("\n========================================\n");
fprintf(" EXPORTACAO DAS FIGURAS\n");
fprintf("========================================\n");


% ------------------------------------------------------------
% FIGURA 01 - RMS equivalente e coeficiente de variacao
% ------------------------------------------------------------

f=nova_figura(1300,850);

for q=1:2

  subplot(2,1,q);

  B=reshape(CV(q,:,:),3,4);

  bar(B,"grouped");

  set(gca,"xtick",1:3,"xticklabel",FASES);

  ylabel("CV do RMS [%]");

  title(sprintf("Variabilidade em suporte equivalente - %s", ...
                nome_regime(q)));

  legend({"Regime inteiro","Ciclos normalizados", ...
          "Ciclos nativos","Suporte comum"}, ...
         "location","northeastoutside");

  grid on;

endfor

exportar(f,FIG,"G4_E02_4_1_01_CV_RMS");


% ------------------------------------------------------------
% FIGURA 02 - Referencias limitadas e globais
% ------------------------------------------------------------

f=nova_figura(1350,1050);

theta=(0:NREF-1)'*360/NREF;

for p=1:3

  subplot(3,1,p);

  hold on;

  rl=REF_LOCAL{2};

  rg=alinhar(REF_GLOBAL{2},rl,LGLOBAL);

  plot(theta,rl(:,p),"--", ...
       "color",[0.45 0.45 0.45], ...
       "linewidth",1.5);

  plot(theta,rg(:,p), ...
       "color",CORES(p,:), ...
       "linewidth",1.7);

  ylabel("Corrente [A]");

  title(sprintf("Referencia tardia - fase %s",FASES{p}));

  legend({"Limitada","Global refinada"}, ...
         "location","northeastoutside");

  grid on;

  if p==3
    xlabel("Coordenada normalizada [graus]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_02_Referencias");


% ------------------------------------------------------------
% FIGURA 03 - Funcao custo da busca circular
% Registros R01 e R11.
% ------------------------------------------------------------

f=nova_figura(1350,850);

ids=[1 11];

for j=1:2

  r=ids(j);

  subplot(2,1,j);

  A=C{r,2}(:,:,1);

  ref=REF_CROSSFIT{r,2,1}(:,:,1);

  [Y,d,e,curva]=alinhar(A,ref,LGLOBAL);

  plot(curva(:,1)*360/NREF,curva(:,2), ...
       "color",[0.10 0.35 0.80], ...
       "linewidth",1.7);

  hold on;

  yl=ylim();

  lim=LLOCAL*360/NREF;

  plot([-lim -lim],yl,"--", ...
       "color",[0.85 0.30 0.10], ...
       "linewidth",1.4);

  plot([lim lim],yl,"--", ...
       "color",[0.85 0.30 0.10], ...
       "linewidth",1.4);

  plot(d*360/NREF,e,"ko", ...
       "markersize",8, ...
       "markerfacecolor","k");

  ylabel("Custo normalizado");

  title(sprintf("R%02d - busca circular completa",r));

  xlim([-180 180]);

  grid on;

  if j==2
    xlabel("Deslocamento circular [graus]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_03_Funcao_Custo");


% ------------------------------------------------------------
% FIGURA 04 - Comparacao dos quatro metodos
% ------------------------------------------------------------

f=nova_figura(1500,1050);

for p=1:3

  subplot(3,1,p);

  B=reshape(NRMSE(REGISTROS,2,:,p), ...
            numel(REGISTROS),4);

  bar(B,"grouped");

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS);

  ylabel("NRMSE [%]");

  title(sprintf("Comparacao tardia - fase %s",FASES{p}));

  legend(METODOS,"location","northeastoutside");

  grid on;

  if p==3
    xlabel("Registro");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_04_Comparacao_NRMSE");


% ------------------------------------------------------------
% FIGURA 05 - Reducao do erro por registro
% ------------------------------------------------------------

f=nova_figura(1400,600);

B=reshape(REDUCAO(REGISTROS,2,:), ...
          numel(REGISTROS),3);

bar(B,"grouped");

set(gca,"xtick",1:numel(REGISTROS), ...
        "xticklabel",REGISTROS);

ylabel("Reducao relativa do NRMSE [%]");

xlabel("Registro");

title("Alinhamento limitado versus referencia global");

legend(FASES,"location","northeastoutside");

grid on;

exportar(f,FIG,"G4_E02_4_1_05_Reducao_Erro");


% ------------------------------------------------------------
% FIGURAS 06 e 07 - Ciclos antes e depois do alinhamento
% ------------------------------------------------------------

ids=[1 11];

for j=1:2

  r=ids(j);

  f=nova_figura(1500,1100);

  for p=1:3

    subplot(3,2,2*p-1);

    hold on;

    Z=ALIGNED{r,2,1};

    ref=REF_CROSSFIT{r,2,1}(:,:,1);

    for k=1:size(Z,3)

      plot(theta,Z(:,p,k), ...
           "color",[0.68 0.72 0.79], ...
           "linewidth",0.5);

    endfor

    plot(theta,ref(:,p), ...
         "color",CORES(p,:), ...
         "linewidth",1.8);

    ylabel("Corrente [A]");

    title(sprintf("R%02d - fase %s - limitada", ...
                  r,FASES{p}));

    grid on;

    subplot(3,2,2*p);

    hold on;

    Z=ALIGNED{r,2,3};

    ref=REF_CROSSFIT{r,2,3}(:,:,1);

    for k=1:size(Z,3)

      plot(theta,Z(:,p,k), ...
           "color",[0.68 0.72 0.79], ...
           "linewidth",0.5);

    endfor

    plot(theta,ref(:,p), ...
         "color",CORES(p,:), ...
         "linewidth",1.8);

    title(sprintf("R%02d - fase %s - global", ...
                  r,FASES{p}));

    grid on;

    if p==3

      xlabel("Coordenada normalizada [graus]");

      subplot(3,2,2*p-1);

      xlabel("Coordenada normalizada [graus]");

    endif

  endfor

  exportar(f,FIG, ...
           sprintf("G4_E02_4_1_%02d_Ciclos_R%02d",5+j,r));

endfor


% ------------------------------------------------------------
% FIGURA 08 - Residuo antes e depois
% ------------------------------------------------------------

f=nova_figura(1350,1050);

r=11;

for p=1:3

  subplot(3,1,p);

  hold on;

  a=RES{r,2,1}(:,p,1);

  b=RES{r,2,3}(:,p,1);

  c=RES{r,2,4}(:,p,1);

  plot(theta,a, ...
       "color",[0.85 0.25 0.15], ...
       "linewidth",1.1);

  plot(theta,b, ...
       "color",[0.10 0.35 0.80], ...
       "linewidth",1.2);

  plot(theta,c, ...
       "color",[0.10 0.60 0.30], ...
       "linewidth",1.2);

  ylabel("Residuo [A]");

  title(sprintf("R11 - residuos - fase %s",FASES{p}));

  legend({"Local","Global","Intraregistro"}, ...
         "location","northeastoutside");

  grid on;

  if p==3
    xlabel("Coordenada normalizada [graus]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_08_Residuos_R11");


% ------------------------------------------------------------
% FIGURA 09 - Variabilidade intraregistro pre-evento
% ------------------------------------------------------------

f=nova_figura(1350,1050);

for p=1:3

  subplot(3,1,p);

  B=reshape(NRMSE(REGISTROS,1,4,p),[],1);

  bar(B);

  set(gca,"xtick",1:numel(REGISTROS), ...
          "xticklabel",REGISTROS);

  ylabel("NRMSE [%]");

  title(sprintf("Pre-evento intraregistro LOCO - fase %s", ...
                FASES{p}));

  grid on;

  if p==3
    xlabel("Registro");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_09_Intraregistro_Pre");


% ------------------------------------------------------------
% FIGURA 10 - Distribuicoes residuais tardias
% ------------------------------------------------------------

f=nova_figura(1450,1000);

for p=1:3

  for j=1:2

    m=j+2;

    subplot(3,2,2*(p-1)+j);

    x=POOL{1,2,m,p};

    if !isempty(x)

      [h,c]=hist(x,100);

      h=h/max(sum(h),eps);

      bar(c,h,1, ...
          "facecolor",CORES(p,:), ...
          "edgecolor","none");

    endif

    title(sprintf("%s - fase %s", ...
                  METODOS{m},FASES{p}));

    ylabel("Frequencia relativa");

    grid on;

    if p==3
      xlabel("Residuo [A]");
    endif

  endfor

endfor

exportar(f,FIG,"G4_E02_4_1_10_Histogramas");


% ------------------------------------------------------------
% FIGURA 11 - ACF residual
% ------------------------------------------------------------

f=nova_figura(1450,900);

for q=1:2

  for j=1:2

    m=j+2;

    subplot(2,2,2*(q-1)+j);

    hold on;

    lg={};

    for p=1:3

      M=ACFM{q,m,p};

      if !isempty(M)

        a=mean(M,2);

        plot((0:numel(a)-1)*TS*1000,a, ...
             "color",CORES(p,:), ...
             "linewidth",1.3);

        lg{end+1}=FASES{p};

      endif

    endfor

    if !isempty(lg)

      legend(lg,"location","northeastoutside");

    else

      text(0.5,0.5,"Sem residuos elegiveis", ...
           "units","normalized", ...
           "horizontalalignment","center");

    endif

    title(sprintf("%s - %s",nome_regime(q),METODOS{m}));

    ylabel("Autocorrelacao");

    ylim([-1 1]);

    xlim([0 (NREF-2)*TS*1000]);

    grid on;

    if q==2
      xlabel("Atraso equivalente [ms]");
    endif

  endfor

endfor

exportar(f,FIG,"G4_E02_4_1_11_ACF");


% ------------------------------------------------------------
% FIGURA 12 - RMS residual por posicao no ciclo
% ------------------------------------------------------------

f=nova_figura(1400,900);

ang=((1:NBINS)-0.5)*360/NBINS;

for q=1:2

  for j=1:2

    m=j+2;

    subplot(2,2,2*(q-1)+j);

    hold on;

    lg={};

    for p=1:3

      a=reshape(PHASE_RMS(q,m,:,p),[],1);

      if any(isfinite(a))

        plot(ang,a,"-o", ...
             "color",CORES(p,:), ...
             "linewidth",1.4);

        lg{end+1}=FASES{p};

      endif

    endfor

    if !isempty(lg)

      legend(lg,"location","northeastoutside");

    else

      text(0.5,0.5,"Sem residuos elegiveis", ...
           "units","normalized", ...
           "horizontalalignment","center");

    endif

    title(sprintf("%s - %s",nome_regime(q),METODOS{m}));

    ylabel("RMS residual [A]");

    xlim([0 360]);

    grid on;

    if q==2
      xlabel("Posicao no ciclo [graus]");
    endif

  endfor

endfor

exportar(f,FIG,"G4_E02_4_1_12_Residuos_Fase_Ciclo");


% ------------------------------------------------------------
% FIGURA 13 - Convergencia das referencias
% ------------------------------------------------------------

f=nova_figura(1300,600);

for q=1:2

  subplot(1,2,q);

  hold on;

  H=H_LOCAL{q};

  plot(H(:,1),H(:,2),"-o", ...
       "color",[0.85 0.30 0.10], ...
       "linewidth",1.5);

  H=H_GLOBAL{q};

  plot(H(:,1),H(:,2),"-o", ...
       "color",[0.10 0.35 0.80], ...
       "linewidth",1.5);

  xlabel("Iteracao");

  ylabel("Custo anterior a atualizacao");

  title(sprintf("Refinamento - %s",nome_regime(q)));

  legend({"Local","Global"}, ...
         "location","northeastoutside");

  grid on;

endfor

exportar(f,FIG,"G4_E02_4_1_13_Convergencia");


% ------------------------------------------------------------
% FIGURA 14 - Deslocamentos globais
% ------------------------------------------------------------

f=nova_figura(1350,650);

hold on;

for r=REGISTROS

  d=SHIFTS{r,2,3};

  if isempty(d)
    continue;
  endif

  if GRUPO(r)==1

    cor=[0.10 0.35 0.80];

  else

    cor=[0.85 0.30 0.10];

  endif

  plot(r*ones(size(d)),d*360/NREF,"o", ...
       "color",cor, ...
       "markerfacecolor",cor, ...
       "markersize",4);

endfor

set(gca,"xtick",REGISTROS);

xlabel("Registro");

ylabel("Deslocamento circular [graus]");

title("Deslocamentos relativos as referencias cross-fit");

ylim([-180 180]);

grid on;

exportar(f,FIG,"G4_E02_4_1_14_Deslocamentos");


% ------------------------------------------------------------
% FIGURA 15 - Potencia espectral residual
% ------------------------------------------------------------

f=nova_figura(1400,1000);

bandas={"0-100","100-500","500-1000", ...
        "1k-5k","5k-12.5k"};

for p=1:3

  subplot(3,1,p);

  B=NaN(5,2);

  for j=1:2

    m=j+2;

    A=PSDPOOL{2,m,p};

    if !isempty(A)
      B(:,j)=median(A,1)';
    endif

  endfor

  bar(B,"grouped");

  set(gca,"yscale","log", ...
          "xtick",1:5, ...
          "xticklabel",bandas);

  ylabel("Potencia [A^2]");

  title(sprintf("PSD residual tardia - fase %s",FASES{p}));

  legend({"Inter-registros","Intraregistro"}, ...
         "location","northeastoutside");

  grid on;

  if p==3
    xlabel("Banda de frequencia [Hz]");
  endif

endfor

exportar(f,FIG,"G4_E02_4_1_15_PSD_Residuos");


% ------------------------------------------------------------
% FIGURA 16 - Envoltoria P01-P99
% ------------------------------------------------------------

f=nova_figura(1350,850);

for q=1:2

  subplot(2,1,q);

  hold on;

  for p=1:3

    for j=1:2

      m=j+2;

      E=reshape(ENV(q,m,p,:),1,4);

      if !all(isfinite(E))
        continue;
      endif

      xpos=p+(j-1.5)*0.23;

      if m==3

        cor=[0.10 0.35 0.80];

      else

        cor=[0.10 0.60 0.30];

      endif

      plot([xpos xpos],E(1:2), ...
           "color",cor, ...
           "linewidth",5);

      plot(xpos,E(1),"o","color",cor);

      plot(xpos,E(2),"o","color",cor);

    endfor

  endfor

  set(gca,"xtick",1:3,"xticklabel",FASES);

  xlim([0.5 3.5]);

  ylabel("Residuo [A]");

  title(sprintf("P01-P99: %s",nome_regime(q)));

  grid on;

endfor

exportar(f,FIG,"G4_E02_4_1_16_Envoltoria");


% ============================================================
% PROTOCOLO METODOLOGICO
% ============================================================

fid=fopen(fullfile(OUT, ...
          "G4_E02_4_1_Protocolo.txt"),"w");

fprintf(fid,"GATE 4 - EXPERIMENTO CORRETIVO 02.4.1\n\n");

fprintf(fid,"Fs original: %d Hz\n",FS0);

fprintf(fid,"Fs canonica: %d Hz\n",FS);

fprintf(fid,"Amostras por ciclo: %d\n",NREF);

fprintf(fid,"Limite local: +/- %d amostras\n",LLOCAL);

fprintf(fid,"Busca global: %d posicoes\n",NREF);

fprintf(fid,"Iteracoes globais maximas: %d\n",MAXIT);

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
 ["Regularizacao por ultima ocorrencia nominal.\n" ...
  "Reconstrucao ZOH exata exigida.\n" ...
  "Alinhamento circular conjunto das tres fases.\n" ...
  "Nenhum ajuste independente de amplitude ou fase.\n" ...
  "Metodo 1: referencia e busca limitadas.\n" ...
  "Metodo 2: referencia limitada e busca global.\n" ...
  "Metodo 3: referencia e busca globais.\n" ...
  "Metodo 4: leave-one-cycle-out intraregistro.\n" ...
  "Referencias externas ajustadas somente no treino.\n" ...
  "Exclusao de registros alvo no cross-fit.\n" ...
  "Exclusao de trajetorias pre-evento exatamente iguais.\n" ...
  "Residuos intraregistro possuem finalidade descritiva.\n" ...
  "Nenhuma independencia fisica e presumida.\n" ...
  "ACF calculada separadamente em cada ciclo.\n" ...
  "PSD calculada em coordenada temporal normalizada.\n" ...
  "Envoltorias estimadas exclusivamente no treino.\n" ...
  "Validacao nao empregada na parametrizacao.\n" ...
  "Transitorio excluido do bootstrap estacionario.\n" ...
  "Nenhum bootstrap foi executado.\n" ...
  "Nenhuma variante definitiva produzida.\n" ...
  "Nenhum modelo de IA treinado.\n" ...
  "Arquivos originais preservados.\n" ...
  "Registros 06 e 12 nao carregados.\n\n"]);

fprintf(fid,"CRITERIOS EXPLORATORIOS DE TRIAGEM\n");

fprintf(fid,"NRMSE mediana maxima: %.4f %%\n", ...
        LIM_NRMSE_MEDIANA);

fprintf(fid,"NRMSE maxima permitida: %.4f %%\n", ...
        LIM_NRMSE_MAX);

fprintf(fid,"Reducao relativa minima: %.4f %%\n", ...
        LIM_REDUCAO);

fprintf(fid, ...
 ["Esses criterios nao constituem certificacao fisica.\n" ...
  "A aprovacao metodologica depende de interpretacao.\n" ...
  "O alinhamento retrospectivo nao e detector causal.\n" ...
  "A normalizacao de ciclos nao substitui os sinais.\n" ...
  "Residuos podem conter erros de referencia e fase.\n"]);

fclose(fid);


% ============================================================
% CONSOLIDACAO FINAL
% ============================================================

fprintf("\n========================================\n");
fprintf(" RESULTADOS CONSOLIDADOS\n");
fprintf("========================================\n");

fprintf("Registros processados: %d\n",numel(REGISTROS));

fprintf("Trajetorias pre-evento distintas: %d/%d\n", ...
        numel(REP),numel(TREINO));

fprintf("Suporte comum: %d amostras\n",NCOMUM);

fprintf("Fs canonica: %d Hz\n",FS);

fprintf("\n--- SATURACAO DA BUSCA LOCAL ---\n");

nsat=0;

ntotal=0;

for r=TREINO

  d=SHIFTS{r,2,1};

  if isempty(d)
    continue;
  endif

  nsat=nsat+sum(abs(d)==LLOCAL);

  ntotal=ntotal+numel(d);

endfor

fprintf("Ciclos saturados: %d/%d (%.6f %%)\n", ...
        nsat,ntotal,100*nsat/max(ntotal,1));


fprintf("\n--- COMPARACAO DOS METODOS TARDIOS ---\n");

for g=1:2

  if g==1
    ids=TREINO;
    nome="TREINAMENTO";
  else
    ids=VALIDACAO;
    nome="VALIDACAO";
  endif

  fprintf("\n%s\n",nome);

  for m=1:4

    fprintf("\nMetodo: %s\n",METODOS{m});

    for p=1:3

      a=reshape(RMS_ERRO(ids,2,m,p),[],1);

      b=reshape(NRMSE(ids,2,m,p),[],1);

      a=a(isfinite(a));

      b=b(isfinite(b));

      fprintf(["Fase %s | RMSE medio=%.8f A | " ...
               "NRMSE medio=%.8f %% | " ...
               "NRMSE maximo=%.8f %%\n"], ...
              FASES{p},mean(a),mean(b),max(b));

    endfor

  endfor

endfor


fprintf("\n--- VARIABILIDADE PRE-EVENTO INTRAREGISTRO ---\n");

for p=1:3

  a=reshape(NRMSE(TREINO,1,4,p),[],1);

  b=POOL{1,1,4,p};

  fprintf("\nFase %s\n",FASES{p});

  fprintf("NRMSE intraregistro medio: %.8f %%\n",mean(a));

  fprintf("NRMSE intraregistro mediano: %.8f %%\n",median(a));

  fprintf("NRMSE intraregistro maximo: %.8f %%\n",max(a));

  fprintf("RMS residual agregado: %.8f A\n",rms_local(b));

  fprintf("P01/P99: %.8f / %.8f A\n",pct(b,1),pct(b,99));

  fprintf("CV RMS suporte comum: %.12g %%\n",CV(1,p,4));

endfor


fprintf("\n--- NOVA ENVOLTORIA TARDIA INTER-REGISTROS ---\n");

for p=1:3

  x=POOL{1,2,3,p};

  y=POOL{2,2,3,p};

  fprintf("\nFase %s\n",FASES{p});

  fprintf("RMS residual: %.8f A\n",rms_local(x));

  fprintf("MAD: %.8f A\n",median(abs(x-median(x))));

  fprintf("P01/P99: %.8f / %.8f A\n",pct(x,1),pct(x,99));

  fprintf("Cobertura validacao: %.8f %%\n", ...
          cobertura(y,pct(x,1),pct(x,99)));

  a=TAU{2,3,p};

  if !isempty(a)

    fprintf("Tau mediana: %.8f ms\n",1000*median(a));

  endif

  fprintf("Heterogeneidade no ciclo: %.8f\n", ...
          HETERO(2,3,p));

endfor


fprintf("\n--- TRIAGEM QUANTITATIVA ---\n");

for p=1:3

  fprintf("Tardio | fase %s | criterios atendidos: %d\n", ...
          FASES{p},TRIAGEM(2,p));

endfor

fprintf("\n========================================\n");
fprintf(" G4/E02.4.1 - EXECUCAO CONCLUIDA\n");
fprintf("========================================\n");

fprintf("Identidade pre-evento: AUDITADA\n");

fprintf("RMS em suporte equivalente: ANALISADO\n");

fprintf("Alinhamento circular global: EXECUTADO\n");

fprintf("Referencias cross-fit: ANALISADAS\n");

fprintf("Variabilidade intraregistro: ANALISADA\n");

fprintf("Reducao do erro: QUANTIFICADA\n");

fprintf("Residuos condicionados ao ciclo: ANALISADOS\n");

fprintf("ACF e PSD residuais: ANALISADAS\n");

fprintf("Envoltorias empiricas: PRODUZIDAS\n");

fprintf("Triagem de bootstrap: EXECUTADA\n");

fprintf("Figuras exportadas: 16\n");

fprintf("Bootstrap definitivo: NAO EXECUTADO\n");

fprintf("Variantes definitivas: ZERO\n");

fprintf("Treinamento de IA: NENHUM\n");

fprintf("Registros 06 e 12: PRESERVADOS\n");

fprintf("Arquivos originais: PRESERVADOS\n");

fprintf("\nDiretorio de saida:\n%s\n",OUT);

fprintf("\nAguardando interpretacao cientifica.\n");

fprintf("========================================\n");

diary off;
