
1;

%% ============================================================
%% GATE 4 - EXPERIMENTO 02.4.2
%% VALIDACAO FINAL DO BOOTSTRAP MULTIVARIADO
%%
%% GNU Octave 11.3
%%
%% Objetivos:
%%   1. Comparar bootstrap convencional e condicionado.
%%   2. Preservar dependencia temporal e trifasica.
%%   3. Auditar heterogeneidade residual no ciclo.
%%   4. Avaliar descontinuidades nas juncoes.
%%   5. Selecionar parametros usando somente treinamento.
%%   6. Auditar a configuracao selecionada na validacao.
%%   7. Emitir decisao final para G4/E03.
%%
%% Nenhum registro original sera alterado.
%% Registros 06 e 12 permanecem reservados.
%% Nenhuma variante definitiva sera salva.
%% ============================================================

clear;
clc;
close all;

pkg load signal;

graphics_toolkit("gnuplot");
set(0,"defaultfigurevisible","off");

rng(42,"twister");

%% ============================================================
%% FUNCOES AUXILIARES
%% ============================================================

function y = rms_local(x)

  y = sqrt(mean(x.^2,1));

endfunction


function q = percentil(x,p)

  x = sort(x(isfinite(x)));

  if isempty(x)
    q = NaN;
    return;
  endif

  n = numel(x);

  k = 1+(n-1)*p/100;

  a = floor(k);
  b = ceil(k);

  q = x(a)+(k-a)*(x(b)-x(a));

endfunction


function e = erro_relativo(a,b)

  e = 100*abs(a-b)./max(abs(b),1e-12);

endfunction


function exportar(h,OUT,nome)

  set(h,"paperpositionmode","auto");

  print(h,fullfile(OUT,nome),"-dpng","-r150");

  close(h);

  fprintf("Figura exportada: %s\n",nome);

endfunction


%% ============================================================
%% REGULARIZACAO TEMPORAL E REDUCAO ZOH
%% ============================================================

function [tu,U,ke,info] = carregar_registro( ...
    arquivo,te,FS,M)

  D = load(arquivo,"results");

  t = D.results.time(:);

  X = D.results.signals(3).values;

  E = D.results.signals(6).values;

  E = E(:);

  if rows(X)!=numel(t)
    X = X';
  endif

  assert(columns(X)==3);
  assert(rows(X)==numel(t));
  assert(numel(E)==numel(t));

  assert(all(isfinite(t)));
  assert(all(isfinite(X(:))));
  assert(all(isfinite(E)));

  assert(all(diff(t)>=0));

  TS = 1/FS;

  tol = 0.001*TS;

  Nu = round((t(end)-t(1))/TS)+1;

  tg = t(1)+(0:Nu-1)'*TS;

  k = round((t-t(1))/TS)+1;

  v = find(k>=1 & k<=Nu);

  v = v(abs(t(v)-tg(k(v)))<=tol);

  contagem = accumarray( ...
      k(v),ones(numel(v),1),[Nu 1],@sum,0);

  assert(all(contagem>0), ...
      "Grade temporal incompleta.");

  idx = accumarray( ...
      k(v),v,[Nu 1],@max,0);

  assert(all(idx>0));

  X = X(idx,:);
  E = E(idx);

  assert(Nu==110000);

  %% Auditoria do marcador.

  mudancas = find(diff(E)!=0)+1;

  assert(numel(mudancas)==1, ...
      "Marcador possui numero inesperado de transicoes.");

  assert(abs(tg(mudancas)-te)<=TS+1e-10);

  %% Identificacao da grade de atualizacao.

  H = zeros(M,3);

  for p=1:3

    ii = find(diff(X(:,p))!=0)+1;

    H(:,p) = accumarray( ...
        mod(ii-1,M)+1, ...
        ones(numel(ii),1), ...
        [M 1],@sum,0);

  endfor

  [~,inicio] = max(sum(H,2));

  idxu = inicio:M:Nu;

  U = X(idxu,:);

  tu = tg(idxu);

  %% Reconstrucao na grade original.

  mapa = floor(((1:Nu)'-inicio)/M)+1;

  ok = mapa>=1 & mapa<=rows(U);

  Z = U(mapa(ok),:);

  erro = max(max(abs(X(ok,:)-Z)));

  assert(erro<=1e-10, ...
      "Reconstrucao ZOH nao exata.");

  ke = find(tu>=te-1e-10,1,"first");

  assert(!isempty(ke));

  info = [numel(t),Nu,rows(U), ...
          inicio,erro,sum(!ok)];

endfunction


%% ============================================================
%% ESTIMATIVA DO PERIODO
%% ============================================================

function [T,acfmax] = estimar_periodo(X,FS,Tref)

  L0 = round(Tref*FS);

  lo = round(0.85*L0);

  hi = round(1.15*L0);

  ac = zeros(hi+1,1);

  usados = 0;

  for p=1:3

    y = X(:,p)-mean(X(:,p));

    if norm(y)<=1e-12
      continue;
    endif

    a = xcorr(y,hi,"coeff");

    ac += a(hi+1:end);

    usados++;

  endfor

  assert(usados>0);

  ac /= usados;

  [acfmax,j] = max(ac(lo+1:hi+1));

  lag = lo+j-1;

  assert(lag>lo && lag<hi, ...
      "Periodo estimado no limite da busca.");

  a = ac(lag);
  b = ac(lag+1);
  c = ac(lag+2);

  den = a-2*b+c;

  delta = 0;

  if abs(den)>1e-12
    delta = 0.5*(a-c)/den;
  endif

  T = (lag+delta)/FS;

endfunction


%% ============================================================
%% EXTRACAO DE CICLOS NORMALIZADOS
%% ============================================================

function C = extrair_ciclos(X,T,FS,N)

  periodo = T*FS;

  nc = floor(rows(X)/periodo);

  C = zeros(N,3,nc);

  for j=1:nc

    inicio = 1+(j-1)*periodo;

    tq = inicio+(0:N-1)'*periodo/N;

    assert(tq(end)<=rows(X));

    for p=1:3

      C(:,p,j) = interp1( ...
          (1:rows(X))', ...
          X(:,p),tq,"linear");

    endfor

  endfor

  assert(all(isfinite(C(:))));

endfunction


%% ============================================================
%% ALINHAMENTO CIRCULAR VIA FFT
%% ============================================================

function [Y,d,erro] = alinhar(X,R)

  N = rows(R);

  escala = max(rms_local(R),1e-6);

  Xn = bsxfun(@rdivide,X,escala);

  Rn = bsxfun(@rdivide,R,escala);

  FX = fft(Xn,[],1);

  FR = fft(Rn,[],1);

  corr = real(ifft(sum(FR.*conj(FX),2)));

  [~,k] = max(corr);

  d = k-1;

  if d>floor(N/2)
    d -= N;
  endif

  Y = circshift(X,d,1);

  D = bsxfun(@rdivide,Y-R,escala);

  erro = sqrt(mean(D(:).^2));

endfunction


%% ============================================================
%% REFERENCIA PERIODICA
%% ============================================================

function [R,H] = construir_referencia(C,NITER)

  nc = size(C,3);

  assert(nc>0);

  R = C(:,:,1);

  H = zeros(NITER,1);

  for it=1:NITER

    A = zeros(size(C));

    for j=1:nc

      A(:,:,j) = alinhar(C(:,:,j),R);

    endfor

    Rnovo = mean(A,3);

    D = Rnovo-R;

    H(it) = sqrt(mean(D(:).^2));

    R = Rnovo;

  endfor

endfunction


%% ============================================================
%% CROSS-FITTING POR REGISTRO
%% ============================================================

function [A,E,refs,shifts] = crossfit( ...
    C,IDS,TREINO,NITER,Rglobal)

  N = size(C,1);

  nc = size(C,3);

  A = zeros(size(C));

  E = zeros(size(C));

  refs = zeros(size(C));

  shifts = zeros(nc,1);

  registros = unique(IDS);

  for ir=1:numel(registros)

    r = registros(ir);

    dst = find(IDS==r);

    src = find(ismember(IDS,TREINO) & IDS!=r);

    assert(!isempty(src));

    R = construir_referencia(C(:,:,src),NITER);

    %% Todas as referencias compartilham a mesma
    %% origem angular da referencia global.

    R = alinhar(R,Rglobal);

    fprintf("Cross-fit R%02d | doadores=%d | ciclos=%d\n", ...
        r,numel(unique(IDS(src))),numel(dst));

    for j=dst(:)'

      [y,d] = alinhar(C(:,:,j),R);

      A(:,:,j) = y;

      E(:,:,j) = y-R;

      refs(:,:,j) = R;

      shifts(j) = d;

    endfor

  endfor

endfunction


%% ============================================================
%% BOOTSTRAP MULTIVARIADO
%%
%% metodo 1: convencional, origem aleatoria no ciclo.
%% metodo 2: condicionado, mesma posicao no ciclo.
%%
%% As tres fases usam o mesmo bloco do mesmo ciclo.
%% Nenhum bloco atravessa artificialmente a fronteira
%% entre dois ciclos doadores.
%% ============================================================

function [Y,J] = bootstrap_ciclos(E,L,metodo,NC)

  N = size(E,1);

  ND = size(E,3);

  Y = zeros(N,3,NC);

  J = false(N-1,NC);

  for c=1:NC

    pos = 1;

    while pos<=N

      n = min(L,N-pos+1);

      doador = randi(ND);

      if metodo==1

        origem = randi(N-n+1);

      else

        origem = pos;

      endif

      Y(pos:pos+n-1,:,c) = ...
          E(origem:origem+n-1,:,doador);

      if pos>1

        J(pos-1,c) = true;

      endif

      pos += n;

    endwhile

  endfor

endfunction


%% ============================================================
%% AUTOCORRELACAO RESIDUAL
%%
%% ACF calculada apenas dentro de cada ciclo.
%% Nao concatena ciclos independentes.
%% ============================================================

function AC = acf_residual(E,ML)

  N = size(E,1);

  NC = size(E,3);

  AC = zeros(ML+1,3);

  for p=1:3

    X = reshape(E(:,p,:),N,NC);

    X = bsxfun(@minus,X,mean(X,1));

    den = sum(X(:).^2);

    if den<=1e-15
      continue;
    endif

    for k=0:ML

      P = X(1:N-k,:).*X(1+k:N,:);

      AC(k+1,p) = sum(P(:))/den;

    endfor

  endfor

endfunction


%% ============================================================
%% PERIODGRAMA RESIDUAL
%%
%% Frequencia nominal equivalente do ciclo normalizado.
%% Nao representa uma nova medicao fisica.
%% ============================================================

function [P,f] = psd_residual(E,FS)

  N = size(E,1);

  NC = size(E,3);

  w = 0.5-0.5*cos(2*pi*(0:N-1)'/N);

  K = floor(N/2)+1;

  P = zeros(K,3);

  for p=1:3

    X = reshape(E(:,p,:),N,NC);

    X = bsxfun(@minus,X,mean(X,1));

    X = bsxfun(@times,X,w);

    F = fft(X,[],1);

    Q = abs(F(1:K,:)).^2/(FS*sum(w.^2));

    if rem(N,2)==0

      Q(2:end-1,:) *= 2;

    else

      Q(2:end,:) *= 2;

    endif

    P(:,p) = mean(Q,2);

  endfor

  f = (0:K-1)'*FS/N;

endfunction


%% ============================================================
%% CARACTERISTICAS FISICAS DOS CICLOS
%% ============================================================

function F = features_ciclos(X,FS)

  NC = size(X,3);

  F = zeros(NC,11);

  for j=1:NC

    Y = X(:,:,j);

    R = rms_local(Y);

    P = max(abs(Y),[],1);

    E = sum(Y.^2,1)/FS;

    tri = sqrt(mean(R.^2));

    deseq = 100*std(R)/max(mean(R),1e-12);

    F(j,:) = [R,P,E,tri,deseq];

  endfor

endfunction


%% ============================================================
%% METRICAS DE CARACTERIZACAO RESIDUAL
%% ============================================================

function S = caracterizar(E,X,FS,ML)

  N = size(E,1);

  NC = size(E,3);

  V = reshape(permute(E,[1 3 2]),N*NC,3);

  S.media = mean(V,1);

  S.rms = rms_local(V);

  S.desvio = std(V,0,1);

  S.corr = corrcoef(V);

  S.acf = acf_residual(E,ML);

  [S.psd,S.f] = psd_residual(E,FS);

  S.perfil = sqrt(mean(E.^2,3));

  S.quant = zeros(5,3);

  pp = [1 5 50 95 99];

  for p=1:3

    for q=1:5

      S.quant(q,p) = percentil(V(:,p),pp(q));

    endfor

  endfor

  S.features = features_ciclos(X,FS);

  S.features_media = mean(S.features,1);

  S.features_p05 = zeros(1,11);

  S.features_p95 = zeros(1,11);

  for k=1:11

    S.features_p05(k) = percentil(S.features(:,k),5);

    S.features_p95(k) = percentil(S.features(:,k),95);

  endfor

  D = diff(E,1,1);

  S.incremento99 = zeros(1,3);

  for p=1:3

    z = abs(D(:,p,:));

    S.incremento99(p) = percentil(z(:),99);

  endfor

endfunction


%% ============================================================
%% AUDITORIA DAS JUNCOES
%%
%% Compara incrementos nas fronteiras artificiais com
%% incrementos naturais nas mesmas posicoes do ciclo.
%% ============================================================

function [razao,frac] = auditar_juncoes(E,J,BASE)

  D = abs(diff(E,1,1));

  B = abs(diff(BASE,1,1));

  razao = ones(1,3);

  frac = zeros(1,3);

  if !any(J(:))
    return;
  endif

  %% Posicoes de juncao desta configuracao.

  pos = find(any(J,2));

  for p=1:3

    d = reshape(D(:,p,:),rows(J),columns(J));

    artificial = d(J);

    natural = B(pos,p,:);

    natural = natural(:);

    limiar = percentil(natural,99);

    q99 = percentil(artificial,99);

    razao(p) = q99/max(limiar,1e-12);

    frac(p) = 100*mean(artificial>limiar);

  endfor

endfunction


%% ============================================================
%% COMPARACAO ESTATISTICA
%%
%% Metricas:
%%  1 - Erro RMS residual (%)
%%  2 - Diferenca de correlacao entre fases
%%  3 - RMSE da ACF
%%  4 - Distancia espectral normalizada
%%  5 - Erro do perfil RMS no ciclo (%)
%%  6 - Erro normalizado dos quantis
%%  7 - Erro RMS do sinal reconstruido (%)
%%  8 - Erro de pico (%)
%%  9 - Erro de energia (%)
%% 10 - Erro RMS trifasico (%)
%% 11 - Erro desequilibrio (pontos percentuais)
%% 12 - Razao P99 das juncoes
%% 13 - Fracao de juncoes acima do P99 natural (%)
%% 14 - Deslocamento da media residual normalizada
%% 15 - Erro dos quantis das features fisicas (%)
%% ============================================================

function M = comparar(A,B,E,J,BASE)

  m1 = max(erro_relativo(A.rms,B.rms));

  C = abs(A.corr-B.corr);

  m2 = max(C(:));

  D = A.acf-B.acf;

  m3 = max(sqrt(mean(D.^2,1)));

  %% Distancia de variacao total entre espectros
  %% normalizados para potencia unitaria.

  PA = bsxfun(@rdivide,A.psd, ...
      max(sum(A.psd,1),1e-15));

  PB = bsxfun(@rdivide,B.psd, ...
      max(sum(B.psd,1),1e-15));

  m4 = max(0.5*sum(abs(PA-PB),1));

  dp = A.perfil-B.perfil;

  m5 = 100*max(rms_local(dp)./ ...
      max(rms_local(B.perfil),1e-12));

  dq = abs(A.quant-B.quant);

  dq = bsxfun(@rdivide,dq, ...
      max(B.rms,1e-12));

  m6 = max(dq(:));

  ef = erro_relativo( ...
      A.features_media,B.features_media);

  m7 = max(ef(1:3));

  m8 = max(ef(4:6));

  m9 = max(ef(7:9));

  m10 = ef(10);

  m11 = abs(A.features_media(11)- ...
            B.features_media(11));

  [razao,frac] = auditar_juncoes(E,J,BASE);

  m12 = max(razao);

  m13 = max(frac);

  m14 = max(abs(A.media-B.media)./ ...
      max(B.rms,1e-12));

  q05 = erro_relativo( ...
      A.features_p05,B.features_p05);

  q95 = erro_relativo( ...
      A.features_p95,B.features_p95);

  m15 = max([q05(1:10),q95(1:10)]);

  M = [m1,m2,m3,m4,m5,m6,m7,m8, ...
       m9,m10,m11,m12,m13,m14,m15];

endfunction


%% ============================================================
%% RECONSTRUCAO PILOTO
%% ============================================================

function X = reconstruir(E,R,MU,ALFA)

  NC = size(E,3);

  X = zeros(size(E));

  for j=1:NC

    X(:,:,j) = R+MU+ALFA*E(:,:,j);

  endfor

endfunction


%% ============================================================
%% ARQUIVOS CSV
%% ============================================================

function salvar_matriz(arquivo,cabecalho,X)

  fid = fopen(arquivo,"w");

  assert(fid>0);

  fprintf(fid,"%s\n",cabecalho);

  fclose(fid);

  dlmwrite(arquivo,X, ...
      "-append","delimiter",",","precision","%.12g");

endfunction


%% ============================================================
%% CONFIGURACOES
%% ============================================================

ROOT = [ ...
  "PASTA_DO_USUARIO/Documents/" ...
  "Pesquisa_SRM/TesteNo1_31OCT2026/" ...
  "Data and Scripts"];

RAW = fullfile(ROOT,"Results_1_to_12");

TREINO = [1 2 4 5 7 8 10 11];

VALIDACAO = [3 9];

TESTE = [6 12];

DEV = sort([TREINO VALIDACAO]);

assert(isempty(intersect(DEV,TESTE)));

EVENTOS = [ ...
  1.0017 1.0019 1.0021 1.0024 ...
  1.0026 1.0030 1.0040 1.0050 ...
  1.0060 1.0070 1.0075 1.0080];

FS_ORIGINAL = 200000;

MZOH = 8;

FS = FS_ORIGINAL/MZOH;

N = 327;

TREF = 0.01308;

TRANSITORIO = 0.200;

NITER = 3;

BLOCOS = [16 32 64 128 327];

METODOS = {"Convencional","Condicionado"};

%% Campanha deliberadamente pequena.

NREP = 20;

NC_PILOTO = 64;

MAXLAG = 80;

%% Intensidade integral para auditoria estatistica.
%% Nao constitui autorizacao de uso em toda a trajetoria.

ALFA = 1.0;

%% Limites operacionais de engenharia.
%% Nao sao intervalos de confianca nem tolerancias
%% fisicas certificadas do motor.
%%
%% Ordem identica a saida de comparar().

LIMITES = [ ...
  15, ...    % RMS residual
  0.12, ...  % correlacao
  0.08, ...  % ACF
  0.20, ...  % espectro
  20, ...    % perfil angular
  0.65, ...  % quantis residuais
  3, ...     % RMS reconstruido
  5, ...     % pico
  6, ...     % energia
  3, ...     % RMS trifasico
  3, ...     % desequilibrio em p.p.
  2, ...     % incremento nas juncoes
  5, ...     % excedencia nas juncoes
  0.20, ...  % media residual
  8];        % quantis das features fisicas

%% Requer aprovacao simultanea em pelo menos 90%
%% das repeticoes e mediana dentro de todos os limites.

FRAC_APROVACAO = 0.90;

%% ============================================================
%% DIRETORIO
%% ============================================================

runid = 1;

while true

  OUT = fullfile(ROOT, ...
      sprintf("Gate4_Exp02_4_2_Run%02d",runid));

  if exist(OUT,"dir")==0
    break;
  endif

  runid++;

endwhile

mkdir(OUT);

diary(fullfile(OUT,"G4_E02_4_2_Console.txt"));

fprintf("\n========================================\n");
fprintf(" GATE 4 - EXPERIMENTO 02.4.2\n");
fprintf(" VALIDACAO FINAL DO BOOTSTRAP\n");
fprintf("========================================\n");

fprintf("Fs original: %.0f Hz\n",FS_ORIGINAL);
fprintf("Fs canonica: %.0f Hz\n",FS);
fprintf("Amostras por ciclo: %d\n",N);
fprintf("Repeticoes por configuracao: %d\n",NREP);
fprintf("Ciclos por repeticao: %d\n",NC_PILOTO);

fprintf("Registros 06 e 12: RESERVADOS\n");
fprintf("Saida: %s\n",OUT);

TSTART = tic();

%% ============================================================
%% ETAPA 1 - EXTRACAO DOS CICLOS TARDIOS
%% ============================================================

fprintf("\n========================================\n");
fprintf(" EXTRACAO DOS CICLOS TARDIOS\n");
fprintf("========================================\n");

C = zeros(N,3,0);

IDS = [];

META = [];

for r=DEV

  arquivo = fullfile(RAW, ...
      sprintf("results_%d.mat",r));

  assert(exist(arquivo,"file")==2, ...
      sprintf("Arquivo ausente: %s",arquivo));

  te = EVENTOS(r);

  [tu,U,ke,info] = carregar_registro( ...
      arquivo,te,FS_ORIGINAL,MZOH);

  mask = tu>=te+TRANSITORIO-1e-10;

  X = U(mask,:);

  [T,acfmax] = estimar_periodo(X,FS,TREF);

  Cr = extrair_ciclos(X,T,FS,N);

  nc = size(Cr,3);

  assert(nc>=5);

  C = cat(3,C,Cr);

  IDS = [IDS;repmat(r,nc,1)];

  grupo = 1+ismember(r,VALIDACAO);

  META = [META; ...
      r,grupo,T*1000,acfmax,nc,info];

  fprintf(["R%02d | grupo %d | periodo=%.8f ms" ...
           " | ACF=%.6f | ciclos=%d | ZOH=%.6g A\n"], ...
           r,grupo,T*1000,acfmax,nc,info(5));

  clear U X Cr;

endfor

IT = find(ismember(IDS,TREINO));

IV = find(ismember(IDS,VALIDACAO));

fprintf("\nCiclos de treinamento: %d\n",numel(IT));

fprintf("Ciclos de validacao: %d\n",numel(IV));

%% ============================================================
%% ETAPA 2 - REFERENCIAS CROSS-FIT
%% ============================================================

fprintf("\n========================================\n");
fprintf(" REFERENCIAS CROSS-FIT\n");
fprintf("========================================\n");

[Rglobal,H] = construir_referencia( ...
    C(:,:,IT),NITER);

[A,E,REFS,SHIFTS] = crossfit( ...
    C,IDS,TREINO,NITER,Rglobal);

%% Residuos de treinamento exclusivamente.

ET = E(:,:,IT);

EV = E(:,:,IV);

%% Media residual condicionada a posicao no ciclo.
%% Calculada exclusivamente no treinamento.
%%
%% A media sistematica nao deve ser confundida
%% com variabilidade aleatoria.

MU = mean(ET,3);

ETC = bsxfun(@minus,ET,MU);

EVC = bsxfun(@minus,EV,MU);

%% Referencia comum para comparar o efeito isolado
%% da reamostragem residual.

XT = reconstruir(ETC,Rglobal,MU,ALFA);

XV = reconstruir(EVC,Rglobal,MU,ALFA);

ST = caracterizar(ETC,XT,FS,MAXLAG);

SV = caracterizar(EVC,XV,FS,MAXLAG);

%% ============================================================
%% ETAPA 3 - ESTATISTICAS RESIDUAIS
%% ============================================================

fprintf("\n========================================\n");
fprintf(" ESTATISTICAS RESIDUAIS\n");
fprintf("========================================\n");

FASES = {"A","B","C"};

RESID = [];

for p=1:3

  original = ET(:,p,:);

  centrado = ETC(:,p,:);

  media_original = mean(original(:));

  media_centrada = mean(centrado(:));

  rms_original = sqrt(mean(original(:).^2));

  rms_centrado = sqrt(mean(centrado(:).^2));

  rms_mu = sqrt(mean(MU(:,p).^2));

  fracao_mu = 100*rms_mu^2/ ...
      max(rms_original^2,1e-12);

  fprintf("\nFase %s\n",FASES{p});

  fprintf("Media residual original: %.8f A\n", ...
      media_original);

  fprintf("Media residual centrada: %.8f A\n", ...
      media_centrada);

  fprintf("RMS residual original: %.8f A\n", ...
      rms_original);

  fprintf("RMS residual centrado: %.8f A\n", ...
      rms_centrado);

  fprintf("RMS componente sistematica: %.8f A\n", ...
      rms_mu);

  fprintf("Energia residual sistematica: %.6f %%\n", ...
      fracao_mu);

  fprintf("P01/P99 centrado: %.8f / %.8f A\n", ...
      ST.quant(1,p),ST.quant(5,p));

  RESID = [RESID; ...
      p,media_original,media_centrada, ...
      rms_original,rms_centrado,rms_mu, ...
      fracao_mu,ST.quant(1,p),ST.quant(5,p)];

endfor

fprintf("\nCorrelacao residual centrada - treino:\n");

disp(ST.corr);

fprintf("Correlacao residual centrada - validacao:\n");

disp(SV.corr);

%% ============================================================
%% ETAPA 4 - CAMPANHA PILOTO
%% ============================================================

fprintf("\n========================================\n");
fprintf(" CAMPANHA PILOTO - TREINAMENTO\n");
fprintf("========================================\n");

ALL = [];

SUM = [];

DETALHES = cell(2,numel(BLOCOS));

for metodo=1:2

  for ib=1:numel(BLOCOS)

    L = BLOCOS(ib);

    MM = zeros(NREP,numel(LIMITES));

    AP = false(NREP,1);

    for rep=1:NREP

      %% Sementes reproduziveis e correspondentes.

      rng(42000+rep,"twister");

      [ES,J] = bootstrap_ciclos( ...
          ETC,L,metodo,NC_PILOTO);

      XS = reconstruir(ES,Rglobal,MU,ALFA);

      SS = caracterizar(ES,XS,FS,MAXLAG);

      MM(rep,:) = comparar(SS,ST,ES,J,ETC);

      AP(rep) = all(isfinite(MM(rep,:))) && ...
                all(MM(rep,:)<=LIMITES);

      ALL = [ALL; ...
          metodo,L,rep,AP(rep),MM(rep,:)];

    endfor

    med = median(MM,1);

    frac = mean(AP);

    %% Custo normalizado pelos limites operacionais.
    %% Nao constitui uma medida universal de qualidade.

    score = mean(med./LIMITES);

    aprovado = all(isfinite(med)) && ...
               all(med<=LIMITES) && ...
               frac>=FRAC_APROVACAO;

    SUM = [SUM; ...
        metodo,L,frac,score,aprovado,med];

    DETALHES{metodo,ib} = MM;

    fprintf("\n%s | bloco=%d | %.3f ms\n", ...
        METODOS{metodo},L,1000*L/FS);

    fprintf("Aprovacao conjunta: %.2f %%\n",100*frac);

    fprintf("Score: %.8f\n",score);

    fprintf("Erro RMS residual: %.4f %%\n",med(1));

    fprintf("Erro correlacao: %.6f\n",med(2));

    fprintf("Erro ACF: %.6f\n",med(3));

    fprintf("Distancia espectral: %.6f\n",med(4));

    fprintf("Erro perfil angular: %.4f %%\n",med(5));

    fprintf("Erro quantis residuais: %.6f\n",med(6));

    fprintf("Erro RMS reconstruido: %.4f %%\n",med(7));

    fprintf("Erro pico reconstruido: %.4f %%\n",med(8));

    fprintf("Erro energia: %.4f %%\n",med(9));

    fprintf("Erro RMS trifasico: %.4f %%\n",med(10));

    fprintf("Erro desequilibrio: %.4f p.p.\n",med(11));

    fprintf("Razao P99 juncoes: %.6f\n",med(12));

    fprintf("Excedencia juncoes: %.4f %%\n",med(13));

    fprintf("Erro media residual: %.6f\n",med(14));

    fprintf("Erro quantis features: %.4f %%\n",med(15));

    fprintf("Aprovado: %d\n",aprovado);

  endfor

endfor

%% ============================================================
%% ETAPA 5 - SELECAO EXCLUSIVA NO TREINAMENTO
%%
%% Apenas o metodo condicionado e elegivel para uso.
%% O convencional funciona como comparador experimental.
%% ============================================================

fprintf("\n========================================\n");
fprintf(" SELECAO DOS PARAMETROS\n");
fprintf("========================================\n");

elegiveis = find(SUM(:,1)==2 & SUM(:,5)==1);

tem_candidato = !isempty(elegiveis);

METODO_FINAL = 0;

BLOCO_FINAL = 0;

APROVADO_TREINO = false;

APROVADO_VAL = false;

STATUS = "BOOTSTRAP_EXCLUIDO";

VAL = zeros(0,16);

VAL_ORIGINAL = NaN(1,numel(LIMITES));

SSEL = [];

ESSEL = [];

XSSEL = [];

JSEL = [];

if tem_candidato

  [~,ii] = min(SUM(elegiveis,4));

  selecionado = elegiveis(ii);

  METODO_FINAL = SUM(selecionado,1);

  BLOCO_FINAL = SUM(selecionado,2);

  APROVADO_TREINO = true;

  fprintf("Metodo selecionado: %s\n", ...
      METODOS{METODO_FINAL});

  fprintf("Bloco selecionado: %d amostras\n", ...
      BLOCO_FINAL);

  fprintf("Duracao nominal: %.8f ms\n", ...
      1000*BLOCO_FINAL/FS);

  fprintf("Score treino: %.8f\n", ...
      SUM(selecionado,4));

  fprintf("Aprovacao treino: %.2f %%\n", ...
      100*SUM(selecionado,3));

else

  fprintf("Nenhum candidato condicionado aprovado.\n");

  fprintf("Bootstrap sera excluido da expansao.\n");

  fprintf("Nenhum reteste ou ajuste de limites.\n");

endif

%% ============================================================
%% ETAPA 6 - AUDITORIA NA VALIDACAO
%%
%% Nenhum parametro pode ser modificado nesta etapa.
%% Nenhum residuo da validacao entra no banco doador.
%% ============================================================

fprintf("\n========================================\n");
fprintf(" AUDITORIA FINAL - VALIDACAO\n");
fprintf("========================================\n");

if tem_candidato

  MMV = zeros(NREP,numel(LIMITES));

  APV = false(NREP,1);

  %% Diferenca natural entre treinamento e validacao.
  %% Apenas para interpretacao descritiva.

  JN = false(N-1,size(EVC,3));

  VAL_ORIGINAL = comparar(SV,ST,EVC,JN,ETC);

  fprintf("\nDiferencas naturais validacao/treino:\n");

  disp(VAL_ORIGINAL);

  for rep=1:NREP

    rng(84000+rep,"twister");

    [ES,J] = bootstrap_ciclos( ...
        ETC,BLOCO_FINAL,METODO_FINAL,NC_PILOTO);

    XS = reconstruir(ES,Rglobal,MU,ALFA);

    SS = caracterizar(ES,XS,FS,MAXLAG);

    MMV(rep,:) = comparar(SS,SV,ES,J,EVC);

    APV(rep) = all(isfinite(MMV(rep,:))) && ...
               all(MMV(rep,:)<=LIMITES);

    VAL = [VAL;rep,MMV(rep,:)];

    if rep==1

      ESSEL = ES;

      XSSEL = XS;

      JSEL = J;

      SSEL = SS;

    endif

  endfor

  medv = median(MMV,1);

  fracv = mean(APV);

  APROVADO_VAL = ...
      all(isfinite(medv)) && ...
      all(medv<=LIMITES) && ...
      fracv>=FRAC_APROVACAO;

  fprintf("\nAprovacao conjunta: %.2f %%\n",100*fracv);

  fprintf("Erro RMS residual: %.4f %%\n",medv(1));

  fprintf("Erro correlacao: %.6f\n",medv(2));

  fprintf("Erro ACF: %.6f\n",medv(3));

  fprintf("Distancia espectral: %.6f\n",medv(4));

  fprintf("Erro perfil angular: %.4f %%\n",medv(5));

  fprintf("Erro quantis residuais: %.6f\n",medv(6));

  fprintf("Erro RMS reconstruido: %.4f %%\n",medv(7));

  fprintf("Erro pico: %.4f %%\n",medv(8));

  fprintf("Erro energia: %.4f %%\n",medv(9));

  fprintf("Erro RMS trifasico: %.4f %%\n",medv(10));

  fprintf("Erro desequilibrio: %.4f p.p.\n",medv(11));

  fprintf("Razao P99 juncoes: %.6f\n",medv(12));

  fprintf("Excedencia juncoes: %.4f %%\n",medv(13));

  fprintf("Erro media residual: %.6f\n",medv(14));

  fprintf("Erro quantis features: %.4f %%\n",medv(15));

  if APROVADO_VAL

    STATUS = "BOOTSTRAP_APROVADO_PARA_EXPANSAO";

  else

    STATUS = "BOOTSTRAP_EXCLUIDO";

    fprintf("Configuracao nao aprovada na validacao.\n");

    fprintf("Nao sera escolhido outro bloco usando validacao.\n");

  endif

else

  fprintf("Auditoria do candidato nao aplicavel.\n");

endif

%% ============================================================
%% ETAPA 7 - EXPORTACAO DOS RESULTADOS
%% ============================================================

fprintf("\n========================================\n");
fprintf(" EXPORTACAO DOS RESULTADOS\n");
fprintf("========================================\n");

CAB_MET = [ ...
  "erro_rms_residual_pct," ...
  "erro_correlacao," ...
  "erro_acf," ...
  "distancia_espectral," ...
  "erro_perfil_angular_pct," ...
  "erro_quant_residuais," ...
  "erro_rms_sinal_pct," ...
  "erro_pico_pct," ...
  "erro_energia_pct," ...
  "erro_rms_trifasico_pct," ...
  "erro_desequilibrio_pp," ...
  "razao_p99_juncoes," ...
  "excedencia_juncoes_pct," ...
  "erro_media_residual," ...
  "erro_quant_features_pct"];

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Extracao.csv"), ...
    ["registro,grupo,periodo_ms,acf,ciclos," ...
     "amostras_brutas,amostras_regulares," ...
     "amostras_canonicas,origem_grade," ...
     "erro_zoh_A,amostras_nao_reconstruidas"], ...
    META);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Residuos.csv"), ...
    ["fase,media_original_A,media_centrada_A," ...
     "rms_original_A,rms_centrado_A," ...
     "rms_media_sistematica_A," ...
     "energia_sistematica_pct,p01_A,p99_A"], ...
    RESID);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Repeticoes.csv"), ...
    ["metodo,bloco,repeticao,aprovado," CAB_MET], ...
    ALL);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Resumo.csv"), ...
    ["metodo,bloco,fracao_aprovada,score," ...
     "aprovado," CAB_MET], ...
    SUM);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Limites.csv"), ...
    CAB_MET,LIMITES);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Correlacao_Treino.csv"), ...
    "A,B,C",ST.corr);

salvar_matriz( ...
    fullfile(OUT,"G4_E02_4_2_Correlacao_Validacao.csv"), ...
    "A,B,C",SV.corr);

if tem_candidato

  salvar_matriz( ...
      fullfile(OUT,"G4_E02_4_2_Validacao.csv"), ...
      ["repeticao," CAB_MET],VAL);

  salvar_matriz( ...
      fullfile(OUT,"G4_E02_4_2_Diferenca_Natural.csv"), ...
      CAB_MET,VAL_ORIGINAL);

endif

%% ============================================================
%% ETAPA 8 - FIGURAS
%% ============================================================

fprintf("\n========================================\n");
fprintf(" EXPORTACAO DAS FIGURAS\n");
fprintf("========================================\n");

nomes = { ...
  "RMS residual", ...
  "Correlacao", ...
  "ACF", ...
  "Espectro", ...
  "Perfil angular", ...
  "Quantis residuais", ...
  "RMS sinal", ...
  "Pico", ...
  "Energia", ...
  "RMS trifasico", ...
  "Desequilibrio", ...
  "P99 juncoes", ...
  "Excedencia", ...
  "Media residual", ...
  "Quantis features"};

theta = (0:N-1)'*360/N;

%% Figura 01 - Componente residual sistematica.

h = figure("position",[100 100 1100 850]);

for p=1:3

  subplot(3,1,p);

  plot(theta,MU(:,p),"linewidth",1.3);

  grid on;

  ylabel("Corrente (A)");

  title(sprintf("Fase %s - media residual",FASES{p}));

  if p==3
    xlabel("Posicao no ciclo (graus)");
  endif

endfor

exportar(h,OUT,"G4_E02_4_2_01_Media_Residual.png");

%% Figura 02 - Comparacao das correlacoes.

h = figure("position",[100 100 1200 550]);

subplot(1,2,1);

imagesc(ST.corr,[-1 1]);

axis square;

colorbar;

set(gca,"xtick",1:3,"xticklabel",FASES);
set(gca,"ytick",1:3,"yticklabel",FASES);

title("Treinamento");

subplot(1,2,2);

imagesc(SV.corr,[-1 1]);

axis square;

colorbar;

set(gca,"xtick",1:3,"xticklabel",FASES);
set(gca,"ytick",1:3,"yticklabel",FASES);

title("Validacao");

exportar(h,OUT,"G4_E02_4_2_02_Correlacoes.png");

%% Figura 03 - Score por comprimento de bloco.

h = figure("position",[100 100 1100 650]);

Y = zeros(numel(BLOCOS),2);

for m=1:2
  Y(:,m) = SUM(SUM(:,1)==m,4);
endfor

bar(Y);

set(gca,"xtick",1:numel(BLOCOS));

set(gca,"xticklabel", ...
    arrayfun(@num2str,BLOCOS,"uniformoutput",false));

xlabel("Comprimento do bloco (amostras)");

ylabel("Score normalizado");

title("Comparacao das configuracoes");

legend(METODOS,"location","northoutside", ...
    "orientation","horizontal");

grid on;

exportar(h,OUT,"G4_E02_4_2_03_Score.png");

%% Figura 04 - Aprovacao conjunta.

h = figure("position",[100 100 1100 650]);

for m=1:2
  Y(:,m) = 100*SUM(SUM(:,1)==m,3);
endfor

bar(Y);

hold on;

plot([0.5 numel(BLOCOS)+0.5], ...
    100*FRAC_APROVACAO*[1 1],"k--","linewidth",1.5);

set(gca,"xtick",1:numel(BLOCOS));

set(gca,"xticklabel", ...
    arrayfun(@num2str,BLOCOS,"uniformoutput",false));

ylim([0 105]);

xlabel("Comprimento do bloco (amostras)");

ylabel("Repeticoes aprovadas (%)");

title("Aprovacao simultanea dos criterios");

legend({"Convencional","Condicionado","Limite"}, ...
    "location","northoutside", ...
    "orientation","horizontal");

grid on;

exportar(h,OUT,"G4_E02_4_2_04_Aprovacao.png");

%% Figura 05 - Matriz de desempenho normalizado.

h = figure("position",[100 100 1700 850]);

Q = bsxfun(@rdivide,SUM(:,6:end),LIMITES);

imagesc(log10(max(Q,1e-4)));

colorbar;

labels = cell(rows(SUM),1);

for j=1:rows(SUM)

  labels{j} = sprintf("%s-%d", ...
      METODOS{SUM(j,1)},SUM(j,2));

endfor

set(gca,"ytick",1:rows(SUM),"yticklabel",labels);

set(gca,"xtick",1:numel(nomes),"xticklabel",nomes);

xtickangle(45);

title("Log10 da razao metrica / limite");

exportar(h,OUT,"G4_E02_4_2_05_Matriz_Criterios.png");

%% Figura 06 - Dependencia temporal.

h = figure("position",[100 100 1250 900]);

for p=1:3

  subplot(3,1,p);

  lag = (0:MAXLAG)'/FS*1000;

  plot(lag,ST.acf(:,p),"linewidth",1.5);

  hold on;

  plot(lag,SV.acf(:,p),"--","linewidth",1.3);

  if tem_candidato

    plot(lag,SSEL.acf(:,p),"linewidth",1.2);

    lg = {"Treino","Validacao","Bootstrap"};

  else

    lg = {"Treino","Validacao"};

  endif

  plot([0 lag(end)],[0 0],"k:");

  title(sprintf("Fase %s",FASES{p}));

  ylabel("ACF");

  grid on;

  if p==1

    legend(lg,"location","northeast");

  endif

  if p==3

    xlabel("Atraso (ms)");

  endif

endfor

exportar(h,OUT,"G4_E02_4_2_06_ACF.png");

%% Figura 07 - Perfil angular.

h = figure("position",[100 100 1250 900]);

for p=1:3

  subplot(3,1,p);

  plot(theta,ST.perfil(:,p),"linewidth",1.4);

  hold on;

  plot(theta,SV.perfil(:,p),"--","linewidth",1.3);

  if tem_candidato

    plot(theta,SSEL.perfil(:,p),"linewidth",1.1);

  endif

  title(sprintf("Fase %s",FASES{p}));

  ylabel("RMS residual (A)");

  grid on;

  if p==1

    legend(lg,"location","northeast");

  endif

  if p==3

    xlabel("Posicao no ciclo (graus)");

  endif

endfor

exportar(h,OUT,"G4_E02_4_2_07_Perfil_Angular.png");

%% Figura 08 - PSD residual.

h = figure("position",[100 100 1250 900]);

for p=1:3

  subplot(3,1,p);

  plot(ST.f/1000, ...
      10*log10(max(ST.psd(:,p),1e-18)), ...
      "linewidth",1.3);

  hold on;

  plot(SV.f/1000, ...
      10*log10(max(SV.psd(:,p),1e-18)), ...
      "--","linewidth",1.2);

  if tem_candidato

    plot(SSEL.f/1000, ...
      10*log10(max(SSEL.psd(:,p),1e-18)), ...
      "linewidth",1.1);

  endif

  xlim([0 FS/2000]);

  title(sprintf("Fase %s",FASES{p}));

  ylabel("PSD (dB A^2/Hz)");

  grid on;

  if p==1

    legend(lg,"location","northeast");

  endif

  if p==3

    xlabel("Frequencia nominal equivalente (kHz)");

  endif

endfor

exportar(h,OUT,"G4_E02_4_2_08_PSD.png");

%% Figura 09 - Efeito das juncoes.

h = figure("position",[100 100 1200 800]);

subplot(2,1,1);

for m=1:2

  %% Coluna 17: metrica 12.

  Y(:,m) = SUM(SUM(:,1)==m,17);

endfor

bar(Y);

hold on;

plot([0.5 5.5],LIMITES(12)*[1 1],"k--");

ylabel("Razao P99");

title("Magnitude dos incrementos nas juncoes");

set(gca,"xtick",1:5,"xticklabel", ...
    arrayfun(@num2str,BLOCOS,"uniformoutput",false));

legend({"Convencional","Condicionado","Limite"}, ...
    "location","northoutside", ...
    "orientation","horizontal");

grid on;

subplot(2,1,2);

for m=1:2

  %% Coluna 18: metrica 13.

  Y(:,m) = SUM(SUM(:,1)==m,18);

endfor

bar(Y);

hold on;

plot([0.5 5.5],LIMITES(13)*[1 1],"k--");

set(gca,"xtick",1:5,"xticklabel", ...
    arrayfun(@num2str,BLOCOS,"uniformoutput",false));

xlabel("Comprimento do bloco (amostras)");

ylabel("Excedencia (%)");

grid on;

exportar(h,OUT,"G4_E02_4_2_09_Juncoes.png");

%% Figuras adicionais da configuracao selecionada.

if tem_candidato

  %% Figura 10 - Ciclos piloto.

  h = figure("position",[100 100 1250 900]);

  for p=1:3

    subplot(3,1,p);

    hold on;

    for j=1:min(8,size(XSSEL,3))

      plot(theta,XSSEL(:,p,j), ...
          "color",[0.75 0.75 0.75]);

    endfor

    plot(theta,Rglobal(:,p)+MU(:,p), ...
        "k","linewidth",1.8);

    plot(theta,XSSEL(:,p,1), ...
        "linewidth",1.3);

    ylabel("Corrente (A)");

    title(sprintf("Fase %s",FASES{p}));

    grid on;

    if p==3

      xlabel("Posicao no ciclo (graus)");

    endif

  endfor

  exportar(h,OUT,"G4_E02_4_2_10_Ciclos_Piloto.png");

  %% Figura 11 - Residuos e fronteiras.

  h = figure("position",[100 100 1250 900]);

  for p=1:3

    subplot(3,1,p);

    plot(theta,ESSEL(:,p,1),"linewidth",1.2);

    hold on;

    yl = ylim();

    jj = find(JSEL(:,1));

    for j=jj(:)'

      plot(theta(j+1)*[1 1],yl,"k:");

    endfor

    ylabel("Residuo (A)");

    title(sprintf("Fase %s",FASES{p}));

    grid on;

    if p==3

      xlabel("Posicao no ciclo (graus)");

    endif

  endfor

  exportar(h,OUT,"G4_E02_4_2_11_Residuos_Juncoes.png");

  %% Figura 12 - Distribuicoes residuais.

  h = figure("position",[100 100 1200 900]);

  for p=1:3

    subplot(3,1,p);

    z1 = ETC(:,p,:);

    z2 = ESSEL(:,p,:);

    z3 = EVC(:,p,:);

    lo = min([z1(:);z2(:);z3(:)]);

    hi = max([z1(:);z2(:);z3(:)]);

    xx = linspace(lo,hi,90);

    a = hist(z1(:),xx);

    b = hist(z2(:),xx);

    c = hist(z3(:),xx);

    dx = xx(2)-xx(1);

    plot(xx,a/(sum(a)*dx),"linewidth",1.4);

    hold on;

    plot(xx,b/(sum(b)*dx),"linewidth",1.2);

    plot(xx,c/(sum(c)*dx),"--","linewidth",1.2);

    title(sprintf("Fase %s",FASES{p}));

    ylabel("Densidade (1/A)");

    grid on;

    if p==1

      legend({"Treino","Bootstrap","Validacao"}, ...
          "location","northeast");

    endif

    if p==3

      xlabel("Corrente residual (A)");

    endif

  endfor

  exportar(h,OUT,"G4_E02_4_2_12_Distribuicoes.png");

  %% Figura 13 - Caracteristicas fisicas.

  h = figure("position",[100 100 1450 950]);

  FF = {ST.features, ...
        SSEL.features,SV.features};

  FLO = {"RMS A","RMS B","RMS C", ...
         "Pico A","Pico B","Pico C", ...
         "Energia A","Energia B","Energia C"};

  for p=1:9

    subplot(3,3,p);

    mm = zeros(3,1);

    low = zeros(3,1);

    high = zeros(3,1);

    for j=1:3

      z = FF{j}(:,p);

      mm(j) = median(z);

      low(j) = mm(j)-percentil(z,5);

      high(j) = percentil(z,95)-mm(j);

    endfor

    errorbar(1:3,mm,low,high,"o");

    xlim([0.5 3.5]);

    set(gca,"xtick",1:3, ...
        "xticklabel",{"Treino","Piloto","Val."});

    title(FLO{p});

    if p<=6
      ylabel("A");
    else
      ylabel("A^2 s");
    endif

    grid on;

  endfor

  exportar(h,OUT,"G4_E02_4_2_13_Features.png");

  %% Figura 14 - RMS trifasico e desequilibrio.

  h = figure("position",[100 100 1050 800]);

  for p=10:11

    subplot(2,1,p-9);

    hold on;

    for j=1:3

      z = sort(FF{j}(:,p));

      F = (1:numel(z))'/numel(z);

      plot(z,F,"linewidth",1.3);

    endfor

    ylabel("Probabilidade acumulada");

    grid on;

    if p==10

      xlabel("RMS trifasico (A)");

      title("Distribuicao do RMS trifasico");

      legend({"Treino","Piloto","Validacao"}, ...
          "location","southeast");

    else

      xlabel("Desequilibrio RMS (%)");

      title("Distribuicao do desequilibrio");

    endif

  endfor

  exportar(h,OUT,"G4_E02_4_2_14_Features_Globais.png");

  %% Figura 15 - Auditoria final.

  h = figure("position",[100 100 1600 700]);

  T = SUM(selecionado,6:end)./LIMITES;

  V = medv./LIMITES;

  bar([T(:),V(:)]);

  hold on;

  plot([0.5 numel(nomes)+0.5], ...
      [1 1],"k--","linewidth",1.5);

  set(gca,"xtick",1:numel(nomes), ...
      "xticklabel",nomes);

  xtickangle(45);

  ylabel("Razao metrica / limite");

  title("Auditoria da configuracao selecionada");

  legend({"Treinamento","Validacao","Limite"}, ...
      "location","northoutside", ...
      "orientation","horizontal");

  grid on;

  exportar(h,OUT,"G4_E02_4_2_15_Auditoria_Final.png");

endif

%% ============================================================
%% ETAPA 9 - DECISAO FINAL
%% ============================================================

fprintf("\n========================================\n");
fprintf(" DECISAO FINAL PARA G4/E03\n");
fprintf("========================================\n");

fprintf("Status: %s\n",STATUS);

fprintf("Aprovado no treinamento: %d\n", ...
    APROVADO_TREINO);

fprintf("Aprovado na validacao: %d\n", ...
    APROVADO_VAL);

fprintf("\nPRE-EVENTO:\n");

fprintf("Bootstrap inter-registros: NAO AUTORIZADO\n");

fprintf("AWGN e perturbacoes instrumentais: ELEGIVEIS\n");

fprintf("\nTRANSITORIO:\n");

fprintf("Bootstrap estacionario: NAO AUTORIZADO\n");

fprintf("Preservacao da trajetoria: OBRIGATORIA\n");

fprintf("AWGN e perturbacoes controladas: ELEGIVEIS\n");

fprintf("\nTARDIO:\n");

if APROVADO_TREINO && APROVADO_VAL

  fprintf("Bootstrap condicionado ao ciclo: ELEGIVEL\n");

  fprintf("Bloco nominal: %d amostras\n",BLOCO_FINAL);

  fprintf("Duracao nominal: %.8f ms\n", ...
      1000*BLOCO_FINAL/FS);

  fprintf("Fases reamostradas conjuntamente\n");

  fprintf("Doadores restritos ao treinamento\n");

  if BLOCO_FINAL==N

    fprintf("Modalidade: reamostragem de ciclos inteiros\n");

    fprintf("Juncoes internas: INEXISTENTES\n");

    fprintf("Diversidade obtida por recombinacao: LIMITADA\n");

  else

    fprintf("Modalidade: blocos condicionados ao ciclo\n");

  endif

  fprintf("Juncoes entre ciclos completos: NAO VALIDADAS\n");

  fprintf("Expansao deve preservar continuidade temporal\n");

else

  fprintf("Bootstrap: EXCLUIDO\n");

  fprintf("Utilizar AWGN e perturbacoes controladas\n");

  fprintf("Nao realizar nova selecao de parametros\n");

endif

%% ============================================================
%% PROTOCOLO E ARQUIVO DE DECISAO
%% ============================================================

f = fopen(fullfile(OUT, ...
    "G4_E02_4_2_Decisao.txt"),"w");

assert(f>0);

fprintf(f,"GATE 4 - EXPERIMENTO 02.4.2\n");

fprintf(f,"VALIDACAO FINAL DO BOOTSTRAP\n\n");

fprintf(f,"Status: %s\n",STATUS);

fprintf(f,"Aprovado_treino=%d\n",APROVADO_TREINO);

fprintf(f,"Aprovado_validacao=%d\n",APROVADO_VAL);

fprintf(f,"Fs_canonica=%d\n",FS);

fprintf(f,"Amostras_ciclo=%d\n",N);

fprintf(f,"Repeticoes=%d\n",NREP);

fprintf(f,"Ciclos_piloto_repeticao=%d\n",NC_PILOTO);

fprintf(f,"Intensidade_residual_piloto=%.6f\n",ALFA);

fprintf(f,"Registros_treino=");

fprintf(f,"%d ",TREINO);

fprintf(f,"\nRegistros_validacao=");

fprintf(f,"%d ",VALIDACAO);

fprintf(f,"\nRegistros_reservados=6 12\n");

fprintf(f,"Ciclos_treino=%d\n",numel(IT));

fprintf(f,"Ciclos_validacao=%d\n",numel(IV));

fprintf(f,"Metodo_candidato=%d\n",METODO_FINAL);

fprintf(f,"Bloco_candidato=%d\n",BLOCO_FINAL);

fprintf(f,"Bootstrap_pre=0\n");

fprintf(f,"Bootstrap_transitorio=0\n");

fprintf(f,"Bootstrap_tardio=%d\n", ...
    APROVADO_TREINO && APROVADO_VAL);

fprintf(f,"Variantes_definitivas=0\n");

fprintf(f,"Treinamento_IA=0\n");

fprintf(f,"\nLIMITACOES:\n");

fprintf(f, ...
    "Trajetoria pre-evento comum nao independente.\n");

fprintf(f, ...
    "Expansao artificial nao cria ensaios fisicos independentes.\n");

fprintf(f, ...
    "Validacao limitada aos registros de desenvolvimento.\n");

fprintf(f, ...
    "Ciclos normalizados nao equivalem a novos ensaios fisicos.\n");

fprintf(f, ...
    "Limites de aprovacao sao operacionais e exploratorios.\n");

fprintf(f, ...
    "Aprovacao restrita a distribuicoes de ciclos tardios.\n");

fprintf(f, ...
    "Juncoes entre ciclos completos nao foram validadas.\n");

fprintf(f, ...
    "Continuidade e preservacao da trajetoria serao requisitos da expansao.\n");

fprintf(f, ...
    "Nenhum parametro foi selecionado pela validacao.\n");

fprintf(f, ...
    "Nenhuma diferenca observada implica nova classe de falha.\n");

fprintf(f, ...
    "G4/E03 autorizado metodologicamente, sujeito a decisao do pesquisador.\n");

fclose(f);

%% ============================================================
%% CONSOLIDACAO
%% ============================================================

fprintf("\n========================================\n");
fprintf(" RESULTADOS CONSOLIDADOS\n");
fprintf("========================================\n");

fprintf("Registros processados: %d\n",numel(DEV));

fprintf("Ciclos treino: %d\n",numel(IT));

fprintf("Ciclos validacao: %d\n",numel(IV));

fprintf("Configuracoes investigadas: %d\n",rows(SUM));

fprintf("Repeticoes no treinamento: %d\n",rows(ALL));

fprintf("Ciclos piloto no treinamento: %d\n", ...
    rows(ALL)*NC_PILOTO);

fprintf("Repeticoes de validacao: %d\n",rows(VAL));

fprintf("Ciclos piloto na validacao: %d\n", ...
    rows(VAL)*NC_PILOTO);

fprintf("Configuracoes condicionadas aprovadas: %d\n", ...
    numel(elegiveis));

fprintf("\nRESUMO POR CONFIGURACAO\n");

fprintf(["Metodo                 Bloco    " ...
         "Aprovacao       Score       Status\n"]);

for j=1:rows(SUM)

  fprintf("%-22s %5d    %8.2f %%    %.6f    %d\n", ...
      METODOS{SUM(j,1)}, ...
      SUM(j,2), ...
      100*SUM(j,3), ...
      SUM(j,4), ...
      SUM(j,5));

endfor

fprintf("\nStatus definitivo: %s\n",STATUS);

fprintf("Tempo total: %.2f segundos\n",toc(TSTART));

fprintf("\n========================================\n");
fprintf(" G4/E02.4.2 - EXECUCAO CONCLUIDA\n");
fprintf("========================================\n");

fprintf("Centralizacao residual: ANALISADA\n");

fprintf("Dependencia trifasica: ANALISADA\n");

fprintf("Dependencia temporal: ANALISADA\n");

fprintf("Comprimento dos blocos: INVESTIGADO\n");

fprintf("Bootstrap convencional: INVESTIGADO\n");

fprintf("Bootstrap condicionado: INVESTIGADO\n");

fprintf("Juncoes internas: AUDITADAS\n");

fprintf("ACF e PSD: AUDITADAS\n");

fprintf("Caracteristicas fisicas: AUDITADAS\n");

fprintf("Selecao por treinamento: EXECUTADA\n");

fprintf("Decisao para expansao: PRODUZIDA\n");

fprintf("Variantes definitivas: ZERO\n");

fprintf("Treinamento de IA: NENHUM\n");

fprintf("Registros 06 e 12: PRESERVADOS\n");

fprintf("Arquivos originais: PRESERVADOS\n");

fprintf("\nDiretorio de saida:\n%s\n",OUT);

fprintf("\nProxima etapa: G4/E03 - EXPANSAO DEFINITIVA\n");

fprintf("Aguardando aprovacao do pesquisador.\n");

fprintf("========================================\n");

diary off;
