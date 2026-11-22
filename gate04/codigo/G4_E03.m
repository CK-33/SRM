
1;

%% ============================================================
%% GATE 4 - EXPERIMENTO 03
%% EXPANSAO DEFINITIVA DOS DADOS SRM
%%
%% Arquivo: G4_E03.m
%% Ambiente: GNU Octave 11.3
%%
%% ============================================================
%% OBJETIVOS
%%
%% 1. Gerar 480 variantes de 10 registros de desenvolvimento.
%% 2. Preservar os 10 originais em representacao canonica.
%% 3. Introduzir diversidade estatistica controlada.
%% 4. Preservar a rastreabilidade de todas as variantes.
%% 5. Utilizar Latin Hypercube nas perturbacoes combinadas.
%% 6. Preservar a estrutura trifasica e o instante do evento.
%% 7. Produzir uma base adequada ao ML tradicional e CNN.
%% 8. Preparar os dados para a auditoria independente E04.
%%
%% ============================================================
%% DIMENSIONAMENTO
%%
%% Originais canonicos:                 10
%%
%% Familia A - AWGN:                   120
%% Familia B - Ganhos instrumentais:   120
%% Familia C - Transformacao temporal:  80
%% Familia D - Perturbacoes combinadas: 160
%%
%% TOTAL:                              490
%%
%% Treinamento:                        392
%% Validacao:                           98
%%
%% Registros 06 e 12:        NAO ACESSADOS
%%
%% ============================================================
%% RESTRICOES CIENTIFICAS
%%
%% - Nenhuma alteracao dos arquivos proprietarios.
%% - Nenhuma comunicacao externa.
%% - Nenhum treinamento de IA.
%% - Nenhum bootstrap.
%% - Nenhuma colagem de segmentos.
%% - Nenhuma extrapolacao temporal.
%% - Nenhuma alteracao da classe fisica da falha.
%% - Nenhuma selecao baseada no desempenho de classificadores.
%% - Variantes herdam o grupo do registro original.
%% - Variantes nao constituem ensaios fisicos independentes.
%% - Validacao nao participa da definicao dos parametros.
%%
%% ============================================================

clear;
clc;
close all;

pkg load signal;

graphics_toolkit("gnuplot");
set(0,"defaultfigurevisible","off");

%% ============================================================
%% SECAO 01 - FUNCOES ESTATISTICAS
%% ============================================================

function y = rms_local(x)

  y = sqrt(mean(double(x).^2,1));

endfunction


function q = percentil(x,p)

  x = double(x(:));

  x = sort(x(isfinite(x)));

  if isempty(x)

    q = NaN;

    return;

  endif

  k = 1+(numel(x)-1)*p/100;

  a = floor(k);

  b = ceil(k);

  q = x(a)+(k-a)*(x(b)-x(a));

endfunction


function y = erro_relativo(x,ref)

  y = 100*abs(x-ref)./max(abs(ref),1e-12);

endfunction


function C = correlacao_segura(X)

  C = corrcoef(double(X));

  if any(!isfinite(C(:)))

    C = NaN(3,3);

  endif

endfunction


function y = desequilibrio(R)

  y = 100*std(R,0,2)/max(mean(R),1e-12);

endfunction


function y = rms_trifasico(R)

  y = sqrt(mean(R.^2));

endfunction


function t = grade_temporal(t0,N,FS)

  t = t0+(0:N-1)'/FS;

endfunction


%% ============================================================
%% SECAO 02 - REPRODUTIBILIDADE E LATIN HYPERCUBE
%% ============================================================

function s = semente_registro(r,familia,indice)

  s = 4200000+10000*r+1000*familia+indice;

endfunction


function L = lhs_local(N,D,seed)

  rng(seed,"twister");

  L = zeros(N,D);

  for j=1:D

    estratos = randperm(N)';

    L(:,j) = (estratos-1+rand(N,1))/N;

  endfor

  assert(all(L(:)>=0));

  assert(all(L(:)<1));

  %% Cada dimensao deve visitar todos os estratos.

  for j=1:D

    assert(numel(unique(floor(N*L(:,j))))==N);

  endfor

endfunction


function y = mapear(u,a,b)

  y = a+(b-a).*u;

endfunction


%% ============================================================
%% SECAO 03 - CARREGAMENTO E REGULARIZACAO TEMPORAL
%%
%% Convencao validada nos experimentos anteriores:
%% selecionar a ultima ocorrencia de cada instante nominal.
%%
%% Nenhuma interpolacao dos dados originais.
%% ============================================================

function [X25,t25,info] = carregar_original( ...
    arquivo,te,FS200,M)

  S = load(arquivo,"results");

  assert(isfield(S,"results"));

  t = double(S.results.time(:));

  X = double(S.results.signals(3).values);

  E = double(S.results.signals(6).values(:));

  clear S;

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

  TS = 1/FS200;

  TOL = 0.001*TS;

  Nraw = numel(t);

  Nreg = round((t(end)-t(1))/TS)+1;

  tg = grade_temporal(t(1),Nreg,FS200);

  k = round((t-t(1))/TS)+1;

  v = find(k>=1 & k<=Nreg);

  v = v(abs(t(v)-tg(k(v)))<=TOL);

  contagem = accumarray( ...
      k(v),ones(numel(v),1), ...
      [Nreg 1],@sum,0);

  assert(all(contagem>0), ...
      "Grade temporal incompleta.");

  IDX = accumarray( ...
      k(v),v,[Nreg 1],@max,0);

  assert(all(IDX>0));

  erro_grade = max(abs(t(IDX)-tg));

  Xreg = X(IDX,:);

  Ereg = E(IDX);

  assert(Nreg==110000);

  %% Marcador fisico original.

  eventos = find(diff(Ereg)!=0)+1;

  assert(numel(eventos)==1, ...
      "Numero inesperado de transicoes do marcador.");

  te_marcador = tg(eventos);

  assert(abs(te_marcador-te)<=TS+1e-10);

  %% Identificacao da grade ZOH.

  H = zeros(M,3);

  for p=1:3

    ii = find(diff(Xreg(:,p))!=0)+1;

    H(:,p) = accumarray( ...
        mod(ii-1,M)+1, ...
        ones(numel(ii),1), ...
        [M 1],@sum,0);

  endfor

  [~,origem] = max(sum(H,2));

  assert(origem==1, ...
      "Origem inesperada da grade ZOH.");

  assert(rem(Nreg,M)==0);

  X25 = Xreg(1:M:end,:);

  t25 = tg(1:M:end);

  Xrec = repelem(X25,M,1);

  erro_zoh = max(abs(Xrec(:)-Xreg(:)));

  assert(erro_zoh<=1e-10, ...
      "Reconstrucao ZOH nao exata.");

  assert(rows(X25)==13750);

  ultimo_pre = find(t25<te-1e-10,1,"last");

  primeiro_pos = find(t25>=te-1e-10,1,"first");

  assert(!isempty(ultimo_pre));

  assert(!isempty(primeiro_pos));

  offset_evento = ...
      (te-t25(ultimo_pre))*1e6;

  %% Offset dentro do bloco ZOH associado ao evento.

  offset_grade = mod( ...
      round((te-t25(1))*FS200),M);

  info = [ ...
      Nraw, ...
      Nreg, ...
      rows(X25), ...
      Nraw-numel(v), ...
      sum(max(contagem-1,0)), ...
      erro_grade, ...
      origem, ...
      erro_zoh, ...
      te_marcador, ...
      offset_grade/FS200*1e6];

endfunction


%% ============================================================
%% SECAO 04 - MASCARAS TEMPORAIS
%%
%% Regioes:
%% 1 = pre-evento
%% 2 = transitorio: primeiros 200 ms
%% 3 = tardio: a partir de 200 ms
%%
%% Na grade de 25 kHz, utilizam-se blocos completos.
%% Blocos que atravessam fronteiras sao excluidos da
%% caracterizacao por regime.
%%
%% O instante da falha continua armazenado exatamente.
%% ============================================================

function M = mascaras_regimes( ...
    t,te,FS,TRANSITORIO,FS200)

  TS = 1/FS;

  if FS<FS200

    tfim = t+TS-1/FS200;

  else

    tfim = t;

  endif

  tol = 1e-10;

  M = false(numel(t),4);

  M(:,1) = tfim<te-tol;

  M(:,2) = ...
      t>=te-tol & ...
      tfim<te+TRANSITORIO-tol;

  M(:,3) = t>=te+TRANSITORIO-tol;

  M(:,4) = true(numel(t),1);

  assert(sum(M(:,1))>100);

  assert(sum(M(:,2))>100);

  assert(sum(M(:,3))>100);

  assert(all(sum(M(:,1:3),2)<=1));

endfunction


%% ============================================================
%% SECAO 05 - AWGN
%%
%% O RMS pre-evento calibra o desvio-padrao do ruido.
%% O mesmo desvio-padrao permanece ao longo do registro.
%%
%% A realizacao nao e renormalizada para forcar SNR exato.
%% A variacao amostral do SNR e medida e registrada.
%% ============================================================

function [Y,N,meta] = aplicar_awgn( ...
    X,t,te,FS,FS200,SNR,seed)

  M = mascaras_regimes(t,te,FS,0.2,FS200);

  R = rms_local(X(M(:,1),:));

  sigma = R*10^(-SNR/20);

  rng(seed,"twister");

  N = bsxfun(@times,randn(size(X)),sigma);

  Y = X+N;

  Rnoise = rms_local(N(M(:,1),:));

  SNRreal = 20*log10( ...
      max(R,1e-15)./max(Rnoise,1e-15));

  mu = mean(N,1);

  sd = std(N,0,1);

  meta = [sigma,SNRreal,mu,sd];

  assert(all(isfinite(Y(:))));

  assert(all(sigma>0));

endfunction


%% ============================================================
%% SECAO 06 - GANHOS INSTRUMENTAIS
%% ============================================================

function Y = aplicar_ganho(X,G)

  assert(numel(G)==3);

  assert(all(G>0));

  Y = bsxfun(@times,X,G);

endfunction


%% ============================================================
%% SECAO 07 - MODULACAO SUAVE
%%
%% Modula continuamente a amplitude dos sinais.
%% Nao depende do rotulo pre/post-evento.
%%
%% m(t) = 1 + a*sin(2*pi*f*(t-tref)+phi)
%%
%% A frequencia de modulacao nao representa uma nova
%% frequencia fisica validada do motor.
%% ============================================================

function [Y,ENV] = aplicar_modulacao( ...
    X,t,A,F,PHI,TREF)

  ENV = ones(size(X));

  for p=1:3

    ENV(:,p) = 1+A(p)*sin( ...
        2*pi*F(p)*(t-TREF)+PHI(p));

  endfor

  assert(all(ENV(:)>0));

  Y = X.*ENV;

endfunction


%% ============================================================
%% SECAO 08 - FILTRAGEM ANTIALIAS
%%
%% Para compressao temporal, alpha>1:
%% frequencias sao deslocadas para cima.
%%
%% Emprega-se FIR simetrico, com extensao por reflexao
%% dentro de cada regime pre/post-evento.
%%
%% Nao se permite filtragem cruzando a fronteira do evento.
%% A regiao afetada pela borda do filtro e registrada.
%%
%% O filtro nao representa uma medicao instrumental real.
%% ============================================================

function Y = filtrar_reflexao(X,b)

  L = (numel(b)-1)/2;

  assert(L==floor(L));

  assert(rows(X)>L+1);

  esquerda = flipud(X(2:L+1,:));

  direita = flipud(X(end-L:end-1,:));

  Z = [esquerda;X;direita];

  Y = zeros(size(X));

  for p=1:columns(X)

    aux = conv(Z(:,p),b,"valid");

    assert(numel(aux)==rows(X));

    Y(:,p) = aux;

  endfor

endfunction


%% ============================================================
%% SECAO 09 - TRANSFORMACAO TEMPORAL COERENTE
%%
%% tau = te + alpha*(t-te)
%%
%% alpha > 1: compressao
%% alpha < 1: dilatacao
%%
%% A mesma transformacao e utilizada nas tres fases.
%% O evento permanece fixo no tempo.
%%
%% - Nao ha extrapolacao externa.
%% - Nao ha preenchimento com zeros.
%% - Nao ha permutacao de amostras.
%% - Nao ha interpolacao atraves do evento.
%% - Bordas sem suporte sao descartadas.
%%
%% Em intervalos inferiores a uma amostra junto ao evento,
%% utiliza-se o extremo do mesmo lado, sem misturar regimes.
%%
%% O E04 podera identificar essas amostras nos metadados.
%% ============================================================

function [Y,tnew,meta] = aplicar_temporal( ...
    X,t,te,FS,alpha)

  assert(alpha>=0.95 && alpha<=1.05);

  if abs(alpha-1)<1e-15

    Y = X;

    tnew = t;

    meta = [0,0,0,0,0,1,0];

    return;

  endif

  pre = t<te-1e-10;

  pos = !pre;

  Z = X;

  ordem = 0;

  fc = 1;

  guard = 0;

  if alpha>1

    ordem = 160;

    fc = 0.97/alpha;

    b = fir1(ordem,fc,"low", ...
        hamming(ordem+1));

    Z(pre,:) = filtrar_reflexao(X(pre,:),b);

    Z(pos,:) = filtrar_reflexao(X(pos,:),b);

    guard = (ordem/2+2)/FS;

  endif

  tau = te+alpha*(t-te);

  ok = tau>=t(1)-1e-10 & ...
       tau<=t(end)+1e-10;

  idx = find(ok);

  assert(!isempty(idx));

  assert(all(diff(idx)==1));

  tnew = t(idx);

  tq = tau(idx);

  Y = zeros(numel(idx),3);

  pre_new = tnew<te-1e-10;

  pos_new = !pre_new;

  nfix = 0;

  for lado=1:2

    if lado==1

      src = pre;

      dst = pre_new;

    else

      src = pos;

      dst = pos_new;

    endif

    ts = t(src);

    xs = Z(src,:);

    q = tq(dst);

    %% Corrige apenas a borda subamostral junto ao evento.
    %% Nenhuma consulta pode sair do suporte global.

    qc = min(max(q,ts(1)),ts(end));

    diferenca = abs(qc-q);

    assert(all(diferenca<=1/FS+1e-9));

    nfix += sum(diferenca>1e-10);

    Y(dst,:) = interp1(ts,xs,qc,"linear");

  endfor

  removidas_inicio = idx(1)-1;

  removidas_fim = numel(t)-idx(end);

  assert(all(isfinite(Y(:))));

  assert(all(isfinite(tnew)));

  assert(abs(median(diff(tnew))-1/FS)<1e-10);

  assert(tnew(1)<te);

  assert(tnew(end)>te);

  %% Metadados:
  %% 1 = truncamento inicial em amostras
  %% 2 = truncamento final em amostras
  %% 3 = correcoes subamostrais junto ao evento
  %% 4 = ordem FIR
  %% 5 = guarda temporal do filtro em segundos
  %% 6 = frequencia de corte normalizada
  %% 7 = maximo deslocamento temporal em segundos

  meta = [ ...
      removidas_inicio, ...
      removidas_fim, ...
      nfix, ...
      ordem, ...
      guard, ...
      fc, ...
      max(abs(tq-tnew))];

endfunction


%% ============================================================
%% SECAO 10 - CONFIGURACAO DE UMA VARIANTE
%% ============================================================

function P = parametros_default()

  P.familia = 0;

  P.subfamilia = 0;

  P.indice = 0;

  P.metodo = "original_canonico";

  P.seed = 0;

  P.seed_ruido = 0;

  P.snr = NaN;

  P.ruido_modo = 0;

  %% ruido_modo:
  %% 0 = sem ruido
  %% 1 = ruido na grade canonica de 25 kHz
  %% 2 = ruido pos-ZOH na grade de 200 kHz

  P.ganho_modo = 0;

  %% ganho_modo:
  %% 0 = sem ganho
  %% 1 = ganho comum
  %% 2 = ganho independente

  P.ganho_limite = 0;

  P.ganho = [1 1 1];

  P.alpha = 1;

  P.mod_amp = [0 0 0];

  P.mod_freq = [0 0 0];

  P.mod_fase = [0 0 0];

  P.lhs_seed = 0;

  P.lhs_linha = 0;

  P.lhs = NaN(1,15);

endfunction


%% ============================================================
%% SECAO 11 - PLANEJAMENTO DAS 49 REPRESENTACOES
%% ============================================================

function PLAN = planejar_registro(r)

  P0 = parametros_default();

  PLAN = repmat(P0,49,1);

  k = 1;

  %% ----------------------------------------------------------
  %% ORIGINAL CANONICO
  %% ----------------------------------------------------------

  PLAN(k) = P0;

  %% ----------------------------------------------------------
  %% FAMILIA A - 12 VARIANTES
  %%
  %% 3 SNR x 2 modalidades x 2 realizacoes.
  %% ----------------------------------------------------------

  niveis = [50 40 30];

  idx = 0;

  for modo=1:2

    for s=1:numel(niveis)

      for rep=1:2

        idx++;

        k++;

        P = parametros_default();

        P.familia = 1;

        P.subfamilia = modo;

        P.indice = idx;

        P.seed = semente_registro(r,1,idx);

        P.seed_ruido = P.seed+8000000;

        P.snr = niveis(s);

        P.ruido_modo = modo;

        if modo==1

          P.metodo = "AWGN_25k";

        else

          P.metodo = "AWGN_posZOH_200k";

        endif

        PLAN(k) = P;

      endfor

    endfor

  endfor

  %% ----------------------------------------------------------
  %% FAMILIA B - 12 VARIANTES
  %%
  %% Ganhos de +/-1%, +/-5% e +/-10%.
  %%
  %% Duas realizacoes antiteticas por intensidade.
  %% Isso evita que ambas as variantes fiquem concentradas
  %% acidentalmente no mesmo lado do intervalo de ganho.
  %% ----------------------------------------------------------

  niveis = [0.01 0.05 0.10];

  idx = 0;

  for modo=1:2

    for s=1:numel(niveis)

      limite = niveis(s);

      seedbase = semente_registro( ...
          r,2,500+10*modo+s);

      rng(seedbase,"twister");

      if modo==1

        d = limite*(0.5+0.5*rand());

        d = [d d d];

      else

        d = limite*(0.5+0.5*rand(1,3));

        sinal = 2*(rand(1,3)>=0.5)-1;

        d = d.*sinal;

        %% Evita confundir a perturbacao diferencial
        %% com um ganho aproximadamente comum.

        if all(sinal==sinal(1))

          d(3) = -d(3);

        endif

      endif

      for rep=1:2

        idx++;

        k++;

        P = parametros_default();

        P.familia = 2;

        P.subfamilia = modo;

        P.indice = idx;

        P.seed = seedbase;

        P.ganho_modo = modo;

        P.ganho_limite = limite;

        P.ganho = 1+(-1)^(rep-1)*d;

        if modo==1

          P.metodo = "ganho_comum";

        else

          P.metodo = "ganho_independente";

        endif

        PLAN(k) = P;

      endfor

    endfor

  endfor

  %% ----------------------------------------------------------
  %% FAMILIA C - 8 VARIANTES
  %%
  %% Transformacoes deterministicas e simetricas.
  %% ----------------------------------------------------------

  niveis = [0.0025 0.005 0.01 0.02];

  idx = 0;

  for s=1:numel(niveis)

    for sentido=[-1 1]

      idx++;

      k++;

      P = parametros_default();

      P.familia = 3;

      P.indice = idx;

      P.seed = semente_registro(r,3,idx);

      P.alpha = 1+sentido*niveis(s);

      if sentido<0

        P.subfamilia = 1;

        P.metodo = "dilatacao_temporal";

      else

        P.subfamilia = 2;

        P.metodo = "compressao_temporal";

      endif

      PLAN(k) = P;

    endfor

  endfor

  %% ----------------------------------------------------------
  %% FAMILIA D - 16 VARIANTES
  %%
  %% D1: ganho independente + AWGN
  %% D2: modulacao suave
  %% D3: transformacao temporal + AWGN
  %% D4: ganho + modulacao + AWGN
  %%
  %% Latin Hypercube independente por subfamilia.
  %% Quatro pontos por subfamilia e registro.
  %%
  %% O LHS melhora a cobertura marginal, mas nao garante
  %% cobertura completa de um espaco multidimensional.
  %% ----------------------------------------------------------

  idx = 0;

  for sub=1:4

    seedlhs = semente_registro(r,4,500+sub);

    L = lhs_local(4,15,seedlhs);

    for j=1:4

      idx++;

      k++;

      P = parametros_default();

      P.familia = 4;

      P.subfamilia = sub;

      P.indice = idx;

      P.seed = semente_registro(r,4,idx);

      P.seed_ruido = P.seed+8000000;

      P.lhs_seed = seedlhs;

      P.lhs_linha = j;

      P.lhs = L(j,:);

      u = L(j,:);

      if sub==1

        %% D1 - Ganho + AWGN.

        P.metodo = "ganho_AWGN";

        P.ganho_modo = 2;

        P.ganho_limite = 0.05;

        P.ganho = mapear(u(1:3),0.95,1.05);

        P.snr = mapear(u(4),30,50);

        P.ruido_modo = 1;

      elseif sub==2

        %% D2 - Modulacao suave trifasica.
        %%
        %% Amplitude: 1 a 5%.
        %% Frequencia: 0.5 a 5 Hz.

        P.metodo = "modulacao_suave";

        P.mod_amp = mapear(u(1:3),0.01,0.05);

        P.mod_freq = mapear(u(4:6),0.5,5);

        P.mod_fase = mapear(u(7:9),0,2*pi);

        P.seed_ruido = 0;

      elseif sub==3

        %% D3 - Transformacao temporal + AWGN.
        %%
        %% Os quatro estratos garantem cobertura de
        %% compressao e dilatacao em alpha.

        P.metodo = "temporal_AWGN";

        P.alpha = mapear(u(1),0.98,1.02);

        P.snr = mapear(u(2),30,50);

        P.ruido_modo = 1;

      elseif sub==4

        %% D4 - Ganho + modulacao + AWGN.
        %%
        %% Limites mais conservadores que os das
        %% perturbacoes isoladas.

        P.metodo = "ganho_modulacao_AWGN";

        P.ganho_modo = 2;

        P.ganho_limite = 0.03;

        P.ganho = mapear(u(1:3),0.97,1.03);

        P.mod_amp = mapear(u(4:6),0.01,0.03);

        P.mod_freq = mapear(u(7:9),0.5,5);

        P.mod_fase = mapear(u(10:12),0,2*pi);

        P.snr = mapear(u(13),35,50);

        P.ruido_modo = 1;

      endif

      PLAN(k) = P;

    endfor

  endfor

  assert(k==49);

  familias = [PLAN.familia];

  assert(sum(familias==0)==1);

  assert(sum(familias==1)==12);

  assert(sum(familias==2)==12);

  assert(sum(familias==3)==8);

  assert(sum(familias==4)==16);

endfunction


%% ============================================================
%% SECAO 12 - EXECUCAO DA TRANSFORMACAO
%%
%% Ordem das operacoes:
%%
%% 1. Transformacao temporal.
%% 2. Ganho instrumental.
%% 3. Modulacao de amplitude.
%% 4. Retencao ZOH, quando necessaria.
%% 5. Adicao de ruido.
%%
%% A referencia limpa para calibracao do AWGN e o sinal
%% imediatamente anterior a adicao de ruido.
%% ============================================================

function [Y,t,FS,meta,BASE,CLEAN] = gerar_variante( ...
    X0,t0,te,P,FS25,FS200,M)

  meta.temporal = [0 0 0 0 0 1 0];

  meta.ruido = NaN(1,12);

  meta.envoltoria_min = [1 1 1];

  meta.envoltoria_max = [1 1 1];

  FS = FS25;

  Y = X0;

  t = t0;

  %% Transformacao temporal.

  if abs(P.alpha-1)>1e-15

    [Y,t,meta.temporal] = aplicar_temporal( ...
        Y,t,te,FS25,P.alpha);

  endif

  %% BASE representa o original no mesmo suporte temporal.
  %% Nao incorpora a deformacao temporal.
  %%
  %% Isso permite medir o efeito total da transformacao.

  k = round((t-t0(1))*FS25)+1;

  assert(all(k>=1 & k<=rows(X0)));

  BASE = X0(k,:);

  %% Ganho.

  Y = aplicar_ganho(Y,P.ganho);

  %% Modulacao.

  if any(P.mod_amp>0)

    [Y,ENV] = aplicar_modulacao( ...
        Y,t,P.mod_amp,P.mod_freq, ...
        P.mod_fase,t0(1));

    meta.envoltoria_min = min(ENV,[],1);

    meta.envoltoria_max = max(ENV,[],1);

  endif

  %% Retencao ZOH e representacao instrumental.

  if P.ruido_modo==2

    Y = repelem(Y,M,1);

    BASE = repelem(BASE,M,1);

    FS = FS200;

    t = grade_temporal(t(1),rows(Y),FS);

  endif

  CLEAN = Y;

  %% Ruido aditivo.

  if P.ruido_modo>0

    [Y,~,meta.ruido] = aplicar_awgn( ...
        CLEAN,t,te,FS,FS200, ...
        P.snr,P.seed_ruido);

  endif

  assert(rows(Y)==numel(t));

  assert(all(size(Y)==size(BASE)));

  assert(all(size(Y)==size(CLEAN)));

  assert(columns(Y)==3);

  assert(all(isfinite(Y(:))));

  assert(all(isfinite(t)));

  assert(all(diff(t)>0));

  assert(t(1)<te && t(end)>te);

  assert(max(abs(diff(t)-1/FS))<1e-10);

endfunction


%% ============================================================
%% SECAO 13 - CARACTERIZACAO ESPECTRAL
%%
%% PSD para cada fase:
%% - janela Hann periodica de 20 ms;
%% - sobreposicao de 50%;
%% - densidade espectral unilateral;
%% - componente continua preservada.
%%
%% Comparacoes fisicas comuns:
%% 0-5 kHz e 5-12.5 kHz.
%%
%% Acima de 12.5 kHz:
%% somente representacao a 200 kHz.
%%
%% A banda alta nao e interpretada como informacao
%% fisica independente da grade ZOH.
%% ============================================================

function [P,f] = psd_local(X,FS)

  X = double(X);

  L = round(0.020*FS);

  L = min(L,rows(X));

  assert(L>=32);

  H = max(1,floor(L/2));

  ini = 1:H:rows(X)-L+1;

  w = 0.5-0.5*cos(2*pi*(0:L-1)'/L);

  K = floor(L/2)+1;

  P = zeros(K,columns(X));

  for j=1:numel(ini)

    ii = ini(j):ini(j)+L-1;

    Z = bsxfun(@times,X(ii,:),w);

    F = fft(Z,L,1);

    Q = abs(F(1:K,:)).^2/ ...
        (FS*sum(w.^2));

    if rem(L,2)==0

      Q(2:end-1,:) *= 2;

    else

      Q(2:end,:) *= 2;

    endif

    P += Q;

  endfor

  P /= numel(ini);

  f = (0:K-1)'*FS/L;

endfunction


function B = potencia_bandas(P,f,FS)

  df = f(2)-f(1);

  B = NaN(3,columns(P));

  B(1,:) = sum(P(f<5000,:),1)*df;

  %% Em 25 kHz, inclui explicitamente Nyquist.

  if FS<=25000

    m = f>=5000 & f<=12500;

  else

    m = f>=5000 & f<12500;

  endif

  B(2,:) = sum(P(m,:),1)*df;

  if FS>25000

    B(3,:) = sum(P(f>=12500,:),1)*df;

  endif

endfunction


%% ============================================================
%% SECAO 14 - METRICAS DE ALTERACAO
%%
%% Calculadas para:
%% - pre-evento;
%% - transitorio;
%% - tardio;
%% - registro inteiro.
%%
%% As metricas descrevem a expansao.
%% Nao constituem aprovacao cientifica definitiva.
%% ============================================================

function F = caracteristicas(X,FS)

  X = double(X);

  F.rms = rms_local(X);

  F.media = mean(X,1);

  F.sd = std(X,0,1);

  F.pico = max(abs(X),[],1);

  F.pp = max(X,[],1)-min(X,[],1);

  F.crista = F.pico./max(F.rms,1e-15);

  F.energia = sum(X.^2,1)/FS;

  F.derivada = rms_local(diff(X,1,1)*FS);

  F.negativos = 100*mean(X<0,1);

  F.tri = rms_trifasico(F.rms);

  F.deseq = desequilibrio(F.rms);

  F.corr = correlacao_segura(X);

endfunction


function [MET,ESP,GLOB] = auditar_basico( ...
    Y,BASE,CLEAN,t,te,FS,FS200)

  M = mascaras_regimes(t,te,FS,0.2,FS200);

  MET = zeros(12,26);

  ESP = NaN(12,12);

  GLOB = NaN(4,12);

  k = 0;

  for reg=1:4

    idx = M(:,reg);

    X = Y(idx,:);

    R = BASE(idx,:);

    C = CLEAN(idx,:);

    A = caracteristicas(X,FS);

    B = caracteristicas(R,FS);

    erro = X-R;

    ruido = X-C;

    rmse = rms_local(erro);

    nrmse = 100*rmse./max(B.rms,1e-15);

    snr_total = 20*log10( ...
        max(B.rms,1e-15)./ ...
        max(rmse,1e-15));

    snr_ruido = NaN(1,3);

    rr = rms_local(ruido);

    ref = rms_local(C);

    ok = rr>1e-15;

    snr_ruido(ok) = 20*log10( ...
        ref(ok)./rr(ok));

    %% Auditoria espectral por regime.

    [PY,f] = psd_local(X,FS);

    [PB,~] = psd_local(R,FS);

    BY = potencia_bandas(PY,f,FS);

    BB = potencia_bandas(PB,f,FS);

    %% Distancia de variacao total dos espectros.
    %% Compara formas espectrais normalizadas.

    U = bsxfun(@rdivide,PY, ...
        max(sum(PY,1),1e-15));

    V = bsxfun(@rdivide,PB, ...
        max(sum(PB,1),1e-15));

    dist = 0.5*sum(abs(U-V),1);

    for p=1:3

      k++;

      MET(k,:) = [ ...
          reg, ...
          p, ...
          sum(idx), ...
          B.rms(p), ...
          A.rms(p), ...
          erro_relativo(A.rms(p),B.rms(p)), ...
          B.pico(p), ...
          A.pico(p), ...
          erro_relativo(A.pico(p),B.pico(p)), ...
          B.energia(p), ...
          A.energia(p), ...
          erro_relativo(A.energia(p),B.energia(p)), ...
          A.media(p), ...
          A.sd(p), ...
          A.pp(p), ...
          A.crista(p), ...
          A.derivada(p), ...
          A.negativos(p), ...
          rmse(p), ...
          nrmse(p), ...
          mean(erro(:,p)), ...
          rms_local(ruido(:,p)), ...
          snr_total(p), ...
          snr_ruido(p), ...
          min(X(:,p)), ...
          max(X(:,p))];

      ESP(k,:) = [ ...
          reg, ...
          p, ...
          BB(1,p), ...
          BY(1,p), ...
          BB(2,p), ...
          BY(2,p), ...
          BB(3,p), ...
          BY(3,p), ...
          dist(p), ...
          sum(PY(:,p))*(f(2)-f(1)), ...
          sum(PB(:,p))*(f(2)-f(1)), ...
          FS];

    endfor

    dcorr = abs(A.corr-B.corr);

    GLOB(reg,:) = [ ...
        reg, ...
        B.tri, ...
        A.tri, ...
        erro_relativo(A.tri,B.tri), ...
        B.deseq, ...
        A.deseq, ...
        A.deseq-B.deseq, ...
        A.corr(1,2), ...
        A.corr(1,3), ...
        A.corr(2,3), ...
        max(dcorr(:)), ...
        sum(idx)];

  endfor

endfunction


%% ============================================================
%% SECAO 15 - VERIFICACOES DE INTEGRIDADE
%%
%% Falhas estruturais interrompem a geracao.
%% Alteracoes estatisticas sao registradas, nao rejeitadas.
%%
%% Dessa forma nao selecionamos artificialmente variantes
%% que tornam a classificacao mais facil.
%% ============================================================

function Q = verificar_integridade( ...
    Y,t,FS,FS25,FS200,M,te,BASE,CLEAN,P)

  assert(isreal(Y));

  assert(columns(Y)==3);

  assert(rows(Y)==numel(t));

  assert(all(isfinite(Y(:))));

  assert(!any(isnan(Y(:))));

  assert(!any(isinf(Y(:))));

  assert(FS==FS25 || FS==FS200);

  assert(all(diff(t)>0));

  assert(t(1)<te && t(end)>te);

  assert(all(size(Y)==size(BASE)));

  assert(all(size(Y)==size(CLEAN)));

  max_erro_grade = max(abs(diff(t)-1/FS));

  assert(max_erro_grade<1e-10);

  proporcao_transicoes = ...
      sum(diff(Y,1,1)!=0,1)/max(rows(Y)-1,1);

  if FS==FS25

    erro_reconstrucao = 0;

    zoh_reconstruivel = 1;

  else

    YY = repelem(Y(1:M:end,:),M,1);

    erro_reconstrucao = max(abs(Y(:)-YY(:)));

    zoh_reconstruivel = ...
        erro_reconstrucao<=1e-10;

  endif

  if P.ruido_modo==2

    assert(FS==FS200);

    %% A reconstrucao ZOH exata nao e esperada
    %% depois da adicao de ruido instrumental.

  else

    assert(FS==FS25);

  endif

  npre = sum(t<te-1e-10);

  npos = sum(t>=te-1e-10);

  assert(npre>100);

  assert(npos>100);

  diferenca = Y-BASE;

  if P.familia==0

    assert(max(abs(diferenca(:)))==0);

  else

    assert(any(diferenca(:)!=0), ...
        "Variante identica ao original.");

  endif

  if P.ruido_modo==0

    assert(max(abs(Y(:)-CLEAN(:)))==0);

  endif

  Q = [ ...
      rows(Y), ...
      FS, ...
      t(1), ...
      t(end), ...
      te, ...
      npre, ...
      npos, ...
      max_erro_grade, ...
      zoh_reconstruivel, ...
      erro_reconstrucao, ...
      proporcao_transicoes];

endfunction


%% ============================================================
%% SECAO 16 - PREPARACAO DE ARQUIVOS CSV
%% ============================================================

function iniciar_csv(arquivo,cabecalho)

  fid = fopen(arquivo,"w");

  assert(fid>0);

  fprintf(fid,"%s\n",cabecalho);

  fclose(fid);

endfunction


function acrescentar_csv(arquivo,id,X)

  fid = fopen(arquivo,"a");

  assert(fid>0);

  for j=1:rows(X)

    fprintf(fid,"%s",id);

    fprintf(fid,",%.12g",X(j,:));

    fprintf(fid,"\n");

  endfor

  fclose(fid);

endfunction


function salvar_csv_num(arquivo,cabecalho,X)

  iniciar_csv(arquivo,cabecalho);

  fid = fopen(arquivo,"a");

  assert(fid>0);

  for j=1:rows(X)

    fprintf(fid,"%.12g",X(j,1));

    fprintf(fid,",%.12g",X(j,2:end));

    fprintf(fid,"\n");

  endfor

  fclose(fid);

endfunction


%% ============================================================
%% SECAO 17 - ARMAZENAMENTO DOS SINAIS
%%
%% Cada arquivo MAT contem uma estrutura chamada registro.
%%
%% Campos principais:
%% registro.I_A
%% registro.Fs_Hz
%% registro.t0_s
%% registro.N
%% registro.t_evento_s
%% registro.registro_origem
%% registro.grupo
%% registro.parametros
%%
%% Tempo implícito para economizar espaco:
%% t = t0_s + (0:N-1)'/Fs_Hz
%%
%% Correntes preservadas em double.
%%
%% O instante do evento e metadado, nao entrada de ML.
%% Marcadores e identificadores nao devem ser utilizados
%% como caracteristicas dos classificadores.
%% ============================================================

function salvar_registro( ...
    arquivo,Y,t,FS,te,id,r,grupo,P,meta,Q)

  registro = struct();

  registro.versao_schema = 1;

  registro.id = id;

  registro.registro_origem = r;

  registro.grupo = grupo;

  registro.familia = P.familia;

  registro.subfamilia = P.subfamilia;

  registro.metodo = P.metodo;

  registro.I_A = double(Y);

  registro.fases = {"A","B","C"};

  registro.N = rows(Y);

  registro.Fs_Hz = FS;

  registro.t0_s = t(1);

  registro.tf_s = t(end);

  registro.t_evento_s = te;

  registro.t_transitorio_s = 0.2;

  registro.Fs_original_Hz = 200000;

  registro.Fs_canonica_Hz = 25000;

  registro.fator_ZOH = 8;

  registro.parametros = P;

  registro.transformacao = meta;

  registro.integridade = Q;

  registro.variante_artificial = P.familia>0;

  registro.ensaio_independente = false;

  registro.rotulo = "evento_temporal_pre_pos";

  registro.bootstrap = false;

  registro.referencia_saudavel_certificada = false;

  registro.evento_offset_grade_s = ...
      te-(t(1)+floor((te-t(1))*FS+1e-8)/FS);

  registro.aviso = [ ...
      "Dados derivados. Metadados de evento, grupo, " ...
      "origem e identidade nao sao entradas de ML."];

  %% Gravacao temporaria seguida de renomeacao.
  %% Evita que uma interrupcao deixe um arquivo final
  %% aparentemente completo.

  temporario = [arquivo ".tmp"];

  assert(exist(arquivo,"file")==0, ...
      "Arquivo de destino ja existente.");

  assert(exist(temporario,"file")==0);

  save("-mat7-binary",temporario,"registro");

  D = dir(temporario);

  assert(!isempty(D));

  assert(D.bytes>0);

  [status,msg] = movefile(temporario,arquivo);

  assert(status==1,msg);

endfunction


%% ============================================================
%% SECAO 18 - FIGURAS
%% ============================================================

function exportar(h,OUT,nome)

  set(h,"paperpositionmode","auto");

  print(h,fullfile(OUT,nome),"-dpng","-r150");

  close(h);

  fprintf("Figura exportada: %s\n",nome);

endfunction


function figura_comparacao( ...
    B,Y,t,te,FS,OUT,id)

  h = figure("position",[100 100 1450 950]);

  fases = {"A","B","C"};

  %% Visualizacao em janela de 60 ms ao redor do evento.

  m = t>=te-0.020 & t<=te+0.040;

  ii = find(m);

  salto = max(1,floor(numel(ii)/5000));

  ii = ii(1:salto:end);

  x = 1000*(t(ii)-te);

  for p=1:3

    subplot(3,1,p);

    plot(x,B(ii,p), ...
        "color",[0.35 0.35 0.35], ...
        "linewidth",1.0);

    hold on;

    plot(x,Y(ii,p), ...
        "color",[0.12 0.48 0.78], ...
        "linewidth",1.2);

    yl = ylim();

    plot([0 0],yl,"r--","linewidth",1);

    ylabel("Corrente (A)");

    title(sprintf("%s - Fase %s",id,fases{p}));

    grid on;

    if p==1

      legend({"Original","Variante","Evento"}, ...
          "location","northoutside", ...
          "orientation","horizontal");

    endif

    if p==3

      xlabel("Tempo relativo ao evento (ms)");

    endif

  endfor

  exportar(h,OUT, ...
      sprintf("Comparacao_%s.png",id));

endfunction


%% ============================================================
%% SECAO 19 - CONFIGURACOES PRINCIPAIS
%% ============================================================

ROOT = [ ...
    "PASTA_DO_USUARIO/Documents/" ...
    "Pesquisa_SRM/TesteNo1_31OCT2026/" ...
    "Data and Scripts"];

RAW = fullfile(ROOT,"Results_1_to_12");

TREINO = [1 2 4 5 7 8 10 11];

VALIDACAO = [3 9];

RESERVADOS = [6 12];

DEV = sort([TREINO VALIDACAO]);

assert(numel(DEV)==10);

assert(isempty(intersect(TREINO,VALIDACAO)));

assert(isempty(intersect(DEV,RESERVADOS)));

EVENTOS = [ ...
    1.0017, ...
    1.0019, ...
    1.0021, ...
    1.0024, ...
    1.0026, ...
    1.0030, ...
    1.0040, ...
    1.0050, ...
    1.0060, ...
    1.0070, ...
    1.0075, ...
    1.0080];

FS200 = 200000;

FS25 = 25000;

MZOH = FS200/FS25;

assert(MZOH==8);

TRANSITORIO = 0.2;

N_ESPERADO = 490;

N_TREINO = 392;

N_VALIDACAO = 98;

%% Exemplos graficos somente do R01.
%% Nao gerar centenas de figuras.

REGISTRO_FIGURAS = 1;

GERAR_FIGURAS = true;

%% Verifica leitura dos arquivos gerados, um por familia,
%% alem de todas as dimensoes registradas no manifesto.

VERIFICAR_RELEITURA = true;

PASTAS = { ...
    "Originais_Canonicos", ...
    "Familia_A_AWGN", ...
    "Familia_B_Ganho", ...
    "Familia_C_Temporal", ...
    "Familia_D_Combinada"};

NOMES_FAMILIAS = { ...
    "Original", ...
    "AWGN", ...
    "Ganho", ...
    "Temporal", ...
    "Combinada"};

REGIOES = { ...
    "pre", ...
    "transitorio", ...
    "tardio", ...
    "global"};

%% ============================================================
%% SECAO 20 - CRIACAO DO DIRETORIO DE SAIDA
%% ============================================================

runid = 1;

while true

  OUT = fullfile(ROOT, ...
      sprintf("Gate4_Exp03_Run%02d",runid));

  if exist(OUT,"dir")==0

    break;

  endif

  runid++;

endwhile

mkdir(OUT);

for j=1:numel(PASTAS)

  mkdir(fullfile(OUT,PASTAS{j}));

endfor

FIG = fullfile(OUT,"Figuras");

mkdir(FIG);

diary(fullfile(OUT,"G4_E03_Console.txt"));

TSTART = tic();

%% O marcador permanece presente se houver interrupcao.

MARCADOR = fullfile(OUT,"EXECUCAO_INCOMPLETA.txt");

f = fopen(MARCADOR,"w");

assert(f>0);

fprintf(f,"Geracao iniciada.\n");

fprintf(f,"Nao utilizar esta base antes da conclusao.\n");

fclose(f);

fprintf("\n========================================\n");
fprintf(" GATE 4 - EXPERIMENTO 03\n");
fprintf(" EXPANSAO DEFINITIVA DOS DADOS\n");
fprintf("========================================\n");

fprintf("Registros de desenvolvimento: %d\n",numel(DEV));

fprintf("Registros de treinamento: %d\n",numel(TREINO));

fprintf("Registros de validacao: %d\n",numel(VALIDACAO));

fprintf("Registros 06 e 12: RESERVADOS\n");

fprintf("Fs original: %.0f Hz\n",FS200);

fprintf("Fs canonica: %.0f Hz\n",FS25);

fprintf("Registros planejados: %d\n",N_ESPERADO);

fprintf("Variantes planejadas: %d\n",N_ESPERADO-10);

fprintf("Bootstrap: EXCLUIDO\n");

fprintf("Latin Hypercube: ATIVADO\n");

fprintf("Diretorio de saida:\n%s\n",OUT);

%% ============================================================
%% SECAO 21 - MANIFESTOS E METADADOS
%% ============================================================

ARQ_MANIFESTO = fullfile(OUT,"G4_E03_Manifesto.csv");

ARQ_PARAM = fullfile(OUT,"G4_E03_Parametros.csv");

ARQ_LHS = fullfile(OUT,"G4_E03_LHS.csv");

ARQ_RUIDO = fullfile(OUT,"G4_E03_Ruido.csv");

ARQ_MET = fullfile(OUT,"G4_E03_Metricas.csv");

ARQ_ESP = fullfile(OUT,"G4_E03_Espectral.csv");

ARQ_GLOB = fullfile(OUT,"G4_E03_Trifasico.csv");

ARQ_INT = fullfile(OUT,"G4_E03_Integridade.csv");

ARQ_TRANS = fullfile(OUT,"G4_E03_Transformacoes.csv");

ARQ_ORIG = fullfile(OUT,"G4_E03_Originais.csv");

iniciar_csv(ARQ_MANIFESTO, ...
    ["id,registro_origem,grupo,familia,subfamilia," ...
     "indice,metodo,arquivo,fs_Hz,n_amostras," ...
     "t_inicio_s,t_fim_s,t_evento_s," ...
     "artificial,seed,seed_ruido,bytes"]);

iniciar_csv(ARQ_PARAM, ...
    ["id,seed,seed_ruido,snr_dB,ruido_modo," ...
     "ganho_modo,ganho_limite," ...
     "ganho_A,ganho_B,ganho_C,alpha," ...
     "mod_amp_A,mod_amp_B,mod_amp_C," ...
     "mod_freq_A_Hz,mod_freq_B_Hz,mod_freq_C_Hz," ...
     "mod_fase_A_rad,mod_fase_B_rad,mod_fase_C_rad," ...
     "lhs_seed,lhs_linha"]);

CAB_LHS = "id,subfamilia,seed,estrato_linha";

for j=1:15

  CAB_LHS = [CAB_LHS,sprintf(",u%02d",j)];

endfor

iniciar_csv(ARQ_LHS,CAB_LHS);

iniciar_csv(ARQ_RUIDO, ...
    ["id,snr_alvo_dB,modo," ...
     "sigma_A,sigma_B,sigma_C," ...
     "snr_real_A_dB,snr_real_B_dB,snr_real_C_dB," ...
     "media_ruido_A,media_ruido_B,media_ruido_C," ...
     "sd_ruido_A,sd_ruido_B,sd_ruido_C"]);

iniciar_csv(ARQ_MET, ...
    ["id,regiao,fase,n_amostras," ...
     "rms_original_A,rms_variante_A,erro_rms_pct," ...
     "pico_original_A,pico_variante_A,erro_pico_pct," ...
     "energia_original_A2s,energia_variante_A2s," ...
     "erro_energia_pct,media_A,desvio_A," ...
     "pico_a_pico_A,fator_crista," ...
     "rms_derivada_A_s,negativos_pct," ...
     "rmse_A,nrmse_pct,media_erro_A," ...
     "rms_ruido_A,snr_alteracao_total_dB," ...
     "snr_ruido_dB,minimo_A,maximo_A"]);

iniciar_csv(ARQ_ESP, ...
    ["id,regiao,fase," ...
     "pot_0_5k_original,pot_0_5k_variante," ...
     "pot_5_12k5_original,pot_5_12k5_variante," ...
     "pot_12k5_100k_original," ...
     "pot_12k5_100k_variante," ...
     "distancia_espectral," ...
     "pot_total_variante,pot_total_original,fs_Hz"]);

iniciar_csv(ARQ_GLOB, ...
    ["id,regiao,rms_tri_original_A," ...
     "rms_tri_variante_A,erro_rms_tri_pct," ...
     "deseq_original_pct,deseq_variante_pct," ...
     "delta_deseq_pp,corr_AB,corr_AC,corr_BC," ...
     "max_delta_corr,n_amostras"]);

iniciar_csv(ARQ_INT, ...
    ["id,n_amostras,fs_Hz,t_inicio_s,t_fim_s," ...
     "t_evento_s,n_pre,n_pos,erro_grade_s," ...
     "zoh_reconstruivel,erro_reconstrucao_A," ...
     "fracao_transicoes_A,fracao_transicoes_B," ...
     "fracao_transicoes_C"]);

iniciar_csv(ARQ_TRANS, ...
    ["id,removidas_inicio,removidas_fim," ...
     "correcoes_borda_evento,ordem_FIR," ...
     "guarda_filtro_s,corte_normalizado," ...
     "deslocamento_temporal_max_s," ...
     "envoltoria_min_A,envoltoria_min_B," ...
     "envoltoria_min_C,envoltoria_max_A," ...
     "envoltoria_max_B,envoltoria_max_C"]);

iniciar_csv(ARQ_ORIG, ...
    ["id,n_bruto,n_regular,n_canonico," ...
     "fora_grade,duplicatas_excedentes," ...
     "erro_grade_s,origem_zoh,erro_zoh_A," ...
     "t_marcador_s,offset_evento_us"]);

%% ============================================================
%% SECAO 22 - REGISTRO DO PROTOCOLO ANTES DA GERACAO
%% ============================================================

f = fopen(fullfile(OUT,"G4_E03_Protocolo.txt"),"w");

assert(f>0);

fprintf(f,"GATE 4 - EXPERIMENTO 03\n");

fprintf(f,"EXPANSAO DEFINITIVA SRM\n\n");

fprintf(f,"Originais_canonicos=10\n");

fprintf(f,"Variantes_A=120\n");

fprintf(f,"Variantes_B=120\n");

fprintf(f,"Variantes_C=80\n");

fprintf(f,"Variantes_D=160\n");

fprintf(f,"Total_planejado=490\n");

fprintf(f,"Fs_canonica=25000\n");

fprintf(f,"Fs_instrumental=200000\n");

fprintf(f,"Treino=1 2 4 5 7 8 10 11\n");

fprintf(f,"Validacao=3 9\n");

fprintf(f,"Reservados=6 12\n");

fprintf(f,"Bootstrap=0\n");

fprintf(f,"Treinamento_IA=0\n");

fprintf(f,"\nFAMILIA A:\n");

fprintf(f,"AWGN 50 40 30 dB.\n");

fprintf(f,"Duas realizacoes por intensidade e modalidade.\n");

fprintf(f,"Calibracao pelo RMS pre-evento da fase.\n");

fprintf(f,"Desvio-padrao constante durante a trajetoria.\n");

fprintf(f,"SNR realizado registrado sem renormalizacao.\n");

fprintf(f,"\nFAMILIA B:\n");

fprintf(f,"Ganhos de magnitude 1 5 10 por cento.\n");

fprintf(f,"Ganhos comuns e independentes por fase.\n");

fprintf(f,"Realizacoes antiteticas.\n");

fprintf(f,"\nFAMILIA C:\n");

fprintf(f,"Alpha=1 +/- [0.0025 0.005 0.01 0.02].\n");

fprintf(f,"Evento fixo como ancora temporal.\n");

fprintf(f,"Interpolacao linear no mesmo lado do evento.\n");

fprintf(f,"FIR antialias para compressao temporal.\n");

fprintf(f,"Sem extrapolacao externa.\n");

fprintf(f,"Bordas sem suporte removidas.\n");

fprintf(f,"\nFAMILIA D:\n");

fprintf(f,"Quatro subfamilias com quatro variantes.\n");

fprintf(f,"Latin Hypercube marginal por subfamilia.\n");

fprintf(f,"D1: ganhos independentes +/-5%% e AWGN 30-50 dB.\n");

fprintf(f,"D2: modulacao 1-5%% de 0.5 a 5 Hz.\n");

fprintf(f,"D3: alpha 0.98-1.02 e AWGN 30-50 dB.\n");

fprintf(f,"D4: ganhos +/-3%%, modulacao 1-3%%, AWGN 35-50 dB.\n");

fprintf(f,"\nLIMITACOES CIENTIFICAS:\n");

fprintf(f,"Variantes nao sao ensaios fisicos independentes.\n");

fprintf(f,"Trajetorias pre-evento compartilham suporte identico.\n");

fprintf(f,"Ganhos representam perturbacoes instrumentais.\n");

fprintf(f,"Transformacoes temporais nao simulam novas velocidades fisicas.\n");

fprintf(f,"Modulacoes nao criam novas classes de falha.\n");

fprintf(f,"AWGN e uma hipotese de ruido instrumental.\n");

fprintf(f,"Nao ha referencia saudavel certificada.\n");

fprintf(f,"Regioes pre/post representam rotulos temporais.\n");

fprintf(f,"Amostras negativas por ruido nao sao truncadas.\n");

fprintf(f,"Energia I2dt expressa em A2s e nao em Joules.\n");

fprintf(f,"Amostras de 200 kHz nao equivalem a atualizacoes independentes.\n");

fprintf(f,"Variantes a 25 e 200 kHz requerem tratamento distinto no Gate 5.\n");

fprintf(f,"SNR total inclui efeitos deterministicos de perturbacao.\n");

fprintf(f,"SNR do ruido refere-se exclusivamente a componente AWGN.\n");

fprintf(f,"NaN em metricas nao aplicaveis e intencional.\n");

fprintf(f,"Nenhuma variante sera selecionada por separabilidade de classes.\n");

fprintf(f,"Aprovacao estatistica definitiva pertence ao E04.\n");

fclose(f);

%% ============================================================
%% SECAO 23 - ESTRUTURAS DE CONSOLIDACAO
%% ============================================================

CONT = zeros(5,2);

CONT_FS = zeros(5,2);

CONT_REG = zeros(10,5);

IDS = {};

ARQUIVOS = {};

TAMANHOS = [];

SUMMARY = [];

RUIDOS = [];

PARAM_SUM = [];

ORIGINAL_SUM = [];

STATS_REGISTROS = [];

LHSCONS = [];

GLOBAL_SUM = [];

ZOH_SUM = [];

FIG_GERADAS = false(5,4);

TOTAL = 0;

TOTAL_BYTES = 0;

%% ============================================================
%% SECAO 24 - GERACAO DEFINITIVA
%% ============================================================

for ir=1:numel(DEV)

  r = DEV(ir);

  assert(!ismember(r,RESERVADOS));

  grupo = 1+ismember(r,VALIDACAO);

  if grupo==1

    grupo_nome = "treino";

  else

    grupo_nome = "validacao";

  endif

  te = EVENTOS(r);

  fprintf("\n========================================\n");

  fprintf(" REGISTRO R%02d | %s\n",r,grupo_nome);

  fprintf("========================================\n");

  TR = tic();

  arquivo_original = fullfile(RAW, ...
      sprintf("results_%d.mat",r));

  assert(exist(arquivo_original,"file")==2, ...
      sprintf("Arquivo ausente: %s",arquivo_original));

  [X0,t0,info] = carregar_original( ...
      arquivo_original,te,FS200,MZOH);

  id_orig = sprintf("R%02d",r);

  acrescentar_csv(ARQ_ORIG,id_orig,info);

  ORIGINAL_SUM = [ORIGINAL_SUM;r,grupo,info];

  fprintf("Amostras brutas: %d\n",info(1));

  fprintf("Amostras regularizadas: %d\n",info(2));

  fprintf("Amostras canonicas: %d\n",info(3));

  fprintf("Erro reconstrucao ZOH: %.12g A\n",info(8));

  fprintf("Evento: %.8f s\n",te);

  fprintf("Offset na grade: %.6f us\n",info(10));

  PLAN = planejar_registro(r);

  for iv=1:numel(PLAN)

    TV = tic();

    P = PLAN(iv);

    fam = P.familia;

    if fam==0

      id = sprintf("R%02d_O00",r);

    else

      letras = "ABCD";

      id = sprintf("R%02d_%c%02d", ...
          r,letras(fam),P.indice);

    endif

    assert(!any(strcmp(IDS,id)));

    %% --------------------------------------------------------
    %% GERACAO
    %% --------------------------------------------------------

    [Y,t,FS,meta,BASE,CLEAN] = gerar_variante( ...
        X0,t0,te,P,FS25,FS200,MZOH);

    Q = verificar_integridade( ...
        Y,t,FS,FS25,FS200,MZOH, ...
        te,BASE,CLEAN,P);

    %% --------------------------------------------------------
    %% CARACTERIZACAO
    %% --------------------------------------------------------

    [MET,ESP,GLOB] = auditar_basico( ...
        Y,BASE,CLEAN,t,te,FS,FS200);

    %% NaN e permitido somente nas metricas nao aplicaveis:
    %% - SNR na ausencia de ruido;
    %% - bandas acima do Nyquist;
    %% - correlacao indefinida, caso ocorra.
    %%
    %% Sinais armazenados nao podem conter NaN.

    assert(all(isfinite(MET(:,1:23)(:))));

    assert(all(isfinite(MET(:,25:26)(:))));

    %% --------------------------------------------------------
    %% ARMAZENAMENTO
    %% --------------------------------------------------------

    relativo = [PASTAS{fam+1} "/" id ".mat"];

    destino = fullfile(OUT,PASTAS{fam+1},[id ".mat"]);

    salvar_registro( ...
        destino,Y,t,FS,te,id,r, ...
        grupo_nome,P,meta,Q);

    D = dir(destino);

    assert(!isempty(D));

    bytes = D.bytes;

    %% --------------------------------------------------------
    %% MANIFESTO
    %% --------------------------------------------------------

    f = fopen(ARQ_MANIFESTO,"a");

    assert(f>0);

    fprintf(f, ...
        ["%s,%d,%s,%d,%d,%d,%s,%s," ...
         "%d,%d,%.12f,%.12f,%.12f,%d,%d,%d,%d\n"], ...
        id,r,grupo_nome,fam,P.subfamilia, ...
        P.indice,P.metodo,relativo,FS,rows(Y), ...
        t(1),t(end),te,fam>0, ...
        P.seed,P.seed_ruido,bytes);

    fclose(f);

    %% --------------------------------------------------------
    %% METADADOS COMPLETOS
    %% --------------------------------------------------------

    PARAM = [ ...
        P.seed, ...
        P.seed_ruido, ...
        P.snr, ...
        P.ruido_modo, ...
        P.ganho_modo, ...
        P.ganho_limite, ...
        P.ganho, ...
        P.alpha, ...
        P.mod_amp, ...
        P.mod_freq, ...
        P.mod_fase, ...
        P.lhs_seed, ...
        P.lhs_linha];

    acrescentar_csv(ARQ_PARAM,id,PARAM);

    acrescentar_csv(ARQ_MET,id,MET);

    acrescentar_csv(ARQ_ESP,id,ESP);

    acrescentar_csv(ARQ_GLOB,id,GLOB);

    acrescentar_csv(ARQ_INT,id,Q);

    acrescentar_csv(ARQ_TRANS,id, ...
        [meta.temporal, ...
         meta.envoltoria_min, ...
         meta.envoltoria_max]);

    if P.ruido_modo>0

      acrescentar_csv(ARQ_RUIDO,id, ...
          [P.snr,P.ruido_modo,meta.ruido]);

      RUIDOS = [RUIDOS; ...
          r,grupo,fam,P.subfamilia, ...
          P.indice,P.snr,P.ruido_modo,meta.ruido];

    endif

    if fam==4

      acrescentar_csv(ARQ_LHS,id, ...
          [P.subfamilia,P.lhs_seed, ...
           P.lhs_linha,P.lhs]);

      LHSCONS = [LHSCONS; ...
          r,grupo,P.subfamilia,P.lhs_linha,P.lhs];

    endif

    %% --------------------------------------------------------
    %% CONSOLIDACAO
    %% --------------------------------------------------------

    TOTAL++;

    TOTAL_BYTES += bytes;

    IDS{end+1} = id;

    ARQUIVOS{end+1} = destino;

    TAMANHOS(end+1,1) = bytes;

    CONT(fam+1,grupo)++;

    CONT_REG(ir,fam+1)++;

    CONT_FS(fam+1,1+(FS==FS200))++;

    mg = MET(MET(:,1)==4,:);

    ep = ESP(ESP(:,1)==4,:);

    gg = GLOB(GLOB(:,1)==4,:);

    SUMMARY = [SUMMARY; ...
        r,grupo,fam,P.subfamilia,P.indice, ...
        FS,rows(Y), ...
        mean(mg(:,6)), ...
        max(mg(:,6)), ...
        mean(mg(:,9)), ...
        max(mg(:,9)), ...
        mean(mg(:,12)), ...
        max(mg(:,20)), ...
        mean(ep(:,9)), ...
        max(ep(:,9)), ...
        gg(4), ...
        gg(7), ...
        bytes];

    PARAM_SUM = [PARAM_SUM; ...
        r,grupo,fam,P.subfamilia,P.indice, ...
        P.snr,P.ganho,P.alpha, ...
        P.mod_amp,P.mod_freq];

    GLOBAL_SUM = [GLOBAL_SUM; ...
        repmat([r,grupo,fam,P.indice],4,1),GLOB];

    STATS_REGISTROS = [STATS_REGISTROS; ...
        repmat([r,grupo,fam,P.indice],12,1),MET];

    ZOH_SUM = [ZOH_SUM; ...
        r,grupo,fam,P.indice,FS,Q(9:13)];

    %% --------------------------------------------------------
    %% PROGRESSO
    %% --------------------------------------------------------

    fprintf(["[%03d/%03d] %-12s | FS=%6d | N=%6d" ...
             " | RMSEn max=%9.4f %% | %.2f s\n"], ...
        TOTAL,N_ESPERADO,id,FS,rows(Y), ...
        max(mg(:,20)),toc(TV));

    fflush(stdout);

    %% --------------------------------------------------------
    %% FIGURAS REPRESENTATIVAS
    %%
    %% Apenas exemplos predefinidos do primeiro registro.
    %% Nao selecionados pelo aparente sucesso da variante.
    %% --------------------------------------------------------

    if GERAR_FIGURAS && r==REGISTRO_FIGURAS

      desenhar = false;

      if fam==1 && ismember(P.indice,[1 5 7 11])

        desenhar = true;

      elseif fam==2 && ismember(P.indice,[5 11])

        desenhar = true;

      elseif fam==3 && ismember(P.indice,[7 8])

        desenhar = true;

      elseif fam==4 && ismember(P.indice,[1 5 9 13])

        desenhar = true;

      endif

      if desenhar

        figura_comparacao(BASE,Y,t,te,FS,FIG,id);

      endif

    endif

    clear Y BASE CLEAN t MET ESP GLOB;

  endfor

  assert(all(CONT_REG(ir,:)==[1 12 12 8 16]));

  fprintf("\nR%02d concluido: 49 registros\n",r);

  fprintf("Tempo do registro: %.2f s\n",toc(TR));

  fprintf("Progresso global: %.2f %%\n", ...
      100*TOTAL/N_ESPERADO);

  fprintf("Armazenamento acumulado: %.2f MB\n", ...
      TOTAL_BYTES/1024^2);

  clear X0 t0 PLAN;

endfor

%% ============================================================
%% SECAO 25 - AUDITORIA ESTRUTURAL DO MANIFESTO
%% ============================================================

fprintf("\n========================================\n");

fprintf(" AUDITORIA ESTRUTURAL DA BASE\n");

fprintf("========================================\n");

assert(TOTAL==N_ESPERADO);

assert(numel(unique(IDS))==N_ESPERADO);

assert(sum(CONT(:,1))==N_TREINO);

assert(sum(CONT(:,2))==N_VALIDACAO);

assert(all(sum(CONT,2)==[10;120;120;80;160]));

assert(all(CONT_REG(:)>0));

assert(size(SUMMARY,1)==N_ESPERADO);

assert(size(STATS_REGISTROS,1)==N_ESPERADO*12);

assert(size(GLOBAL_SUM,1)==N_ESPERADO*4);

assert(isempty(intersect(unique(SUMMARY(:,1)),RESERVADOS)));

for j=1:N_ESPERADO

  D = dir(ARQUIVOS{j});

  assert(!isempty(D));

  assert(D.bytes==TAMANHOS(j));

  assert(D.bytes>0);

endfor

fprintf("Total de arquivos MAT: %d\n",TOTAL);

fprintf("IDs exclusivos: %d\n",numel(unique(IDS)));

fprintf("Treinamento: %d\n",sum(CONT(:,1)));

fprintf("Validacao: %d\n",sum(CONT(:,2)));

fprintf("Registros reservados: PRESERVADOS\n");

fprintf("Rastreabilidade dos arquivos: OK\n");

%% ============================================================
%% SECAO 26 - RELEITURA DOS ARQUIVOS GERADOS
%%
%% Releitura predefinida de representantes das cinco
%% familias e de ambas as representacoes de AWGN.
%% ============================================================

if VERIFICAR_RELEITURA

  fprintf("\n========================================\n");

  fprintf(" RELEITURA DOS ARQUIVOS GERADOS\n");

  fprintf("========================================\n");

  amostra = [];

  for fam=0:4

    j = find(SUMMARY(:,3)==fam,1,"first");

    amostra(end+1) = j;

  endfor

  j = find(SUMMARY(:,3)==1 & ...
           SUMMARY(:,6)==FS200,1,"first");

  amostra(end+1) = j;

  j = find(SUMMARY(:,2)==2,1,"last");

  amostra(end+1) = j;

  amostra = unique(amostra);

  for j=amostra

    A = load(ARQUIVOS{j},"registro");

    R = A.registro;

    assert(strcmp(R.id,IDS{j}));

    assert(R.N==rows(R.I_A));

    assert(R.N==SUMMARY(j,7));

    assert(R.Fs_Hz==SUMMARY(j,6));

    assert(columns(R.I_A)==3);

    assert(all(isfinite(R.I_A(:))));

    assert(!ismember(R.registro_origem,RESERVADOS));

    assert(R.t_evento_s==EVENTOS(R.registro_origem));

    assert(abs(R.tf_s- ...
        (R.t0_s+(R.N-1)/R.Fs_Hz))<1e-9);

    if ismember(R.registro_origem,TREINO)

      assert(strcmp(R.grupo,"treino"));

    else

      assert(strcmp(R.grupo,"validacao"));

    endif

    fprintf("%s | releitura: OK\n",R.id);

    clear A R;

  endfor

endif

%% ============================================================
%% SECAO 27 - RESUMOS ESTATISTICOS
%%
%% Todas as estatisticas de desenvolvimento de parametros
%% permanecem separadas por grupo.
%%
%% Os resultados de validacao sao descritivos.
%% Nenhuma variante e descartada pelo resultado obtido.
%% ============================================================

fprintf("\n========================================\n");

fprintf(" CARACTERIZACAO DA EXPANSAO\n");

fprintf("========================================\n");

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_Contagem.csv"), ...
    ["familia,treino,validacao,total," ...
     "n_25k,n_200k"], ...
    [(0:4)',CONT,sum(CONT,2),CONT_FS]);

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_Contagem_Registros.csv"), ...
    "registro,original,A,B,C,D", ...
    [DEV(:),CONT_REG]);

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_Resumo_Variantes.csv"), ...
    ["registro,grupo,familia,subfamilia,indice," ...
     "fs_Hz,n_amostras," ...
     "erro_rms_medio_pct,erro_rms_max_pct," ...
     "erro_pico_medio_pct,erro_pico_max_pct," ...
     "erro_energia_medio_pct,nrmse_max_pct," ...
     "dist_espectral_media,dist_espectral_max," ...
     "erro_rms_tri_pct,delta_deseq_pp,bytes"], ...
    SUMMARY);

RESUMO = [];

COLS = [8 9 10 11 12 13 14 15 16 17];

for grupo=1:2

  if grupo==1

    gn = "TREINAMENTO";

  else

    gn = "VALIDACAO";

  endif

  fprintf("\n%s\n",gn);

  for fam=0:4

    S = SUMMARY( ...
        SUMMARY(:,2)==grupo & ...
        SUMMARY(:,3)==fam,:);

    fprintf("\nFamilia %s | N=%d\n", ...
        NOMES_FAMILIAS{fam+1},rows(S));

    fprintf("Erro RMS medio: %.6f %%\n",mean(S(:,8)));

    fprintf("Erro RMS P95: %.6f %%\n", ...
        percentil(S(:,9),95));

    fprintf("Erro pico medio: %.6f %%\n",mean(S(:,10)));

    fprintf("Erro energia medio: %.6f %%\n",mean(S(:,12)));

    fprintf("NRMSE maximo mediano: %.6f %%\n", ...
        median(S(:,13)));

    fprintf("Distancia espectral media: %.8f\n", ...
        mean(S(:,14)));

    fprintf("Erro RMS trifasico medio: %.6f %%\n", ...
        mean(S(:,16)));

    for j=1:numel(COLS)

      x = S(:,COLS(j));

      RESUMO = [RESUMO; ...
          grupo,fam,COLS(j),numel(x), ...
          mean(x),std(x), ...
          min(x), ...
          percentil(x,5), ...
          median(x), ...
          percentil(x,95), ...
          max(x)];

    endfor

  endfor

endfor

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_Resumo_Estatistico.csv"), ...
    ["grupo,familia,coluna_metrica,n," ...
     "media,dp,min,p05,mediana,p95,max"], ...
    RESUMO);

%% ============================================================
%% SECAO 28 - AUDITORIA AWGN
%% ============================================================

fprintf("\n========================================\n");

fprintf(" AUDITORIA DO RUIDO\n");

fprintf("========================================\n");

%% Estrutura RUIDOS:
%% 1 registro
%% 2 grupo
%% 3 familia
%% 4 subfamilia
%% 5 indice
%% 6 SNR alvo
%% 7 modo
%% 8:10 sigma
%% 11:13 SNR real
%% 14:16 media
%% 17:19 desvio-padrao

assert(rows(RUIDOS)==240);

ERRO_SNR = bsxfun(@minus, ...
    RUIDOS(:,11:13),RUIDOS(:,6));

fprintf("Variantes contendo AWGN: %d\n",rows(RUIDOS));

fprintf("Realizacoes ruido na grade 25 kHz: %d\n", ...
    sum(RUIDOS(:,7)==1));

fprintf("Realizacoes ruido na grade 200 kHz: %d\n", ...
    sum(RUIDOS(:,7)==2));

fprintf("Erro medio absoluto SNR: %.6f dB\n", ...
    mean(abs(ERRO_SNR(:))));

fprintf("Erro maximo absoluto SNR: %.6f dB\n", ...
    max(abs(ERRO_SNR(:))));

fprintf("P95 do erro absoluto SNR: %.6f dB\n", ...
    percentil(abs(ERRO_SNR(:)),95));

for p=1:3

  fprintf("Fase %d | erro medio SNR: %.6f dB\n", ...
      p,mean(ERRO_SNR(:,p)));

endfor

%% ============================================================
%% SECAO 29 - AUDITORIA LATIN HYPERCUBE
%%
%% Verifica cobertura marginal efetivamente produzida.
%% Nao utiliza metricas de classificacao.
%% ============================================================

fprintf("\n========================================\n");

fprintf(" AUDITORIA LATIN HYPERCUBE\n");

fprintf("========================================\n");

assert(rows(LHSCONS)==160);

LHS_AUDIT = [];

for r=DEV

  for sub=1:4

    m = LHSCONS(:,1)==r & LHSCONS(:,3)==sub;

    U = LHSCONS(m,5:end);

    assert(rows(U)==4);

    ocupacao = zeros(1,columns(U));

    for j=1:columns(U)

      ocupacao(j) = numel(unique(floor(4*U(:,j))));

    endfor

    assert(all(ocupacao==4));

    LHS_AUDIT = [LHS_AUDIT; ...
        r,sub,min(ocupacao), ...
        mean(ocupacao),max(ocupacao)];

  endfor

endfor

fprintf("Planejamentos LHS independentes: %d\n", ...
    rows(LHS_AUDIT));

fprintf("Ocupacao marginal dos estratos: 100 %%\n");

fprintf("Dimensoes registradas por planejamento: 15\n");

fprintf("Cobertura conjunta exaustiva: NAO ALEGADA\n");

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_LHS_Auditoria.csv"), ...
    ["registro,subfamilia,ocupacao_min," ...
     "ocupacao_media,ocupacao_max"], ...
    LHS_AUDIT);

%% ============================================================
%% SECAO 30 - DIVERSIDADE PRELIMINAR
%%
%% Avalia a expansao do espaco de caracteristicas
%% eletricas de baixa dimensao.
%%
%% Conjunto de 11 caracteristicas:
%% RMS das tres fases;
%% pico das tres fases;
%% energia I2dt das tres fases;
%% RMS trifasico;
%% desequilibrio.
%%
%% Energia convertida para media de I^2 para evitar
%% que o comprimento da trajetoria domine a analise.
%%
%% Nao substitui a auditoria multivariada completa E04.
%% ============================================================

fprintf("\n========================================\n");

fprintf(" DIVERSIDADE ESTATISTICA PRELIMINAR\n");

fprintf("========================================\n");

%% Construir tensor F por registro derivado e regime.

F = NaN(N_ESPERADO,11,3);

for j=1:N_ESPERADO

  offsetM = (j-1)*12;

  offsetG = (j-1)*4;

  for reg=1:3

    ii = offsetM+(reg-1)*3+(1:3);

    iz = offsetG+reg;

    A = STATS_REGISTROS(ii,5:end);

    G = GLOBAL_SUM(iz,5:end);

    rms3 = A(:,5)';

    pico3 = A(:,8)';

    potencia_media = rms3.^2;

    F(j,:,reg) = [ ...
        rms3,pico3,potencia_media,G(3),G(6)];

  endfor

endfor

assert(all(isfinite(F(:))));

%% Padronizacao para a auditoria preliminar.
%%
%% Ajuste somente em originais de treinamento.
%% Uma escala minima evita divisao por variancias
%% artificialmente nulas da referencia compartilhada.
%%
%% Nao utilizar esta escala como scaler definitivo de ML.

DIVERSIDADE = [];

PCMAP = cell(3,1);

for reg=1:3

  X = F(:,:,reg);

  io = SUMMARY(:,2)==1 & SUMMARY(:,3)==0;

  it = SUMMARY(:,2)==1;

  mu = mean(X(io,:),1);

  sd = std(X(io,:),0,1);

  piso = max(0.01*abs(mu),1e-6);

  %% Piso adicional para desequilibrio em p.p.

  piso(11) = max(piso(11),0.10);

  escala = max(sd,piso);

  Z = bsxfun(@rdivide, ...
      bsxfun(@minus,X,mu),escala);

  %% Base PCA ajustada apenas em dados de treinamento,
  %% incluindo variantes de treinamento.
  %% Validacao apenas projetada.

  centro_pca = mean(Z(it,:),1);

  Zc = bsxfun(@minus,Z,centro_pca);

  [~,S,V] = svd(Zc(it,:),"econ");

  lambdas = diag(S).^2/ ...
      max(sum(it)-1,1);

  explicada = 100*lambdas/ ...
      max(sum(lambdas),1e-15);

  SCORE = Zc*V;

  PCMAP{reg} = SCORE(:,1:2);

  fprintf("\nRegiao: %s\n",REGIOES{reg});

  fprintf("PCA PC1: %.4f %%\n",explicada(1));

  fprintf("PCA PC2: %.4f %%\n",explicada(2));

  fprintf("PC1+PC2: %.4f %%\n",sum(explicada(1:2)));

  for grupo=1:2

    for fam=0:4

      m = SUMMARY(:,2)==grupo & ...
          SUMMARY(:,3)==fam;

      XG = Z(m,:);

      CG = cov(XG);

      e = max(real(eig(CG)),0);

      totalvar = sum(e);

      if totalvar>1e-15

        w = e/totalvar;

        w = w(w>1e-15);

        posto_efetivo = exp(-sum(w.*log(w)));

      else

        posto_efetivo = 0;

      endif

      %% Distancia de cada variante ao seu proprio original.
      %% Evita atribuir variabilidade entre registros
      %% exclusivamente a expansao artificial.

      idx = find(m);

      dd = zeros(numel(idx),1);

      for j=1:numel(idx)

        q = idx(j);

        r = SUMMARY(q,1);

        orig = find( ...
            SUMMARY(:,1)==r & ...
            SUMMARY(:,3)==0,1,"first");

        dd(j) = sqrt(mean((Z(q,:)-Z(orig,:)).^2));

      endfor

      DIVERSIDADE = [DIVERSIDADE; ...
          reg,grupo,fam,rows(XG), ...
          totalvar,posto_efetivo, ...
          median(dd), ...
          percentil(dd,5), ...
          percentil(dd,95), ...
          max(dd)];

      fprintf(["Grupo %d | %-12s | N=%3d" ...
               " | variancia=%.6f" ...
               " | posto efetivo=%.4f" ...
               " | distancia mediana=%.6f\n"], ...
          grupo,NOMES_FAMILIAS{fam+1},rows(XG), ...
          totalvar,posto_efetivo,median(dd));

    endfor

  endfor

  salvar_csv_num( ...
      fullfile(OUT,sprintf("G4_E03_PCA_Regiao_%d.csv",reg)), ...
      "registro,grupo,familia,indice,PC1,PC2", ...
      [SUMMARY(:,[1 2 3 5]),SCORE(:,1:2)]);

  salvar_csv_num( ...
      fullfile(OUT,sprintf("G4_E03_PCA_Escala_%d.csv",reg)), ...
      "feature,media_original_treino,escala,centro_PCA", ...
      [(1:11)',mu',escala',centro_pca']);

  salvar_csv_num( ...
      fullfile(OUT,sprintf("G4_E03_PCA_Cargas_%d.csv",reg)), ...
      "feature,PC1,PC2", ...
      [(1:11)',V(:,1:2)]);

endfor

salvar_csv_num( ...
    fullfile(OUT,"G4_E03_Diversidade_Preliminar.csv"), ...
    ["regiao,grupo,familia,n,variancia_total," ...
     "posto_efetivo,distancia_mediana_original," ...
     "distancia_p05,distancia_p95,distancia_max"], ...
    DIVERSIDADE);

%% ============================================================
%% SECAO 31 - FIGURAS CONSOLIDADAS
%% ============================================================

if GERAR_FIGURAS

  fprintf("\n========================================\n");

  fprintf(" FIGURAS CONSOLIDADAS\n");

  fprintf("========================================\n");

  CORES = [ ...
      0.35 0.35 0.35; ...
      0.12 0.48 0.78; ...
      0.16 0.62 0.43; ...
      0.85 0.52 0.15; ...
      0.60 0.35 0.73];

  %% ----------------------------------------------------------
  %% FIGURA 01 - Dimensao da base.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1100 650]);

  bar(CONT,"stacked");

  set(gca,"xtick",1:5, ...
      "xticklabel",NOMES_FAMILIAS);

  ylabel("Numero de registros");

  title("Composicao da base expandida");

  legend({"Treinamento","Validacao"}, ...
      "location","northoutside", ...
      "orientation","horizontal");

  grid on;

  exportar(h,FIG,"G4_E03_01_Composicao.png");

  %% ----------------------------------------------------------
  %% FIGURA 02 - Representacoes temporais.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1100 650]);

  bar(CONT_FS,"stacked");

  set(gca,"xtick",1:5, ...
      "xticklabel",NOMES_FAMILIAS);

  ylabel("Registros");

  title("Distribuicao das frequencias de amostragem");

  legend({"25 kHz","200 kHz"}, ...
      "location","northoutside", ...
      "orientation","horizontal");

  grid on;

  exportar(h,FIG,"G4_E03_02_Frequencias.png");

  %% ----------------------------------------------------------
  %% FIGURA 03 - SNR alvo versus realizado.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1200 850]);

  fases = {"A","B","C"};

  for p=1:3

    subplot(3,1,p);

    m1 = RUIDOS(:,7)==1;

    m2 = RUIDOS(:,7)==2;

    plot(RUIDOS(m1,6),RUIDOS(m1,10+p), ...
        "o","markersize",4);

    hold on;

    plot(RUIDOS(m2,6),RUIDOS(m2,10+p), ...
        "x","markersize",5);

    plot([29 51],[29 51],"k--");

    xlim([29 51]);

    ylim([29 51]);

    ylabel("SNR real (dB)");

    title(sprintf("Fase %s",fases{p}));

    if p==1

      legend({"25 kHz","200 kHz","Ideal"}, ...
          "location","northoutside", ...
          "orientation","horizontal");

    endif

    grid on;

    if p==3

      xlabel("SNR alvo (dB)");

    endif

  endfor

  exportar(h,FIG,"G4_E03_03_SNR.png");

  %% ----------------------------------------------------------
  %% FIGURA 04 - Distribuicao do erro de SNR.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1100 650]);

  hold on;

  for p=1:3

    x = sort(ERRO_SNR(:,p));

    y = (1:numel(x))'/numel(x);

    plot(x,y,"linewidth",1.5);

  endfor

  xlabel("SNR realizado - SNR alvo (dB)");

  ylabel("Probabilidade acumulada");

  title("Distribuicao do erro de SNR");

  legend(fases,"location","southeast");

  grid on;

  exportar(h,FIG,"G4_E03_04_Erro_SNR.png");

  %% ----------------------------------------------------------
  %% FIGURA 05 - Ganhos gerados.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1300 900]);

  for p=1:3

    subplot(3,1,p);

    m = PARAM_SUM(:,3)==2 | PARAM_SUM(:,3)==4;

    x = PARAM_SUM(m,6+p);

    [a,b] = hist(x,48);

    bar(b,a);

    ylabel("Frequencia");

    title(sprintf("Ganho da fase %s",fases{p}));

    grid on;

    if p==3

      xlabel("Ganho multiplicativo");

    endif

  endfor

  exportar(h,FIG,"G4_E03_05_Ganhos.png");

  %% ----------------------------------------------------------
  %% FIGURA 06 - Transformacoes temporais.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1100 650]);

  m = abs(PARAM_SUM(:,10)-1)>1e-15;

  x = 100*(PARAM_SUM(m,10)-1);

  [a,b] = hist(x,40);

  bar(b,a);

  xlabel("Alteracao de alpha (%)");

  ylabel("Frequencia");

  title("Compressao e dilatacao temporal");

  grid on;

  exportar(h,FIG,"G4_E03_06_Transformacoes_Temporais.png");

  %% ----------------------------------------------------------
  %% FIGURA 07 - Amplitudes de modulacao.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1250 850]);

  for p=1:3

    subplot(3,1,p);

    x = 100*PARAM_SUM(:,10+p);

    x = x(x>0);

    [a,b] = hist(x,30);

    bar(b,a);

    ylabel("Frequencia");

    title(sprintf("Modulacao - fase %s",fases{p}));

    grid on;

    if p==3

      xlabel("Amplitude de modulacao (%)");

    endif

  endfor

  exportar(h,FIG,"G4_E03_07_Modulacoes.png");

  %% ----------------------------------------------------------
  %% FIGURA 08 - Energia, RMS e pico.
  %%
  %% Mediana e intervalo P05-P95.
  %% Apenas variantes de treinamento.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1450 950]);

  cols = [8 10 12 13 14 16];

  titulos = { ...
      "Erro RMS", ...
      "Erro de pico", ...
      "Erro de energia", ...
      "NRMSE maximo", ...
      "Distancia espectral", ...
      "Erro RMS trifasico"};

  for q=1:6

    subplot(2,3,q);

    med = zeros(4,1);

    lo = zeros(4,1);

    hi = zeros(4,1);

    for fam=1:4

      m = SUMMARY(:,2)==1 & SUMMARY(:,3)==fam;

      z = SUMMARY(m,cols(q));

      med(fam) = median(z);

      lo(fam) = med(fam)-percentil(z,5);

      hi(fam) = percentil(z,95)-med(fam);

    endfor

    errorbar(1:4,med,lo,hi,"o");

    xlim([0.5 4.5]);

    set(gca,"xtick",1:4, ...
        "xticklabel",{"A","B","C","D"});

    title(titulos{q});

    if q!=5

      ylabel("Alteracao (%)");

    endif

    grid on;

  endfor

  exportar(h,FIG,"G4_E03_08_Metricas_Familias.png");

  %% ----------------------------------------------------------
  %% FIGURA 09 - Diferenca de desequilibrio.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1200 900]);

  for reg=1:3

    subplot(3,1,reg);

    hold on;

    for fam=1:4

      m = GLOBAL_SUM(:,2)==1 & ...
          GLOBAL_SUM(:,3)==fam & ...
          GLOBAL_SUM(:,5)==reg;

      x = sort(GLOBAL_SUM(m,11));

      y = (1:numel(x))'/numel(x);

      plot(x,y,"linewidth",1.4, ...
          "color",CORES(fam+1,:));

    endfor

    title(REGIOES{reg});

    ylabel("Prob. acumulada");

    grid on;

    if reg==1

      legend(NOMES_FAMILIAS(2:5), ...
          "location","northoutside", ...
          "orientation","horizontal");

    endif

    if reg==3

      xlabel("Alteracao do desequilibrio (p.p.)");

    endif

  endfor

  exportar(h,FIG,"G4_E03_09_Desequilibrio.png");

  %% ----------------------------------------------------------
  %% FIGURAS 10-12 - PCA por regime.
  %%
  %% PCA descritiva, nao utilizada para selecionar variantes.
  %% ----------------------------------------------------------

  for reg=1:3

    h = figure("position",[100 100 1250 750]);

    Z = PCMAP{reg};

    hold on;

    for fam=0:4

      m = SUMMARY(:,2)==1 & SUMMARY(:,3)==fam;

      if fam==0

        scatter(Z(m,1),Z(m,2),55, ...
            CORES(fam+1,:),"filled");

      else

        scatter(Z(m,1),Z(m,2),20, ...
            CORES(fam+1,:),"filled");

      endif

    endfor

    m = SUMMARY(:,2)==2;

    plot(Z(m,1),Z(m,2),"kx","markersize",5);

    xlabel("PC1");

    ylabel("PC2");

    title(sprintf("Diversidade preliminar - %s",REGIOES{reg}));

    legend([NOMES_FAMILIAS,{"Validacao"}], ...
        "location","northoutside", ...
        "orientation","horizontal");

    grid on;

    exportar(h,FIG,sprintf( ...
        "G4_E03_%02d_PCA_%s.png",9+reg,REGIOES{reg}));

  endfor

  %% ----------------------------------------------------------
  %% FIGURA 13 - Variancia multivariada.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1200 850]);

  for reg=1:3

    subplot(3,1,reg);

    m = DIVERSIDADE(:,1)==reg & ...
        DIVERSIDADE(:,2)==1;

    V = DIVERSIDADE(m,5);

    bar(V);

    set(gca,"xtick",1:5, ...
        "xticklabel",NOMES_FAMILIAS);

    ylabel("Variancia total");

    title(REGIOES{reg});

    grid on;

  endfor

  exportar(h,FIG,"G4_E03_13_Variancia_Multivariada.png");

  %% ----------------------------------------------------------
  %% FIGURA 14 - Distancia ao original.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1250 850]);

  for reg=1:3

    subplot(3,1,reg);

    m = DIVERSIDADE(:,1)==reg & ...
        DIVERSIDADE(:,2)==1 & ...
        DIVERSIDADE(:,3)>0;

    med = DIVERSIDADE(m,7);

    lo = med-DIVERSIDADE(m,8);

    hi = DIVERSIDADE(m,9)-med;

    errorbar(1:4,med,lo,hi,"o");

    xlim([0.5 4.5]);

    set(gca,"xtick",1:4, ...
        "xticklabel",NOMES_FAMILIAS(2:5));

    ylabel("Distancia normalizada");

    title(REGIOES{reg});

    grid on;

  endfor

  exportar(h,FIG,"G4_E03_14_Distancia_Original.png");

  %% ----------------------------------------------------------
  %% FIGURA 15 - Posto efetivo da covariancia.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1200 850]);

  for reg=1:3

    subplot(3,1,reg);

    m = DIVERSIDADE(:,1)==reg & ...
        DIVERSIDADE(:,2)==1;

    bar(DIVERSIDADE(m,6));

    set(gca,"xtick",1:5, ...
        "xticklabel",NOMES_FAMILIAS);

    ylabel("Posto efetivo");

    title(REGIOES{reg});

    grid on;

  endfor

  exportar(h,FIG,"G4_E03_15_Posto_Efetivo.png");

  %% ----------------------------------------------------------
  %% FIGURA 16 - LHS D4.
  %%
  %% Projecao de duas dimensoes do planejamento.
  %% Nao alega cobertura exaustiva de todas as dimensoes.
  %% ----------------------------------------------------------

  h = figure("position",[100 100 1050 750]);

  hold on;

  for r=TREINO

    m = LHSCONS(:,1)==r & LHSCONS(:,3)==4;

    plot(LHSCONS(m,5),LHSCONS(m,17), ...
        "o","markersize",7);

  endfor

  xlabel("Parametro LHS do ganho A");

  ylabel("Parametro LHS do SNR");

  xlim([0 1]);

  ylim([0 1]);

  title("Cobertura marginal - Familia D4");

  legend(arrayfun(@(r) sprintf("R%02d",r), ...
      TREINO,"uniformoutput",false), ...
      "location","northoutside", ...
      "orientation","horizontal");

  grid on;

  exportar(h,FIG,"G4_E03_16_Cobertura_LHS.png");

endif

%% ============================================================
%% SECAO 32 - DECISAO DE INTEGRIDADE
%% ============================================================

fprintf("\n========================================\n");

fprintf(" CONSOLIDACAO DEFINITIVA\n");

fprintf("========================================\n");

fprintf("\nFAMILIA                TREINO    VAL    TOTAL\n");

for fam=0:4

  fprintf("%-20s %6d %6d %8d\n", ...
      NOMES_FAMILIAS{fam+1}, ...
      CONT(fam+1,1), ...
      CONT(fam+1,2), ...
      sum(CONT(fam+1,:)));

endfor

fprintf("---------------------------------------------\n");

fprintf("%-20s %6d %6d %8d\n", ...
    "TOTAL",sum(CONT(:,1)),sum(CONT(:,2)),TOTAL);

fprintf("\nRepresentacoes de 25 kHz: %d\n", ...
    sum(CONT_FS(:,1)));

fprintf("Representacoes de 200 kHz: %d\n", ...
    sum(CONT_FS(:,2)));

assert(sum(CONT_FS(:,1))==430);

assert(sum(CONT_FS(:,2))==60);

fprintf("\nOriginais canonicos: %d\n",sum(CONT(1,:)));

fprintf("Variantes artificiais: %d\n",TOTAL-10);

fprintf("Fator de expansao: %.2fx\n",TOTAL/10);

fprintf("Linhas de metricas por fase/regime: %d\n", ...
    rows(STATS_REGISTROS));

fprintf("Linhas de metricas trifasicas: %d\n", ...
    rows(GLOBAL_SUM));

fprintf("Planejamentos LHS: %d\n",rows(LHS_AUDIT));

fprintf("Variantes com AWGN: %d\n",rows(RUIDOS));

fprintf("Armazenamento: %.2f MB\n",TOTAL_BYTES/1024^2);

fprintf("Armazenamento: %.4f GB\n",TOTAL_BYTES/1024^3);

fprintf("Tempo total: %.2f segundos\n",toc(TSTART));

%% ============================================================
%% SECAO 33 - ARQUIVO DE DECISAO
%% ============================================================

f = fopen(fullfile(OUT,"G4_E03_Decisao.txt"),"w");

assert(f>0);

fprintf(f,"GATE 4 - EXPERIMENTO 03\n");

fprintf(f,"EXPANSAO DEFINITIVA\n\n");

fprintf(f,"STATUS=GERACAO_CONCLUIDA\n");

fprintf(f,"APROVACAO_ESTATISTICA_FINAL=PENDENTE_E04\n");

fprintf(f,"Total_registros=%d\n",TOTAL);

fprintf(f,"Originais_canonicos=10\n");

fprintf(f,"Variantes_artificiais=%d\n",TOTAL-10);

fprintf(f,"Treinamento=%d\n",sum(CONT(:,1)));

fprintf(f,"Validacao=%d\n",sum(CONT(:,2)));

fprintf(f,"Familia_A=%d\n",sum(CONT(2,:)));

fprintf(f,"Familia_B=%d\n",sum(CONT(3,:)));

fprintf(f,"Familia_C=%d\n",sum(CONT(4,:)));

fprintf(f,"Familia_D=%d\n",sum(CONT(5,:)));

fprintf(f,"Representacoes_25k=%d\n",sum(CONT_FS(:,1)));

fprintf(f,"Representacoes_200k=%d\n",sum(CONT_FS(:,2)));

fprintf(f,"IDs_exclusivos=%d\n",numel(unique(IDS)));

fprintf(f,"Bootstrap=0\n");

fprintf(f,"Treinamento_IA=0\n");

fprintf(f,"Reservados_acessados=0\n");

fprintf(f,"Variantes_descartadas_por_separabilidade=0\n");

fprintf(f,"Ensaios_fisicos_adicionais=0\n");

fprintf(f,"Bytes_totais=%.0f\n",TOTAL_BYTES);

fprintf(f,"Tempo_segundos=%.6f\n",toc(TSTART));

fprintf(f,"\nREQUISITOS PARA E04:\n");

fprintf(f,"1. Auditar integridade de todos os arquivos MAT.\n");

fprintf(f,"2. Auditar o manifesto e linhagem das variantes.\n");

fprintf(f,"3. Quantificar diversidade marginal e multivariada.\n");

fprintf(f,"4. Comparar distribuicoes por familia e regime.\n");

fprintf(f,"5. Auditar redundancia e vizinhos proximos.\n");

fprintf(f,"6. Auditar ACF e periodicidade.\n");

fprintf(f,"7. Auditar espectros e espectrogramas.\n");

fprintf(f,"8. Verificar efeitos da interpolacao e do FIR.\n");

fprintf(f,"9. Auditar regiao de fronteira do evento.\n");

fprintf(f,"10. Distinguir ruido antes e depois do ZOH.\n");

fprintf(f,"11. Avaliar cobertura do espaco de perturbacoes.\n");

fprintf(f,"12. Consolidar a base elegivel para o Gate 5.\n");

fprintf(f,"\nLIMITACOES:\n");

fprintf(f,"A geracao bem-sucedida nao comprova relevancia estatistica final.\n");

fprintf(f,"A diversidade preliminar nao comprova generalizacao fisica.\n");

fprintf(f,"As unidades experimentais permanecem os registros de origem.\n");

fprintf(f,"A identidade pre-evento limita a independencia entre originais.\n");

fprintf(f,"A utilidade das familias sera avaliada no Gate 5.\n");

fprintf(f,"O Gate 4 permanece aberto ate decisao do pesquisador.\n");

fclose(f);

%% ============================================================
%% SECAO 34 - MARCADOR DE CONCLUSAO
%% ============================================================

f = fopen(fullfile(OUT,"GERACAO_CONCLUIDA.txt"),"w");

assert(f>0);

fprintf(f,"G4/E03 - GERACAO CONCLUIDA\n");

fprintf(f,"490 registros gerados e catalogados.\n");

fprintf(f,"480 variantes artificiais.\n");

fprintf(f,"Auditoria estatistica definitiva pendente do E04.\n");

fclose(f);

delete(MARCADOR);

%% ============================================================
%% SECAO 35 - ENCERRAMENTO
%% ============================================================

fprintf("\n========================================\n");

fprintf(" G4/E03 - EXECUCAO CONCLUIDA\n");

fprintf("========================================\n");

fprintf("Regularizacao temporal: EXECUTADA\n");

fprintf("Reconstrucao ZOH: VERIFICADA\n");

fprintf("Originais canonicos: GERADOS\n");

fprintf("Familia A - AWGN: GERADA\n");

fprintf("Familia B - Ganho: GERADA\n");

fprintf("Familia C - Temporal: GERADA\n");

fprintf("Familia D - Combinada: GERADA\n");

fprintf("Latin Hypercube: VERIFICADO\n");

fprintf("Rastreabilidade: VERIFICADA\n");

fprintf("Manifesto: PRODUZIDO\n");

fprintf("Parametros: REGISTRADOS\n");

fprintf("Integridade estrutural: VERIFICADA\n");

fprintf("Diversidade preliminar: ANALISADA\n");

fprintf("Bootstrap: EXCLUIDO\n");

fprintf("Treinamento de IA: NENHUM\n");

fprintf("Registros 06 e 12: PRESERVADOS\n");

fprintf("Arquivos proprietarios: PRESERVADOS\n");

fprintf("Aprovacao cientifica final: PENDENTE E04\n");

fprintf("\nDiretorio da base expandida:\n%s\n",OUT);

fprintf("\nProxima etapa:\n");

fprintf("G4/E04 - AUDITORIA ESTATISTICA E CONSOLIDACAO\n");

fprintf("\nAguardando avaliacao do pesquisador.\n");

fprintf("========================================\n");

diary off;
