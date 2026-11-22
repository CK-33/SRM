
1;

% ============================================================
% PESQUISA SRM
% GATE 4 - EXPERIMENTO 02.3
%
% Auditoria da resolucao temporal efetiva
% Retencao temporal, PSD, STFT e AWGN
%
% GNU Octave
% ============================================================

function y = rms_local(x)
  y = sqrt(mean(x.^2, 1));
endfunction


function y = energia_local(x, fs)
  y = sum(x.^2, 1) / fs;
endfunction


function y = db_local(x)
  y = 10 * log10(max(x, realmin));
endfunction


function [fase, tabela] = detectar_grade(X, M)

  D = abs(diff(X));

  tabela = zeros(M, 3);

  for p = 1:3

    tol = max(1e-10, max(abs(X(:,p))) * 1e-10);

    idx = find(D(:,p) > tol) + 1;

    if isempty(idx)
      error("Nao foram encontradas transicoes.");
    endif

    classes = mod(idx - 1, M) + 1;

    for k = 1:M
      tabela(k,p) = sum(classes == k);
    endfor

  endfor

  [~, fase] = max(sum(tabela, 2));

endfunction


function [U, R, idx, Xtrim] = reconstruir_zoh(X, M, fase)

  N = rows(X);

  K = floor((N - fase + 1) / M);

  if K < 2
    error("Quantidade insuficiente de blocos.");
  endif

  idx = fase + (0:K-1)' * M;

  U = X(idx,:);

  Xtrim = X(fase:fase + K*M - 1,:);

  R = kron(U, ones(M,1));

endfunction


function [P, f] = psd_local(x, fs, duracao)

  x = x(:);

  L = min(length(x), round(duracao * fs));

  if L < 8
    error("Segmento insuficiente para PSD.");
  endif

  w = 0.5 - 0.5*cos(2*pi*(0:L-1)'/L);

  hop = max(1, floor(L/2));

  nfft = L;

  K = floor(nfft/2) + 1;

  P = zeros(K,1);

  count = 0;

  for a = 1:hop:length(x)-L+1

    z = x(a:a+L-1);

    z = z .* w;

    Z = fft(z,nfft);

    Q = abs(Z(1:K)).^2 / (fs*sum(w.^2));

    if mod(nfft,2) == 0
      Q(2:end-1) = 2*Q(2:end-1);
    else
      Q(2:end) = 2*Q(2:end);
    endif

    P = P + Q;

    count = count + 1;

  endfor

  P = P / count;

  f = (0:K-1)' * fs / nfft;

endfunction


function E = energia_banda(P, f, f1, f2)

  df = f(2)-f(1);

  idx = (f >= f1) & (f < f2);

  E = sum(P(idx)) * df;

endfunction


function [S, f, t] = stft_local(x, fs, Tw, Th)

  x = x(:);

  L = round(Tw * fs);

  hop = max(1,round(Th * fs));

  nfft = L;

  K = floor(nfft/2)+1;

  w = 0.5 - 0.5*cos(2*pi*(0:L-1)'/L);

  inicios = 1:hop:length(x)-L+1;

  S = zeros(K,length(inicios));

  for j = 1:length(inicios)

    a = inicios(j);

    segmento = x(a:a+L-1).*w;

    Y = fft(segmento,nfft);

    P = abs(Y(1:K)).^2 / (fs*sum(w.^2));

    if mod(nfft,2) == 0
      P(2:end-1) = 2*P(2:end-1);
    else
      P(2:end) = 2*P(2:end);
    endif

    S(:,j) = P;

  endfor

  f = (0:K-1)' * fs/nfft;

  t = (inicios-1+(L-1)/2)/fs;

endfunction


function [Y, snr_real] = awgn_local(X, snr_db)

  potencia = mean(X.^2,1);

  ruido = randn(size(X));

  pr = mean(ruido.^2,1);

  alvo = potencia / (10^(snr_db/10));

  ruido = bsxfun(@times, ruido, sqrt(alvo ./ max(pr,realmin)));

  Y = X + ruido;

  snr_real = 10*log10(potencia ./ ...
                     max(mean((Y-X).^2,1),realmin));

endfunction


function frac = fracao_grade(X, M, fase)

  frac = zeros(1,3);

  for p = 1:3

    d = abs(diff(X(:,p)));

    tol = max(1e-10,max(abs(X(:,p)))*1e-10);

    idx = find(d > tol)+1;

    if isempty(idx)
      frac(p) = NaN;
    else
      frac(p) = mean(mod(idx-1,M)+1 == fase);
    endif

  endfor

endfunction


function exportar(fig, pasta, nome)

  arquivo = fullfile(pasta,nome);

  print(fig,arquivo,"-dpng","-r150");

  close(fig);

  fprintf("Figura exportada: %s\n",nome);

endfunction


% ============================================================
% CONFIGURACAO
% ============================================================

clc;
clear;
close all;

graphics_toolkit("gnuplot");

set(0,"defaultfigurevisible","off");

rng(42);

ROOT = ["PASTA_DO_USUARIO/Documents/Pesquisa_SRM/" ...
        "TesteNo1_31OCT2026/Data and Scripts"];

treino = [1 2 4 5 7 8 10 11];

validacao = [3 9];

registros = sort([treino validacao]);

assert(!any(ismember(registros,[6 12])));

fs = 200000;

M = 8;

fs2 = fs/M;

Tref = 0.01308;

Njan = round(Tref*fs);

Ttrans = 0.200;

Tpsd = 0.020;

Tw = 0.008;

Th = 0.001;

SNRs = [30 40 50];

Nrep = 10;

nomes_fases = {"A","B","C"};

nomes_regioes = {"pre","transitorio","tardio"};

eventos = [1.0017 1.0019 1.0021 1.0024 ...
           1.0026 1.0030 1.0040 1.0050 ...
           1.0060 1.0070 1.0075 1.0080];

bandas = [0 5000;
          5000 12500;
          12500 20000;
          20000 50000;
          50000 100000];

% Criacao de diretorio sem sobrescrever execucoes.

run = 1;

while true

  OUT = fullfile(ROOT,sprintf("Gate4_Exp02_3_Run%02d",run));

  if !exist(OUT,"dir")
    mkdir(OUT);
    break;
  endif

  run = run+1;

endwhile

diary(fullfile(OUT,"G4_E02_3_Console.txt"));

fprintf("\n========================================\n");
fprintf(" GATE 4 - EXPERIMENTO 02.3\n");
fprintf(" AUDITORIA DA RESOLUCAO TEMPORAL\n");
fprintf("========================================\n");

fprintf("Fs original: %.0f Hz\n",fs);
fprintf("Fs candidata: %.0f Hz\n",fs2);
fprintf("Nyquist candidato: %.0f Hz\n",fs2/2);
fprintf("Fator de retencao: %d\n",M);
fprintf("Amostras por janela original: %d\n",Njan);
fprintf("Amostras por janela reduzida: %d\n",Njan/M);
fprintf("Registros 06 e 12: RESERVADOS\n");
fprintf("Saida: %s\n",OUT);

% ============================================================
% ARQUIVOS DE RESULTADOS
% ============================================================

fg = fopen(fullfile(OUT,"G4_E02_3_Grade.csv"),"w");

fprintf(fg,["registro,grupo,fase,fase_grade," ...
            "transicoes_na_grade_pct,amostras_originais," ...
            "amostras_reduzidas,amostras_excluidas," ...
            "rmse_A,max_erro_A,nrmse_pct\n"]);

fm = fopen(fullfile(OUT,"G4_E02_3_Grandezas.csv"),"w");

fprintf(fm,["registro,grupo,regiao,fase," ...
            "rms_200k_A,rms_25k_A,erro_rms_pct," ...
            "pico_200k_A,pico_25k_A,erro_pico_pct," ...
            "energia_200k_A2s,energia_25k_A2s," ...
            "erro_energia_pct,rmse_zoh_A,max_erro_zoh_A\n"]);

fc = fopen(fullfile(OUT,"G4_E02_3_Correlacoes.csv"),"w");

fprintf(fc,["registro,grupo,regiao," ...
            "AB_200k,AC_200k,BC_200k," ...
            "AB_25k,AC_25k,BC_25k,erro_maximo\n"]);

fp = fopen(fullfile(OUT,"G4_E02_3_Espectral.csv"),"w");

fprintf(fp,["registro,grupo,regiao,fase," ...
            "f_inicio_Hz,f_fim_Hz," ...
            "potencia_200k_A2,potencia_25k_A2," ...
            "fracao_200k_pct,erro_relativo_pct\n"]);

fa = fopen(fullfile(OUT,"G4_E02_3_AWGN.csv"),"w");

fprintf(fa,["registro,grupo,regiao,fase,SNR_alvo_dB," ...
            "repeticao,metodo,SNR_real_dB," ...
            "transicoes_na_grade_pct," ...
            "ruido_acima_12k5_pct," ...
            "ruido_total_A2,ruido_banda_base_A2\n"]);

assert(all([fg fm fc fp fa] >= 0));

% Matrizes para consolidacao.

GRADE = [];
MET = [];
CORR = [];
SPEC = [];
AW = [];

% ============================================================
% PROCESSAMENTO PRINCIPAL
% ============================================================

for id = registros

  if ismember(id,treino)
    grupo = "treino";
    gid = 1;
  else
    grupo = "validacao";
    gid = 2;
  endif

  arquivo = fullfile(ROOT,"Results_1_to_12", ...
                   sprintf("results_%d.mat",id));

assert(exist(arquivo,"file") == 2, ...
       sprintf("Arquivo nao encontrado: %s",arquivo));

  dados = load(arquivo,"results");

  R = dados.results;

  t = R.time(:);

  X = squeeze(R.signals(3).values);

  if columns(X) != 3 && rows(X) == 3
    X = X';
  endif


assert(columns(X) == 3);
assert(rows(X) == length(t));
assert(all(isfinite(X(:))));
assert(all(isfinite(t)));
assert(all(diff(t) > 0));

% ==================================================
% REGULARIZACAO TEMPORAL VALIDADA NO G4/E02.1
% ==================================================

TS = 1/fs;

TOL_GRADE = 0.001*TS;

N_original = length(t);

Nu = round((t(end)-t(1))/TS)+1;

t_regular = t(1)+(0:Nu-1)'*TS;

k = round((t-t(1))/TS)+1;

validos = find(k >= 1 & k <= Nu);

validos = validos( ...
  abs(t(validos)-t_regular(k(validos))) ...
  <= TOL_GRADE);

contagem = accumarray( ...
  k(validos),ones(length(validos),1), ...
  [Nu 1],@sum,0);

assert(all(contagem > 0), ...
       "Grade nominal possui amostras ausentes.");

% Seleciona a ultima ocorrencia de cada instante.
% Mesmo indice para as tres fases.

indices = accumarray( ...
  k(validos),validos,[Nu 1],@max,0);

assert(all(indices > 0));

X = X(indices,:);

t = t_regular;

assert(rows(X) == length(t));

assert(all(abs(diff(t)-TS) < 1e-10));

fprintf("\nRegularizacao temporal R%02d:\n",id);
fprintf("Amostras brutas: %d\n",N_original);
fprintf("Amostras regularizadas: %d\n",Nu);
fprintf("Amostras removidas: %d\n",N_original-Nu);
fprintf("Pontos ausentes: %d\n",sum(contagem==0));
fprintf("Pontos com multiplas ocorrencias: %d\n", ...
        sum(contagem>1));

te = eventos(id);


  fprintf("\n----------------------------------------\n");
  fprintf("Registro %02d | %s\n",id,grupo);
  fprintf("----------------------------------------\n");

  % ----------------------------------------------------------
  % E02.3.1 - Grade de atualizacao
  % ----------------------------------------------------------

  [fase_grade, hist_grade] = detectar_grade(X,M);

  frac = fracao_grade(X,M,fase_grade);

  [U,Z,idx,Xtrim] = reconstruir_zoh(X,M,fase_grade);

  tu = t(idx);

  Ttrim = t(fase_grade:fase_grade+rows(Z)-1);

  erro = Xtrim-Z;

  erms = rms_local(erro);

  emax = max(abs(erro),[],1);

  enrmse = 100*erms./max(rms_local(Xtrim),realmin);

  nexcl = rows(X)-rows(Z);

  fprintf("Grade dominante: %d de %d\n",fase_grade,M);

  fprintf("Amostras originais: %d\n",rows(X));
  fprintf("Amostras reduzidas: %d\n",rows(U));
  fprintf("Amostras de borda excluidas: %d\n",nexcl);

  for p = 1:3

    fprintf(["Fase %s | grade: %.8f %% | " ...
             "RMSE: %.12g A | max: %.12g A\n"], ...
            nomes_fases{p},100*frac(p),erms(p),emax(p));

    fprintf(fg,["%d,%s,%s,%d,%.12g,%d,%d,%d," ...
                "%.12g,%.12g,%.12g\n"], ...
            id,grupo,nomes_fases{p},fase_grade, ...
            100*frac(p),rows(X),rows(U),nexcl, ...
            erms(p),emax(p),enrmse(p));

    GRADE(end+1,:) = [id gid p fase_grade ...
                      100*frac(p) erms(p) emax(p) ...
                      enrmse(p) nexcl];

  endfor

  % ----------------------------------------------------------
  % Exemplo visual da grade temporal
  % ----------------------------------------------------------

  if id == 1

    fig = figure("position",[100 100 1050 800]);

    for p = 1:3

      subplot(3,1,p);

      a = find(Ttrim < te-0.01,1,"last");

      a = max(1,a-159);

      b = min(rows(Z),a+159);

      tt = (Ttrim(a:b)-Ttrim(a))*1e6;

      plot(tt,Xtrim(a:b,p),"b.-");

      hold on;

      stairs(tt,Z(a:b,p),"r--","linewidth",1.2);

      grid on;

      ylabel("Corrente (A)");

      title(sprintf("Fase %s",nomes_fases{p}));

      legend("Original 200 kHz","Reconstrucao ZOH", ...
             "location","best");

      xlabel("Tempo relativo (us)");

    endfor

    exportar(fig,OUT,"G4_E02_3_01_Reconstrucao_ZOH.png");

    fig = figure("position",[100 100 950 500]);

    bar(0:M-1,hist_grade);

    grid on;

    xlabel("Classe temporal modulo 8");
    ylabel("Numero de transicoes");

    title("Distribuicao das transicoes por grade temporal");

    legend("Fase A","Fase B","Fase C");

    exportar(fig,OUT,"G4_E02_3_02_Grade_Temporal.png");

  endif

  % ----------------------------------------------------------
  % REGIOES TEMPORAIS
  %
  % Apenas blocos completos inteiramente contidos na regiao.
  % Isso evita misturar amostras entre regimes.
  % ----------------------------------------------------------

  tend = tu+(M-1)/fs;

  mascaras = {
    tend < te;
    tu >= te & tend < te+Ttrans;
    tu >= te+Ttrans
  };

  for reg = 1:3

    nome_reg = nomes_regioes{reg};

    k = find(mascaras{reg});

    assert(length(k) > 10);

    rr = bsxfun(@plus,(k-1)*M,1:M);

    rr = reshape(rr',[],1);

    x200 = Xtrim(rr,:);

    x25 = U(k,:);

    xzoh = Z(rr,:);

    fprintf("\nRegiao: %s\n",nome_reg);
    fprintf("Blocos completos: %d\n",length(k));

    % --------------------------------------------------------
    % E02.3.4 - Grandezas eletricas
    % --------------------------------------------------------

    r1 = rms_local(x200);
    r2 = rms_local(x25);

    p1 = max(abs(x200),[],1);
    p2 = max(abs(x25),[],1);

    e1 = energia_local(x200,fs);
    e2 = energia_local(x25,fs2);

    dr = 100*abs(r2-r1)./max(abs(r1),realmin);
    dp = 100*abs(p2-p1)./max(abs(p1),realmin);
    de = 100*abs(e2-e1)./max(abs(e1),realmin);

    rez = rms_local(x200-xzoh);

    mez = max(abs(x200-xzoh),[],1);

    for p = 1:3

      fprintf(["Fase %s | RMS 200k: %.8f | " ...
               "RMS 25k: %.8f | erro: %.9g %%\n"], ...
              nomes_fases{p},r1(p),r2(p),dr(p));

      fprintf(fm,["%d,%s,%s,%s," ...
                  "%.12g,%.12g,%.12g," ...
                  "%.12g,%.12g,%.12g," ...
                  "%.12g,%.12g,%.12g,%.12g,%.12g\n"], ...
              id,grupo,nome_reg,nomes_fases{p}, ...
              r1(p),r2(p),dr(p), ...
              p1(p),p2(p),dp(p), ...
              e1(p),e2(p),de(p),rez(p),mez(p));

      MET(end+1,:) = [id gid reg p ...
                      r1(p) r2(p) dr(p) ...
                      p1(p) p2(p) dp(p) ...
                      e1(p) e2(p) de(p)];

    endfor

    C1 = corrcoef(x200);
    C2 = corrcoef(x25);

    c1 = [C1(1,2) C1(1,3) C1(2,3)];
    c2 = [C2(1,2) C2(1,3) C2(2,3)];

    ec = max(abs(c2-c1));

    fprintf(fc,["%d,%s,%s,%.12g,%.12g,%.12g," ...
                "%.12g,%.12g,%.12g,%.12g\n"], ...
            id,grupo,nome_reg,c1,c2,ec);

    CORR(end+1,:) = [id gid reg c1 c2 ec];

    % --------------------------------------------------------
    % E02.3.3 e E02.3.5 - PSD e bandas
    % --------------------------------------------------------

    if id == 1 || id == 3
      fig = figure("position",[100 100 1100 850]);
    endif

    for p = 1:3

      [P1,f1] = psd_local(x200(:,p),fs,Tpsd);

      [P2,f2] = psd_local(x25(:,p),fs2,Tpsd);

      Etot = sum(P1)*(f1(2)-f1(1));

      alta = sum(P1(f1 > fs2/2))*(f1(2)-f1(1));

      fprintf("Fase %s | energia acima de 12.5 kHz: %.6f %%\n", ...
              nomes_fases{p},100*alta/max(Etot,realmin));

      for b = 1:rows(bandas)

        lo = bandas(b,1);
        hi = bandas(b,2);

        E1 = energia_banda(P1,f1,lo,hi);

        if hi <= fs2/2

          E2 = energia_banda(P2,f2,lo,hi);

          rel = 100*abs(E2-E1)/max(E1,realmin);

        else

          E2 = NaN;
          rel = NaN;

        endif

        fr = 100*E1/max(Etot,realmin);

        fprintf(fp,["%d,%s,%s,%s,%.0f,%.0f," ...
                    "%.12g,%.12g,%.12g,%.12g\n"], ...
                id,grupo,nome_reg,nomes_fases{p},lo,hi, ...
                E1,E2,fr,rel);

        SPEC(end+1,:) = [id gid reg p b E1 E2 fr rel];

      endfor

      if id == 1 || id == 3

        subplot(3,1,p);

        plot(f1/1000,db_local(P1),"b","linewidth",1);

        hold on;

        plot(f2/1000,db_local(P2),"r","linewidth",1.2);

        yl = ylim();

        line([12.5 12.5],yl, ...
             "color",[0.3 0.3 0.3],"linestyle","--");

        grid on;

        xlim([0 100]);

        ylabel("PSD (dB A^2/Hz)");

        title(sprintf("Fase %s | %s", ...
                      nomes_fases{p},nome_reg));

        legend("200 kHz","25 kHz","Nyquist candidato");

        xlabel("Frequencia (kHz)");

      endif

    endfor

    if id == 1 || id == 3

      exportar(fig,OUT, ...
        sprintf("G4_E02_3_PSD_R%02d_%s.png",id,nome_reg));

    endif

    % --------------------------------------------------------
    % E02.3.7 - AWGN antes e depois da retencao
    %
    % Metodo 1: ruido na grade de 25 kHz e posterior ZOH.
    % Metodo 2: ruido independente na grade de 200 kHz.
    %
    % Comparacao com mesma potencia global de ruido.
    % --------------------------------------------------------

    for sn = SNRs

      for rep = 1:Nrep

        [u_noise,sn1] = awgn_local(x25,sn);

        y1 = kron(u_noise,ones(M,1));

        [y2,sn2] = awgn_local(xzoh,sn);

        ru1 = y1-xzoh;
        ru2 = y2-xzoh;

        Y = {y1,y2};

        RU = {ru1,ru2};

        metodos = {"AWGN_25k_ZOH","AWGN_200k"};

        for m = 1:2

          yy = Y{m};
          ruido = RU{m};

          grade_noise = fracao_grade(yy,M,1);

          snr_real = 10*log10( ...
            mean(xzoh.^2,1)./max(mean(ruido.^2,1),realmin));

          for p = 1:3

            [Pn,fn] = psd_local(ruido(:,p),fs,Tpsd);

            df = fn(2)-fn(1);

            total = sum(Pn)*df;

            base = sum(Pn(fn <= fs2/2))*df;

            alta = sum(Pn(fn > fs2/2))*df;

            frac_alta = 100*alta/max(total,realmin);

            fprintf(fa,["%d,%s,%s,%s,%d,%d,%s," ...
                        "%.12g,%.12g,%.12g,%.12g,%.12g\n"], ...
                    id,grupo,nome_reg,nomes_fases{p}, ...
                    sn,rep,metodos{m},snr_real(p), ...
                    100*grade_noise(p),frac_alta,total,base);

            AW(end+1,:) = [id gid reg p sn rep m ...
                           snr_real(p) 100*grade_noise(p) ...
                           frac_alta total base];

          endfor

        endfor

        % Comparacao visual em um registro representativo.

        if id == 1 && reg == 1 && sn == 30 && rep == 1

          fig = figure("position",[100 100 1050 850]);

          Nplot = min(240,rows(xzoh));

          tt = (0:Nplot-1)'/fs*1e6;

          subplot(3,1,1);

          stairs(tt,xzoh(1:Nplot,2),"b");

          grid on;
          title("Original reconstruido");
          ylabel("Corrente B (A)");

          subplot(3,1,2);

          stairs(tt,y1(1:Nplot,2),"r");

          grid on;
          title("AWGN a 25 kHz seguido de ZOH");
          ylabel("Corrente B (A)");

          subplot(3,1,3);

          plot(tt,y2(1:Nplot,2),"k");

          grid on;
          title("AWGN diretamente a 200 kHz");

          ylabel("Corrente B (A)");
          xlabel("Tempo (us)");

          exportar(fig,OUT,"G4_E02_3_03_AWGN_Temporal.png");

          fig = figure("position",[100 100 1000 500]);

          [Pna,fna] = psd_local(ru1(:,2),fs,Tpsd);

          [Pnb,fnb] = psd_local(ru2(:,2),fs,Tpsd);

          plot(fna/1000,db_local(Pna),"b");

          hold on;

          plot(fnb/1000,db_local(Pnb),"r");

          grid on;

          xlabel("Frequencia (kHz)");
          ylabel("PSD do ruido (dB A^2/Hz)");

          title("Comparacao espectral do AWGN");

          legend("AWGN 25 kHz + ZOH","AWGN 200 kHz");

          exportar(fig,OUT,"G4_E02_3_04_AWGN_PSD.png");

        endif

      endfor

    endfor

  endfor

  % ----------------------------------------------------------
  % E02.3.6 - Espectrogramas preliminares
  %
  % Mesma duracao de janela e mesmo salto temporal.
  % Escala cromatica comum entre as representacoes.
  % ----------------------------------------------------------

  if id == 1 || id == 3 || id == 9

    for p = 1:3

      [S1,F1,T1] = stft_local(Xtrim(:,p),fs,Tw,Th);

      [S2,F2,T2] = stft_local(U(:,p),fs2,Tw,Th);

      T1 = T1 + Ttrim(1)-te;

      T2 = T2 + tu(1)-te+(M-1)/(2*fs);

      D1 = db_local(S1);

      D2 = db_local(S2);

      cima = max([D1(:);D2(:)]);

      limites = [cima-80 cima];

      fig = figure("position",[100 100 1150 950]);

      subplot(3,1,1);

      imagesc(T1*1000,F1/1000,D1);

      axis xy;

      caxis(limites);

      colorbar;

      title("Representacao 200 kHz - banda completa");

      ylabel("Frequencia (kHz)");

      hold on;

      line([0 0],[0 100],"color","w","linestyle","--");

      subplot(3,1,2);

      ii = F1 <= fs2/2;

      imagesc(T1*1000,F1(ii)/1000,D1(ii,:));

      axis xy;

      caxis(limites);

      colorbar;

      title("Representacao 200 kHz - banda base");

      ylabel("Frequencia (kHz)");

      hold on;

      line([0 0],[0 12.5],"color","w","linestyle","--");

      subplot(3,1,3);

      imagesc(T2*1000,F2/1000,D2);

      axis xy;

      caxis(limites);

      colorbar;

      title("Representacao candidata 25 kHz");

      ylabel("Frequencia (kHz)");

      xlabel("Tempo relativo ao evento (ms)");

      hold on;

      line([0 0],[0 12.5],"color","w","linestyle","--");

      colormap(jet(256));

      exportar(fig,OUT, ...
        sprintf("G4_E02_3_STFT_R%02d_Fase%s.png", ...
                id,nomes_fases{p}));

    endfor

  endif

  clear dados R X U Z Xtrim x200 x25 xzoh;

endfor

fclose(fg);
fclose(fm);
fclose(fc);
fclose(fp);
fclose(fa);

% ============================================================
% CONSOLIDACAO
% ============================================================

fprintf("\n========================================\n");
fprintf(" RESULTADOS CONSOLIDADOS\n");
fprintf("========================================\n");

fprintf("Registros processados: %d\n",length(registros));

fprintf("Frequencia registrada: %.0f Hz\n",fs);

fprintf("Frequencia candidata: %.0f Hz\n",fs2);

for g = 1:2

  sel = GRADE(:,2) == g;

  if g == 1
    nome = "TREINAMENTO";
  else
    nome = "VALIDACAO";
  endif

  fprintf("\n%s\n",nome);

  fprintf("Grade dominante - cobertura minima: %.12g %%\n", ...
          min(GRADE(sel,5)));

  fprintf("Grade dominante - cobertura media: %.12g %%\n", ...
          mean(GRADE(sel,5)));

  fprintf("RMSE ZOH medio: %.12g A\n",mean(GRADE(sel,6)));

  fprintf("RMSE ZOH maximo: %.12g A\n",max(GRADE(sel,6)));

  fprintf("Erro absoluto maximo: %.12g A\n",max(GRADE(sel,7)));

endfor

fprintf("\nPRESERVACAO DAS GRANDEZAS\n");

fprintf("Erro maximo RMS: %.12g %%\n",max(MET(:,7)));

fprintf("Erro maximo pico: %.12g %%\n",max(MET(:,10)));

fprintf("Erro maximo energia: %.12g %%\n",max(MET(:,13)));

fprintf("Erro maximo correlacoes: %.12g\n",max(CORR(:,10)));

% ============================================================
% RESUMO ESPECTRAL POR REGIME
% ============================================================

fprintf("\nDISTRIBUICAO ESPECTRAL - TREINAMENTO\n");

fsum = fopen(fullfile(OUT,"G4_E02_3_Resumo_Espectral.csv"),"w");

fprintf(fsum,["grupo,regiao,fase,banda_inicio_Hz," ...
              "banda_fim_Hz,fracao_media_pct," ...
              "fracao_dp_pct,erro_medio_pct\n"]);

for g = 1:2

  if g == 1
    grupo = "treino";
  else
    grupo = "validacao";
  endif

  for reg = 1:3

    for p = 1:3

      if g == 1

        fprintf("\n%s | Fase %s\n", ...
                nomes_regioes{reg},nomes_fases{p});

      endif

      for b = 1:rows(bandas)

        s = SPEC(:,2)==g & SPEC(:,3)==reg & ...
            SPEC(:,4)==p & SPEC(:,5)==b;

        v = SPEC(s,8);

        err = SPEC(s,9);

        err = err(isfinite(err));

        if isempty(err)
          em = NaN;
        else
          em = mean(err);
        endif

        if g == 1

          fprintf("%.1f-%.1f kHz: %.8f +/- %.8f %%\n", ...
                  bandas(b,1)/1000,bandas(b,2)/1000, ...
                  mean(v),std(v));

        endif

        fprintf(fsum,"%s,%s,%s,%.0f,%.0f,%.12g,%.12g,%.12g\n", ...
                grupo,nomes_regioes{reg},nomes_fases{p}, ...
                bandas(b,1),bandas(b,2),mean(v),std(v),em);

      endfor

    endfor

  endfor

endfor

fclose(fsum);

% ============================================================
% RESUMO AWGN
% ============================================================

fprintf("\nCOMPARACAO AWGN\n");

faz = fopen(fullfile(OUT,"G4_E02_3_Resumo_AWGN.csv"),"w");

fprintf(faz,["grupo,regiao,SNR_alvo_dB,metodo," ...
             "SNR_medio_dB,grade_media_pct," ...
             "ruido_alta_media_pct,ruido_alta_dp_pct\n"]);

for sn = SNRs

  fprintf("\nSNR alvo: %d dB\n",sn);

  for m = 1:2

    s = AW(:,5)==sn & AW(:,7)==m;

    fprintf(["Metodo %d | SNR: %.6f dB | " ...
             "grade: %.6f %% | " ...
             "ruido acima de Nyquist: %.6f %%\n"], ...
            m,mean(AW(s,8)),mean(AW(s,9)),mean(AW(s,10)));

    for g = 1:2

      for reg = 1:3

        ss = s & AW(:,2)==g & AW(:,3)==reg;

        if g == 1
          grupo = "treino";
        else
          grupo = "validacao";
        endif

        fprintf(faz,"%s,%s,%d,%d,%.12g,%.12g,%.12g,%.12g\n", ...
                grupo,nomes_regioes{reg},sn,m, ...
                mean(AW(ss,8)),mean(AW(ss,9)), ...
                mean(AW(ss,10)),std(AW(ss,10)));

      endfor

    endfor

  endfor

endfor

fclose(faz);

% ============================================================
% FIGURAS CONSOLIDADAS
% ============================================================

fig = figure("position",[100 100 1050 500]);

V = zeros(length(registros),3);

for j = 1:length(registros)

  for p = 1:3

    s = GRADE(:,1)==registros(j) & GRADE(:,3)==p;

    V(j,p) = GRADE(s,5);

  endfor

endfor

bar(V);

set(gca,"xtick",1:length(registros), ...
        "xticklabel",arrayfun(@num2str,registros, ...
                             "uniformoutput",false));

grid on;

xlabel("Registro");

ylabel("Transicoes na grade dominante (%)");

title("Consistencia da grade de atualizacao");

legend("Fase A","Fase B","Fase C");

exportar(fig,OUT,"G4_E02_3_05_Cobertura_Grade.png");


fig = figure("position",[100 100 1050 850]);

for p = 1:3

  subplot(3,1,p);

  s = MET(:,4)==p;

  plot(MET(s,5),MET(s,6),"bo");

  hold on;

  a = min([MET(s,5);MET(s,6)]);
  b = max([MET(s,5);MET(s,6)]);

  plot([a b],[a b],"r--");

  grid on;

  xlabel("RMS a 200 kHz (A)");
  ylabel("RMS a 25 kHz (A)");

  title(sprintf("Preservacao RMS - Fase %s",nomes_fases{p}));

endfor

exportar(fig,OUT,"G4_E02_3_06_Preservacao_RMS.png");


fig = figure("position",[100 100 1100 850]);

for reg = 1:3

  subplot(3,1,reg);

  B = zeros(rows(bandas),3);

  for b = 1:rows(bandas)

    for p = 1:3

      s = SPEC(:,2)==1 & SPEC(:,3)==reg & ...
          SPEC(:,4)==p & SPEC(:,5)==b;

      B(b,p)=mean(SPEC(s,8));

    endfor

  endfor

  bar(B);

  set(gca,"xtick",1:5, ...
          "xticklabel",{"0-5","5-12.5","12.5-20","20-50","50-100"});

  grid on;

  ylabel("Potencia relativa (%)");
  xlabel("Banda (kHz)");

  title(nomes_regioes{reg});

  legend("A","B","C");

endfor

exportar(fig,OUT,"G4_E02_3_07_Distribuicao_Espectral.png");


fig = figure("position",[100 100 950 550]);

A = zeros(length(SNRs),2);

for j = 1:length(SNRs)

  for m = 1:2

    s = AW(:,5)==SNRs(j) & AW(:,7)==m;

    A(j,m)=mean(AW(s,10));

  endfor

endfor

bar(SNRs,A);

grid on;

xlabel("SNR alvo (dB)");
ylabel("Potencia do ruido acima de 12.5 kHz (%)");

title("Distribuicao espectral do ruido");

legend("AWGN 25 kHz + ZOH","AWGN 200 kHz");

exportar(fig,OUT,"G4_E02_3_08_AWGN_Alta_Frequencia.png");


fig = figure("position",[100 100 950 500]);

f = linspace(0,fs/2,4001)';

% Resposta discreta normalizada do bloco de M amostras.

H = ones(size(f));

s = f != 0;

H(s) = abs(sin(pi*M*f(s)/fs)./ ...
           (M*sin(pi*f(s)/fs)));

plot(f/1000,H,"b","linewidth",1.5);

grid on;

xlabel("Frequencia (kHz)");
ylabel("Magnitude normalizada");

title("Resposta teorica da retencao discreta de oito amostras");

hold on;

line([12.5 12.5],[0 1],"color","r","linestyle","--");

legend("Resposta discreta","Nyquist candidato");

exportar(fig,OUT,"G4_E02_3_09_Resposta_Retencao.png");

% ============================================================
% PROTOCOLO E ENCERRAMENTO
% ============================================================

fprot = fopen(fullfile(OUT,"G4_E02_3_Protocolo.txt"),"w");

fprintf(fprot,"GATE 4 - EXPERIMENTO 02.3\n\n");

fprintf(fprot,"Fs original: %.0f Hz\n",fs);

fprintf(fprot,"Fs candidata: %.0f Hz\n",fs2);

fprintf(fprot,"Fator de retencao: %d\n",M);

fprintf(fprot,"Duracao PSD: %.6f s\n",Tpsd);

fprintf(fprot,"Duracao STFT: %.6f s\n",Tw);

fprintf(fprot,"Salto STFT: %.6f s\n",Th);

fprintf(fprot,"AWGN: 30, 40 e 50 dB\n");

fprintf(fprot,"Repeticoes por condicao: %d\n",Nrep);

fprintf(fprot,"Seed: 42\n");

fprintf(fprot,"Registros reservados: 06 e 12\n");

fprintf(fprot,"Originais: preservados\n");

fprintf(fprot,"Variantes definitivas: zero\n");

fprintf(fprot,"Treinamento de IA: nenhum\n");

fprintf(fprot,["A frequencia de 25 kHz constitui uma " ...
               "hipotese investigada, nao uma conclusao " ...
               "sobre a origem fisica do sinal.\n"]);

fprintf(fprot,["PSD acima de Nyquist candidato pode " ...
               "conter imagens espectrais da retencao.\n"]);

fprintf(fprot,["NaN nas bandas acima de Nyquist de 25 kHz " ...
               "representa grandeza indisponivel.\n"]);

fprintf(fprot,["Os espectrogramas sao figuras investigativas " ...
               "e nao entradas definitivas da CNN.\n"]);

fclose(fprot);

fprintf("\n========================================\n");
fprintf(" G4/E02.3 - EXECUCAO CONCLUIDA\n");
fprintf("========================================\n");

fprintf("Grade temporal: ANALISADA\n");
fprintf("Reconstrucao ZOH: ANALISADA\n");
fprintf("Grandezas eletricas: COMPARADAS\n");
fprintf("PSD: COMPARADA\n");
fprintf("AWGN: COMPARADO\n");
fprintf("Espectrogramas: PRODUZIDOS\n");
fprintf("Registros 06 e 12: PRESERVADOS\n");
fprintf("Arquivos originais: PRESERVADOS\n");
fprintf("Variantes definitivas: ZERO\n");
fprintf("Treinamento de IA: NENHUM\n");

fprintf("\nDiretorio:\n%s\n",OUT);

fprintf("\nAguardando interpretacao cientifica.\n");

fprintf("========================================\n");

diary off;
