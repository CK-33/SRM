
% ============================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 02B - PERIODICIDADE E SINCRONIZACAO TRIFASICA
% GNU OCTAVE 11.3.0
% ============================================================

clear; close all; clc;

% ------------------------------------------------------------
% 1. CONFIGURACAO
% ------------------------------------------------------------

pasta_dados = ['PASTA_DO_USUARIO\Documents\' ...
               'Pesquisa_SRM\TesteNo1_31OCT2026\' ...
               'TesteNo1_31OCT2026\Dados'];

arquivo = fullfile(pasta_dados, 'results_1.mat');
pasta_saida = fullfile(pasta_dados, 'Gate2_Exp02B');

if ~exist(pasta_saida, 'dir')
    mkdir(pasta_saida);
end

% Intervalo de analise
t_inicio = 0.85;
t_fim = 0.98;

% Periodicidade esperada
T_min = 0.010;
T_max = 0.016;

% Atraso maximo para correlacao cruzada
lag_cruzado_max = 0.020;

graphics_toolkit('gnuplot');

% ------------------------------------------------------------
% 2. CARREGAMENTO E VALIDACAO
% ------------------------------------------------------------

dados = load(arquivo);

assert(isfield(dados, 'results'), ...
       'Variavel results nao encontrada.');

r = dados.results;

t = r.time(:);
I = r.signals(3).values;
evento = r.signals(6).values;

assert(size(I,1) == length(t), ...
       'Dimensoes incompatíveis.');

assert(size(I,2) >= 3, ...
       'Tres fases nao encontradas.');

idx_evento = find(evento(:,1) > 0, 1, 'first');

if isempty(idx_evento)
    error('Evento nao encontrado.');
end

t_evento = t(idx_evento);
fs = 1/median(diff(t));

assert(t_fim < t_evento, ...
       'A janela deve ser anterior ao evento.');

idx = find(t >= t_inicio & t <= t_fim);

assert(length(idx) > 100, ...
       'Janela insuficiente.');

tw = t(idx);
X_original = I(idx,1:3);

N = length(tw);
nfft = 2^nextpow2(2*N-1);

nomes = {'A','B','C'};

fprintf('\n====================================\n');
fprintf('GATE 2 - EXPERIMENTO 02B\n');
fprintf('====================================\n');

fprintf('Fs: %.2f Hz\n', fs);
fprintf('Evento: %.6f s\n', t_evento);
fprintf('Janela: %.6f a %.6f s\n', tw(1), tw(end));
fprintf('Amostras: %d\n', N);

% ------------------------------------------------------------
% 3. PRE-PROCESSAMENTO
% ------------------------------------------------------------

X = zeros(size(X_original));

for fase = 1:3
    X(:,fase) = X_original(:,fase) - ...
                mean(X_original(:,fase));
end

% ------------------------------------------------------------
% 4. AUTOCORRELACAO POR FASE
% ------------------------------------------------------------

periodos = NaN(3,1);
frequencias = NaN(3,1);
correlacoes = NaN(3,1);

R_auto = cell(3,1);
lags_auto = (0:N-1)'/fs;

for fase = 1:3

    x = X(:,fase);

    if sum(x.^2) <= eps
        warning('Fase %s sem variacao suficiente.', ...
                nomes{fase});
        continue;
    end

    F = fft(x,nfft);

    R = real(ifft(abs(F).^2));
    R = R(1:N);

    R = R/R(1);

    R_auto{fase} = R;

    % Buscar maximos locais na faixa 10-16 ms
    candidatos = [];

    for k = 2:N-1

        if lags_auto(k) < T_min || ...
           lags_auto(k) > T_max
            continue;
        end

        if R(k) > R(k-1) && R(k) >= R(k+1)
            candidatos(end+1) = k;
        end
    end

    if isempty(candidatos)
        continue;
    end

    [valor, pos] = max(R(candidatos));

    j = candidatos(pos);

    periodos(fase) = lags_auto(j);
    frequencias(fase) = 1/periodos(fase);
    correlacoes(fase) = valor;

end

fprintf('\n--- PERIODICIDADE POR FASE ---\n');

fprintf('Fase | Periodo(ms) | Frequencia(Hz) | R\n');

for fase = 1:3
    fprintf(' %s   | %11.5f | %14.5f | %.5f\n', ...
        nomes{fase}, ...
        periodos(fase)*1000, ...
        frequencias(fase), ...
        correlacoes(fase));
end

% ------------------------------------------------------------
% 5. CORRELACAO CRUZADA
% ------------------------------------------------------------

pares = [1 2; 2 3; 1 3];
nomes_pares = {'A-B','B-C','A-C'};

lag_max = min(round(lag_cruzado_max*fs), N-1);

lags_cruzados = (-lag_max:lag_max)'/fs;

atrasos = NaN(3,1);
corr_cruzadas = NaN(3,1);

R_cross = cell(3,1);

for p = 1:3

    a = pares(p,1);
    b = pares(p,2);

    xa = X(:,a);
    xb = X(:,b);

    % Correlacao circular calculada com padding
    % suficiente para equivaler a correlacao linear.
    Fa = fft(xa,nfft);
    Fb = fft(xb,nfft);

    % Convencao: R_ab(lag) = soma xa(n)*xb(n+lag)
    C = real(ifft(conj(Fa).*Fb));

    % Reconstruir atrasos negativos e positivos
    Rneg = C(end-lag_max+1:end);
    Rpos = C(1:lag_max+1);

    R = [Rneg; Rpos];

    denominador = sqrt(sum(xa.^2)*sum(xb.^2));

    if denominador <= eps
        continue;
    end

    R = R/denominador;

    R_cross{p} = R;

    % Pico de correlacao positivo mais forte
    [valor, j] = max(R);

    atrasos(p) = lags_cruzados(j);
    corr_cruzadas(p) = valor;

end

fprintf('\n--- CORRELACAO CRUZADA ---\n');

fprintf('Par | Atraso(ms) | Correlacao\n');

for p = 1:3

    fprintf('%s | %10.5f | %.6f\n', ...
        nomes_pares{p}, ...
        atrasos(p)*1000, ...
        corr_cruzadas(p));

end

fprintf('\nConvencao: atraso positivo indica que\n');
fprintf('a segunda fase ocorre depois da primeira.\n');

% ------------------------------------------------------------
% 6. FIGURA 1 - CORRENTES TRIFASICAS
% ------------------------------------------------------------

figure(1);

plot(tw, X_original(:,1), 'r', ...
     tw, X_original(:,2), 'b', ...
     tw, X_original(:,3), 'g');

xlabel('Tempo (s)');
ylabel('Corrente (A)');

title('Gate 2 - Correntes trifasicas pre-evento');

legend('Fase A','Fase B','Fase C', ...
       'location','northeast');

grid on;

drawnow();

try
    print(gcf, fullfile(pasta_saida, ...
          'G2_E02B_Fig01_Correntes.png'), ...
          '-dpng','-r150');
catch err
    fprintf('Falha na Figura 1: %s\n',err.message);
end

% ------------------------------------------------------------
% 7. FIGURA 2 - AUTOCORRELACAO DAS FASES
% ------------------------------------------------------------

figure(2);

cores = {'r','b','g'};

hold on;

for fase = 1:3

    if ~isempty(R_auto{fase})

        plot(lags_auto*1000, R_auto{fase}, ...
             cores{fase}, 'LineWidth',1.2);

    end
end

xlim([0 50]);

xlabel('Atraso (ms)');
ylabel('Autocorrelacao normalizada');

title('Gate 2 - Autocorrelacao trifasica');

legend('Fase A','Fase B','Fase C', ...
       'location','northeast');

grid on;

drawnow();

try
    print(gcf, fullfile(pasta_saida, ...
          'G2_E02B_Fig02_Autocorrelacao.png'), ...
          '-dpng','-r150');
catch err
    fprintf('Falha na Figura 2: %s\n',err.message);
end

% ------------------------------------------------------------
% 8. FIGURA 3 - CORRELACOES CRUZADAS
% ------------------------------------------------------------

figure(3);

hold on;

cores_pares = {'r','b','g'};

for p = 1:3

    if ~isempty(R_cross{p})

        plot(lags_cruzados*1000, R_cross{p}, ...
             cores_pares{p}, 'LineWidth',1.2);

    end
end

xlabel('Atraso (ms)');
ylabel('Correlacao cruzada normalizada');

title('Gate 2 - Relacao temporal entre fases');

legend('A-B','B-C','A-C', ...
       'location','northeast');

grid on;

drawnow();

try
    print(gcf, fullfile(pasta_saida, ...
          'G2_E02B_Fig03_Correlacao_Cruzada.png'), ...
          '-dpng','-r150');
catch err
    fprintf('Falha na Figura 3: %s\n',err.message);
end

% ------------------------------------------------------------
% 9. EXPORTAR METRICAS
% ------------------------------------------------------------

fid = fopen(fullfile(pasta_saida, ...
    'G2_E02B_Periodicidade.csv'),'w');

if fid < 0
    error('Falha ao criar CSV.');
end

fprintf(fid, ...
    'fase,periodo_ms,frequencia_hz,correlacao\n');

for fase = 1:3

    fprintf(fid,'%s,%.8f,%.8f,%.8f\n', ...
        nomes{fase}, ...
        periodos(fase)*1000, ...
        frequencias(fase), ...
        correlacoes(fase));
end

fclose(fid);

fid = fopen(fullfile(pasta_saida, ...
    'G2_E02B_Correlacao_Cruzada.csv'),'w');

if fid < 0
    error('Falha ao criar CSV de correlacao cruzada.');
end

fprintf(fid,'par,atraso_ms,correlacao\n');

for p = 1:3

    fprintf(fid,'%s,%.8f,%.8f\n', ...
        nomes_pares{p}, ...
        atrasos(p)*1000, ...
        corr_cruzadas(p));
end

fclose(fid);

fprintf('\nExperimento 02B concluido.\n');
fprintf('Resultados em:\n%s\n',pasta_saida);
