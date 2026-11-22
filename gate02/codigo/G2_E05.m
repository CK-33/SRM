
% ================================================================
% PESQUISA SRM - GATE 2
% EXPERIMENTO 05
% EXTRACAO MULTIRREGISTRO DE CARACTERISTICAS
%
% GNU OCTAVE 11.3.0
%
% ENTRADA:
%   results_1.mat ... results_12.mat
%
% SAIDAS:
%   G2_E05_Caracteristicas.csv
%   G2_E05_Resumo_Registros.csv
%   G2_E05_Resumo_Fases.csv
%   G2_E05_Fig01_RMS.png
%   G2_E05_Fig02_Pico.png
%   G2_E05_Fig03_Espectro.png
%   G2_E05_Fig04_Desequilibrio.png
%   G2_E05_Fig05_Contagem.png
%
% IMPORTANTE:
%   pre_evento e pos_evento sao regioes temporais,
%   nao rotulos comprovados de saude ou falha.
% ================================================================

clear; close all; clc;

graphics_toolkit('gnuplot');
set(0,'defaultfigurevisible','off');

% ================================================================
% 1. CONFIGURACAO
% ================================================================

pasta_dados = ['PASTA_DO_USUARIO\Documents\' ...
               'Pesquisa_SRM\TesteNo1_31OCT2026\' ...
               'TesteNo1_31OCT2026\Dados'];

pasta_saida = fullfile(pasta_dados,'Gate2_Exp05');

if ~exist(pasta_saida,'dir')
    mkdir(pasta_saida);
end

num_arquivos = 12;

T_janela = 0.01308;
T_passo = 0.01308;

margem_evento = 0.010;

% Evitar janelas anteriores ao inicio da atividade
limiar_atividade_A = 1.0;

% Bandas espectrais, em Hz
bandas = [0 5000;
          5000 20000;
          20000 50000;
          50000 100000];

nomes_fases = {'A','B','C'};

% ================================================================
% 2. ARQUIVOS DE SAIDA
% ================================================================

arquivo_csv = fullfile( ...
    pasta_saida,'G2_E05_Caracteristicas.csv');

arquivo_resumo = fullfile( ...
    pasta_saida,'G2_E05_Resumo_Registros.csv');

arquivo_fases = fullfile( ...
    pasta_saida,'G2_E05_Resumo_Fases.csv');

fid = fopen(arquivo_csv,'w');

if fid < 0
    error('Nao foi possivel criar o CSV principal.');
end

fprintf(fid, ...
 ['registro,regiao,janela,t_inicio_s,t_fim_s,' ...
  't_evento_s,fs_Hz,' ...
  'fase,rms_A,media_A,desvio_A,' ...
  'pico_abs_A,pico_a_pico_A,' ...
  'fator_crista,mae_derivada_A_s,' ...
  'rms_derivada_A_s,' ...
  'energia_A2_s,' ...
  'potencia_0_5k,' ...
  'potencia_5_20k,' ...
  'potencia_20_50k,' ...
  'potencia_50_100k,' ...
  'centroide_Hz,' ...
  'rms_trifasico_A,' ...
  'desequilibrio_rms_pct\n']);

fid_resumo = fopen(arquivo_resumo,'w');

if fid_resumo < 0
    fclose(fid);
    error('Nao foi possivel criar o resumo.');
end

fprintf(fid_resumo, ...
 ['registro,status,t_inicio_registro_s,' ...
  't_fim_registro_s,t_evento_s,' ...
  'fs_Hz,amostras,' ...
  'janelas_pre,janelas_pos\n']);

fid_fases = fopen(arquivo_fases,'w');

if fid_fases < 0
    fclose(fid);
    fclose(fid_resumo);
    error('Nao foi possivel criar o resumo de fases.');
end

fprintf(fid_fases, ...
 ['registro,regiao,fase,n_janelas,' ...
  'rms_medio_A,rms_dp_A,' ...
  'pico_medio_A,centroide_medio_Hz,' ...
  'desequilibrio_medio_pct\n']);

% ================================================================
% 3. ACUMULADORES
% ================================================================

% Dimensoes: registro x regiao
contagem = zeros(num_arquivos,2);

% Dimensoes: registro x regiao x fase
rms_medio = NaN(num_arquivos,2,3);
pico_medio = NaN(num_arquivos,2,3);
centroide_medio = NaN(num_arquivos,2,3);

% Dimensoes: registro x regiao
deseq_medio = NaN(num_arquivos,2);

% ================================================================
% 4. PROCESSAMENTO DOS 12 REGISTROS
% ================================================================

fprintf('\n========================================\n');
fprintf('GATE 2 - EXPERIMENTO 05\n');
fprintf('EXTRACAO MULTIRREGISTRO\n');
fprintf('========================================\n');

for registro = 1:num_arquivos

    nome_arquivo = sprintf('results_%d.mat',registro);
    caminho = fullfile(pasta_dados,nome_arquivo);

    fprintf('\n----------------------------------------\n');
    fprintf('Registro %d/12\n',registro);

    if ~exist(caminho,'file')

        fprintf('Arquivo nao encontrado.\n');

        fprintf(fid_resumo, ...
            '%d,arquivo_ausente,NaN,NaN,NaN,NaN,0,0,0\n', ...
            registro);

        continue;
    end

    try

        % --------------------------------------------------------
        % 4.1 CARREGAMENTO
        % --------------------------------------------------------

        dados = load(caminho);
        r = dados.results;

        t = r.time(:);
        I = r.signals(3).values;
        evento = r.signals(6).values;

        if size(I,1) ~= length(t)
            error('Dimensoes de tempo e corrente diferentes.');
        end

        if size(I,2) < 3
            error('Menos de tres fases disponiveis.');
        end

        I = I(:,1:3);

        dt = median(diff(t));
        fs = 1/dt;

        if any(diff(t)<=0)
            error('Vetor de tempo nao crescente.');
        end

        % --------------------------------------------------------
        % 4.2 EVENTO
        % --------------------------------------------------------

        ev = evento(:,1);

        if length(ev) ~= length(t)
            error('Marcador de evento incompatível.');
        end

        idx_evento = find(ev>0,1,'first');

        if isempty(idx_evento)
            error('Marcador de evento nao encontrado.');
        end

        t_evento = t(idx_evento);

        % --------------------------------------------------------
        % 4.3 REGIAO ATIVA
        % --------------------------------------------------------

        idx_ativo = find( ...
            any(abs(I)>limiar_atividade_A,2) & ...
            t<t_evento-margem_evento, ...
            1,'first');

        if isempty(idx_ativo)
            error('Inicio ativo nao identificado.');
        end

        t_ativo = t(idx_ativo);

        N_janela = round(T_janela*fs);
        N_passo = round(T_passo*fs);

        if N_janela < 16 || N_passo < 1
            error('Janela ou passo invalidos.');
        end

        fprintf('Fs: %.2f Hz\n',fs);
        fprintf('Evento: %.6f s\n',t_evento);
        fprintf('Inicio ativo: %.6f s\n',t_ativo);
        fprintf('Amostras: %d\n',length(t));

        % --------------------------------------------------------
        % 4.4 DUAS REGIOES TEMPORAIS
        % --------------------------------------------------------

        % PRE: inicio ativo ate antes do evento
        % POS: apos o evento ate o final do registro

        limites = [t_ativo, ...
                   t_evento-margem_evento;
                   t_evento+margem_evento, ...
                   t(end)];

        for regiao = 1:2

            if regiao == 1
                nome_regiao = 'pre';
            else
                nome_regiao = 'pos';
            end

            t0 = limites(regiao,1);
            t1 = limites(regiao,2);

            idx_regiao = find(t>=t0 & t<=t1);

            if length(idx_regiao) < N_janela

                fprintf('%s: janela insuficiente.\n', ...
                    nome_regiao);

                continue;
            end

            inicio_idx = idx_regiao(1);
            ultimo_idx = idx_regiao(end);

            indices_inicio = ...
                inicio_idx:N_passo:(ultimo_idx-N_janela+1);

            K = length(indices_inicio);

            contagem(registro,regiao) = K;

            fprintf('%s: %d janelas.\n', ...
                nome_regiao,K);

            % Acumuladores por janela
            RMSmat = NaN(K,3);
            PICmat = NaN(K,3);
            CENTmat = NaN(K,3);
            DESmat = NaN(K,1);

            % ----------------------------------------------------
            % 4.5 EXTRAIR CARACTERISTICAS
            % ----------------------------------------------------

            for j = 1:K

                a = indices_inicio(j);
                b = a+N_janela-1;

                x = I(a:b,:);

                % Indicadores trifasicos
                rms3 = sqrt(mean(x.^2,1));

                rms_trifasico = sqrt(mean(rms3.^2));

                media_rms = mean(rms3);

                if media_rms > eps

                    desequilibrio = ...
                        100*std(rms3,0,2)/media_rms;

                else

                    desequilibrio = NaN;

                end

                DESmat(j) = desequilibrio;

                % ------------------------------------------------
                % 4.6 CARACTERISTICAS POR FASE
                % ------------------------------------------------

                for fase = 1:3

                    y = x(:,fase);

                    media_y = mean(y);
                    rms_y = sqrt(mean(y.^2));
                    dp_y = std(y);

                    pico_y = max(abs(y));

                    pico_a_pico = max(y)-min(y);

                    if rms_y > eps
                        fator_crista = pico_y/rms_y;
                    else
                        fator_crista = NaN;
                    end

                    dy = diff(y)*fs;

                    mae_derivada = mean(abs(dy));
                    rms_derivada = sqrt(mean(dy.^2));

                    energia = sum(y.^2)/fs;

                    % --------------------------------------------
                    % ESPECTRO UNILATERAL
                    % --------------------------------------------

                    % Remover media para analise AC
                    yc = y-mean(y);

                    N = length(yc);

                    Y = fft(yc);

                    Nf = floor(N/2)+1;

                    P = abs(Y(1:Nf)).^2/N^2;

                    % Converter energia espectral bilateral
                    % para espectro unilateral.
                    if rem(N,2)==0

                        if Nf>2
                            P(2:end-1) = 2*P(2:end-1);
                        end

                    else

                        if Nf>1
                            P(2:end) = 2*P(2:end);
                        end

                    end

                    f = (0:Nf-1)'*fs/N;

                    P = P(:);

                    % Potencia media por banda (A^2)
                    pot_bandas = zeros(1,4);

                    for bb = 1:4

                        fmin = bandas(bb,1);
                        fmax = min(bandas(bb,2),fs/2);

                        if bb < 4
                            mascara = f>=fmin & f<fmax;
                        else
                            mascara = f>=fmin & f<=fmax;
                        end

                        pot_bandas(bb) = sum(P(mascara));

                    end

                    potencia_total = sum(P);

                    if potencia_total > eps

                        centroide = sum(f.*P)/potencia_total;

                    else

                        centroide = NaN;

                    end

                    % --------------------------------------------
                    % GUARDAR RESULTADOS
                    % --------------------------------------------

                    RMSmat(j,fase) = rms_y;
                    PICmat(j,fase) = pico_y;
                    CENTmat(j,fase) = centroide;

                    fprintf(fid, ...
                        ['%d,%s,%d,%.9f,%.9f,' ...
                         '%.9f,%.3f,' ...
                         '%s,%.8f,%.8f,%.8f,' ...
                         '%.8f,%.8f,' ...
                         '%.8f,%.8f,' ...
                         '%.8f,%.8f,' ...
                         '%.10f,' ...
                         '%.10f,%.10f,%.10f,%.10f,' ...
                         '%.8f,%.8f,%.8f\n'], ...
                        registro,nome_regiao,j, ...
                        t(a),t(b), ...
                        t_evento,fs, ...
                        nomes_fases{fase}, ...
                        rms_y,media_y,dp_y, ...
                        pico_y,pico_a_pico, ...
                        fator_crista,mae_derivada, ...
                        rms_derivada,energia, ...
                        pot_bandas(1),pot_bandas(2), ...
                        pot_bandas(3),pot_bandas(4), ...
                        centroide,rms_trifasico, ...
                        desequilibrio);

                end

            end

            % ----------------------------------------------------
            % 4.7 RESUMOS POR FASE
            % ----------------------------------------------------

            for fase = 1:3

                rms_medio(registro,regiao,fase) = ...
                    mean(RMSmat(:,fase));

                pico_medio(registro,regiao,fase) = ...
                    mean(PICmat(:,fase));

                centroide_medio(registro,regiao,fase) = ...
                    mean(CENTmat(:,fase));

                fprintf(fid_fases, ...
                    '%d,%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f\n', ...
                    registro,nome_regiao, ...
                    nomes_fases{fase},K, ...
                    mean(RMSmat(:,fase)), ...
                    std(RMSmat(:,fase)), ...
                    mean(PICmat(:,fase)), ...
                    mean(CENTmat(:,fase)), ...
                    mean(DESmat));

            end

            deseq_medio(registro,regiao) = mean(DESmat);

        end

        % --------------------------------------------------------
        % 4.8 RESUMO DO REGISTRO
        % --------------------------------------------------------

        fprintf(fid_resumo, ...
            '%d,ok,%.9f,%.9f,%.9f,%.3f,%d,%d,%d\n', ...
            registro,t(1),t(end),t_evento,fs, ...
            length(t), ...
            contagem(registro,1), ...
            contagem(registro,2));

    catch err

        fprintf('ERRO no registro %d: %s\n', ...
            registro,err.message);

        fprintf(fid_resumo, ...
            '%d,erro,NaN,NaN,NaN,NaN,0,%d,%d\n', ...
            registro, ...
            contagem(registro,1), ...
            contagem(registro,2));

    end

end

fclose(fid);
fclose(fid_resumo);
fclose(fid_fases);

% ================================================================
% 5. GRAFICOS DE COMPARACAO
% ================================================================

registros = 1:num_arquivos;

% ---------------------------------------------------------------
% FIGURA 1 - RMS MEDIO POR REGISTRO
% ---------------------------------------------------------------

figure(1,'position',[100 100 1200 750]);

for fase = 1:3

    subplot(3,1,fase);

    pre = squeeze(rms_medio(:,1,fase));
    pos = squeeze(rms_medio(:,2,fase));

    plot(registros,pre,'bo-', ...
        'linewidth',1.3);

    hold on;

    plot(registros,pos,'rs-', ...
        'linewidth',1.3);

    ylabel('RMS (A)');
    title(['Fase ' nomes_fases{fase}]);

    xlim([1 num_arquivos]);
    grid on;

    if fase==1
        legend('Pre-evento','Pos-evento', ...
            'location','northeast');
    end

    if fase==3
        xlabel('Registro');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E05_Fig01_RMS.png'), ...
    '-dpng','-r250');

% ---------------------------------------------------------------
% FIGURA 2 - PICO MEDIO POR REGISTRO
% ---------------------------------------------------------------

figure(2,'position',[100 100 1200 750]);

for fase = 1:3

    subplot(3,1,fase);

    pre = squeeze(pico_medio(:,1,fase));
    pos = squeeze(pico_medio(:,2,fase));

    plot(registros,pre,'bo-', ...
        'linewidth',1.3);

    hold on;

    plot(registros,pos,'rs-', ...
        'linewidth',1.3);

    ylabel('Pico (A)');
    title(['Fase ' nomes_fases{fase}]);

    xlim([1 num_arquivos]);
    grid on;

    if fase==1
        legend('Pre-evento','Pos-evento', ...
            'location','northeast');
    end

    if fase==3
        xlabel('Registro');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E05_Fig02_Pico.png'), ...
    '-dpng','-r250');

% ---------------------------------------------------------------
% FIGURA 3 - CENTROIDE ESPECTRAL
% ---------------------------------------------------------------

figure(3,'position',[100 100 1200 750]);

for fase = 1:3

    subplot(3,1,fase);

    pre = squeeze(centroide_medio(:,1,fase));
    pos = squeeze(centroide_medio(:,2,fase));

    plot(registros,pre,'bo-', ...
        'linewidth',1.3);

    hold on;

    plot(registros,pos,'rs-', ...
        'linewidth',1.3);

    ylabel('Centroide (Hz)');
    title(['Fase ' nomes_fases{fase}]);

    xlim([1 num_arquivos]);
    grid on;

    if fase==1
        legend('Pre-evento','Pos-evento', ...
            'location','northeast');
    end

    if fase==3
        xlabel('Registro');
    end

end

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E05_Fig03_Espectro.png'), ...
    '-dpng','-r250');

% ---------------------------------------------------------------
% FIGURA 4 - DESEQUILIBRIO TRIFASICO
% ---------------------------------------------------------------

figure(4,'position',[100 100 1100 550]);

plot(registros,deseq_medio(:,1), ...
    'bo-','linewidth',1.4);

hold on;

plot(registros,deseq_medio(:,2), ...
    'rs-','linewidth',1.4);

xlabel('Registro');
ylabel('Desequilibrio RMS (%)');

title('Gate 2 - Desequilibrio trifasico');

legend('Pre-evento','Pos-evento', ...
    'location','northeast');

grid on;
xlim([1 num_arquivos]);

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E05_Fig04_Desequilibrio.png'), ...
    '-dpng','-r250');

% ---------------------------------------------------------------
% FIGURA 5 - QUANTIDADE DE JANELAS
% ---------------------------------------------------------------

figure(5,'position',[100 100 1100 550]);

bar(registros,contagem);

xlabel('Registro');
ylabel('Numero de janelas');

title('Gate 2 - Janelas extraidas');

legend('Pre-evento','Pos-evento', ...
    'location','northeast');

grid on;

set(gcf,'paperpositionmode','auto');
drawnow();

print(gcf,fullfile(pasta_saida, ...
    'G2_E05_Fig05_Contagem.png'), ...
    '-dpng','-r250');

% ================================================================
% 6. RESUMO FINAL
% ================================================================

fprintf('\n========================================\n');
fprintf('RESUMO FINAL - G2/E05\n');
fprintf('========================================\n');

fprintf('\nRegistro | Pre | Pos\n');

for registro = 1:num_arquivos

    fprintf('%8d | %3d | %3d\n', ...
        registro, ...
        contagem(registro,1), ...
        contagem(registro,2));

end

fprintf('\nTotal de janelas pre: %d\n', ...
    sum(contagem(:,1)));

fprintf('Total de janelas pos: %d\n', ...
    sum(contagem(:,2)));

fprintf('Total de observacoes fase-janela: %d\n', ...
    3*sum(contagem(:)));

fprintf('\nArquivos em:\n%s\n',pasta_saida);

fprintf('\nG2/E05 FINALIZADO.\n');
fprintf('Aguardando avaliacao dos resultados.\n');
