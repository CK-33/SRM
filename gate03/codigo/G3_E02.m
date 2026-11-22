
% ============================================================
% GATE 3 - EXPERIMENTO 02
% Protocolo de validacao por registros
% GNU Octave 11.3.0
% ============================================================

clear; clc; close all;

base = ['PASTA_DO_USUARIO/Documents/Pesquisa_SRM/' ...
        'TesteNo1_31OCT2026/Data and Scripts'];

entrada = fullfile(base, 'Gate2_Exp05', ...
                   'G2_E05_Caracteristicas.csv');

saida = fullfile(base, 'Gate3_Exp02');

if ~exist(saida, 'dir')
    mkdir(saida);
end

if ~exist(entrada, 'file')
    error('CSV de entrada nao encontrado.');
end

% ------------------------------------------------------------
% 1. Ler CSV preservando cabecalhos e textos
% ------------------------------------------------------------

fid = fopen(entrada, 'rt');
if fid < 0
    error('Nao foi possivel abrir o CSV.');
end

cab = fgetl(fid);

if ~ischar(cab)
    fclose(fid);
    error('CSV vazio.');
end

cab = strrep(cab, char([239 187 191]), '');
nomes = strtrim(strsplit(cab, ','));

formato = repmat('%s', 1, numel(nomes));

C = textscan(fid, formato, ...
    'Delimiter', ',', ...
    'ReturnOnError', false);

fclose(fid);

N = numel(C{1});

for j = 1:numel(C)
    if numel(C{j}) ~= N
        error('Colunas com comprimentos diferentes.');
    end
end

% ------------------------------------------------------------
% 2. Metadados obrigatorios
% ------------------------------------------------------------

cr = find(strcmp(nomes, 'registro'));
cy = find(strcmp(nomes, 'regiao'));
cf = find(strcmp(nomes, 'fase'));

if numel(cr) ~= 1 || numel(cy) ~= 1 || numel(cf) ~= 1
    error('Colunas registro, regiao ou fase invalidas.');
end

registro = str2double(C{cr});
regiao = C{cy};
fase = C{cf};

if any(~isfinite(registro)) || ...
   any(registro ~= floor(registro))
    error('Identificadores de registro invalidos.');
end

y = NaN(N,1);

for i = 1:N
    s = lower(strtrim(regiao{i}));

    if strcmp(s, 'pre')
        y(i) = 0;
    elseif strcmp(s, 'pos')
        y(i) = 1;
    end
end

if any(isnan(y))
    error('Rotulos pre/pos nao reconhecidos.');
end

% ------------------------------------------------------------
% 3. Divisao por registro
% ------------------------------------------------------------

% Distribuicao fixa ao longo dos 12 registros.
% Os registros sao ordenados pelo indice original,
% associado aos diferentes instantes de evento.
%
% Esta divisao NAO garante independencia fisica.

train_regs = [1 2 4 5 7 8 10 11];
val_regs   = [3 9];
test_regs  = [6 12];

todos = unique(registro);

esperados = 1:12;

if numel(todos) ~= 12 || ...
   ~isequal(sort(todos(:))', esperados)
    error('Esperados exatamente os registros 1 a 12.');
end

if ~isempty(intersect(train_regs,val_regs)) || ...
   ~isempty(intersect(train_regs,test_regs)) || ...
   ~isempty(intersect(val_regs,test_regs))
    error('Vazamento: registros compartilhados.');
end

grupo = cell(N,1);

for i = 1:N

    r = registro(i);

    if ismember(r, train_regs)
        grupo{i} = 'treino';
    elseif ismember(r, val_regs)
        grupo{i} = 'validacao';
    elseif ismember(r, test_regs)
        grupo{i} = 'teste';
    else
        error('Registro sem grupo.');
    end

end

% ------------------------------------------------------------
% 4. Auditoria de distribuicao
% ------------------------------------------------------------

fprintf('\n====================================\n');
fprintf('G3/E02 - PROTOCOLO DE VALIDACAO\n');
fprintf('====================================\n');

grupos = {'treino','validacao','teste'};

resumo = zeros(3,4);

for g = 1:3

    idx = strcmp(grupo, grupos{g});

    nlinhas = sum(idx);
    npre = sum(idx & y==0);
    npos = sum(idx & y==1);
    nreg = numel(unique(registro(idx)));

    resumo(g,:) = [nreg nlinhas npre npos];

    fprintf('\n%s\n', upper(grupos{g}));
    fprintf('Registros: %d\n', nreg);
    fprintf('Linhas: %d\n', nlinhas);
    fprintf('Pre-evento: %d\n', npre);
    fprintf('Pos-evento: %d\n', npos);

    if npre == 0 || npos == 0
        error('Grupo sem ambas as regioes.');
    end

end

% ------------------------------------------------------------
% 5. Verificar integridade das fases
% ------------------------------------------------------------

fprintf('\n--- INTEGRIDADE POR REGISTRO ---\n');

for r = todos(:)'

    idx = registro == r;

    na = sum(idx & strcmpi(fase,'A'));
    nb = sum(idx & strcmpi(fase,'B'));
    nc = sum(idx & strcmpi(fase,'C'));

    fprintf('Registro %02d: A=%d B=%d C=%d\n', ...
            r, na, nb, nc);

    if na ~= nb || nb ~= nc
        error('Numero desigual de fases no registro %d.',r);
    end

end

% ------------------------------------------------------------
% 6. Verificar caracteristicas disponiveis
% ------------------------------------------------------------

% Excluimos metadados e variaveis temporais.
% Nenhuma transformacao e ajustada nesta etapa.

excluir = { ...
    'registro', 'regiao', 'fase', 'janela', ...
    't_inicio_s', 't_fim_s', 't_evento_s', ...
    'fs_Hz' ...
};

features = {};

for j = 1:numel(nomes)

    if any(strcmp(nomes{j}, excluir))
        continue;
    end

    v = str2double(C{j});

    if all(isfinite(v))
        features{end+1} = nomes{j};
    else
        fprintf('Coluna nao aprovada: %s\n', nomes{j});
    end

end

fprintf('\nCaracteristicas numericas: %d\n', ...
        numel(features));

% ------------------------------------------------------------
% 7. Exportar manifesto das linhas
% ------------------------------------------------------------

% Este manifesto conserva a posicao original no CSV.
% Nao replica caracteristicas nem sinais proprietarios.

manifesto = fullfile(saida, ...
    'G3_E02_Manifesto.csv');

fid = fopen(manifesto,'wt');

fprintf(fid, ...
    'linha_csv,registro,fase,regiao,grupo\n');

for i = 1:N
    fprintf(fid,'%d,%d,%s,%s,%s\n', ...
        i+1, registro(i), fase{i}, ...
        regiao{i}, grupo{i});
end

fclose(fid);

% ------------------------------------------------------------
% 8. Exportar resumo
% ------------------------------------------------------------

fid = fopen(fullfile(saida, ...
    'G3_E02_Resumo.csv'),'wt');

fprintf(fid, ...
    'grupo,n_registros,n_linhas,n_pre,n_pos\n');

for g = 1:3
    fprintf(fid,'%s,%d,%d,%d,%d\n', ...
        grupos{g}, resumo(g,:));
end

fclose(fid);

% ------------------------------------------------------------
% 9. Exportar dicionario
% ------------------------------------------------------------

fid = fopen(fullfile(saida, ...
    'G3_E02_Features.csv'),'wt');

fprintf(fid,'indice,caracteristica\n');

for j = 1:numel(features)
    fprintf(fid,'%d,%s\n',j,features{j});
end

fclose(fid);

% ------------------------------------------------------------
% 10. Registrar configuracao de validacao
% ------------------------------------------------------------

fid = fopen(fullfile(saida, ...
    'G3_E02_Protocolo.txt'),'wt');

fprintf(fid,'G3/E02 - Protocolo de validacao\n\n');
fprintf(fid,'Treino: %s\n',mat2str(train_regs));
fprintf(fid,'Validacao: %s\n',mat2str(val_regs));
fprintf(fid,'Teste: %s\n',mat2str(test_regs));

fprintf(fid,'\nRegras:\n');
fprintf(fid,'1. Nunca dividir aleatoriamente por janela.\n');
fprintf(fid,'2. Manter todas as fases de um registro juntas.\n');
fprintf(fid,'3. Ajustar normalizacao somente no treino.\n');
fprintf(fid,'4. Ajustar selecao de features somente no treino.\n');
fprintf(fid,'5. Nao ajustar modelos usando o teste.\n');
fprintf(fid,'6. Preservar o agrupamento nas augmentations.\n');
fprintf(fid,'7. Rotulos pre/pos sao proxies temporais.\n');
fprintf(fid,'8. Registros podem compartilhar trajetorias.\n');
fprintf(fid,'9. Teste nao comprova generalizacao fisica.\n');
fprintf(fid,'10. LORO requer validacao interna separada.\n');

fclose(fid);

fprintf('\n====================================\n');
fprintf('G3/E02 - PREPARACAO CONCLUIDA\n');
fprintf('====================================\n');

fprintf('Saidas em:\n%s\n',saida);
fprintf('Nenhum classificador foi treinado.\n');
