1;
root = 'CAMINHO_DA_CAMPANHA/Data and Scripts/Gate4_Exp03_Run01/Originais_Canonicos';
ids = {'R01','R02','R04','R05','R07','R08','R10','R11','R03','R09'};
rows = {};
for k = 1:numel(ids)
  id = ids{k};
  split = 'treino';
  if k > 8, split = 'validacao'; endif
  raw = load(fullfile(root, [id '_O00.mat']), 'registro');
  r = raw.registro;
  assert(strcmp(r.id,[id '_O00']) && r.Fs_Hz == 25000 && r.N == 13750);
  assert(isequal(size(r.I_A), [r.N,3]) && all(isfinite(r.I_A(:))));
  event = ceil((r.t_evento_s-r.t0_s)*r.Fs_Hz-1e-8);
  assert(event >= 981 && event < r.N-981);
  for stop = 980:41:r.N-1
    first = stop - [654,981] + 1;
    labels = {'cruzado','cruzado'};
    for j = 1:2
      if stop < event
        labels{j} = 'pre';
      elseif first(j) >= event
        labels{j} = 'pos';
      endif
    endfor
    paired = strcmp(labels{1},labels{2}) && !strcmp(labels{1},'cruzado');
    rows(end+1,:) = {id,split,first(1),first(2),stop,event,r.t0_s+stop/r.Fs_Hz,labels{1},labels{2},paired};
  endfor
endfor
assert(rows{1,4}==0 && rows{1,5}==980);
assert(size(rows,1)==10*numel(980:41:13749));
folder = fileparts(mfilename('fullpath'));
[parent,name] = fileparts(folder);
if strcmp(name,'codigo'), folder = fullfile(parent,'dados','manifestos'); endif
target = fullfile(folder,'G5_E03_Manifesto_Canonicos.csv');
fid = fopen(target,'w');
assert(fid>=0);
fprintf(fid,'origem,split,inicio_654,inicio_981,fim,evento,decisao_s,rotulo_654,rotulo_981,par_puro\n');
for k = 1:size(rows,1)
  fprintf(fid,'%s,%s,%d,%d,%d,%d,%.8f,%s,%s,%d\n',rows{k,:});
endfor
assert(fclose(fid)==0);
