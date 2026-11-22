1;
root = 'CAMINHO_DA_CAMPANHA/Data and Scripts/Gate4_Exp03_Run01/Originais_Canonicos';
rows = struct([]);
for id = {'R01', 'R04'}
  s = load(fullfile(root, [id{1}, '_O00.mat']), 'registro');
  r = s.registro;
  assert(strcmp(r.id, [id{1}, '_O00']) && r.Fs_Hz == 25000);
  assert(isequal(size(r.I_A), [r.N, 3]) && all(isfinite(r.I_A(:))));
  % Primeiro instante na grade canônica em/apos o marcador.
  e = ceil((r.t_evento_s - r.t0_s) * r.Fs_Hz - 1e-8) + 1;
  for setup = [654, 981; 164, 327]
    n = setup(1); L = setup(2);
    w = 0.5 - 0.5*cos(2*pi*(0:L-1)'/(L-1));
    starts = 0:41:(n-L);
    for pos = [0.25, 0.50, 0.75]
      offset = round(pos*n);
      % Mesmo arredondamento do protótipo Python: empate vai para o par.
      if pos*n-floor(pos*n) == 0.5
        offset = 2*round((pos*n-0.5)/2);
      endif
      first = e-offset;
      assert(first >= 1 && first+n-1 <= r.N);
      pre = starts+L-1 < offset;
      post = starts >= offset;
      peaks = zeros(numel(starts), 3);
      for j = 1:numel(starts)
        a = first+starts(j);
        F = abs(fft(r.I_A(a:a+L-1,:) .* w, 1024, 1))/sum(w);
        F(2:512,:) *= 2;
        peaks(j,:) = max(F(1:513,:),[],1);
      endfor
      p0 = []; p1 = [];
      if any(pre), p0 = median(peaks(pre,:),1); endif
      if any(post), p1 = median(peaks(post,:),1); endif
      rows(end+1) = struct('origem',id{1},'segmento',n,'janela',L,'posicao',pos, ...
                           'pre',sum(pre),'cruzados',sum(!(pre|post)),'pos',sum(post), ...
                           'pico_mediano_pre_ABC_A',p0,'pico_mediano_pos_ABC_A',p1);
    endfor
  endfor
endfor
assert(numel(rows)==12);
folder = fileparts(mfilename('fullpath'));
[parent, name] = fileparts(folder);
if strcmp(name, 'codigo')
  folder = fullfile(parent, 'resultados', 'metricas');
endif
target = fullfile(folder, 'G5_E02_R01_R04.json');
fid = fopen(target, 'w');
assert(fid >= 0);
fprintf(fid, '%s\n', jsonencode(struct('fs_Hz',25000,'nfft',1024,'salto',41,'resultados',rows)));
assert(fclose(fid)==0);
