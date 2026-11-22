function G5_E03_Exportar_Tensores(modo = 'verificar')
  assert(any(strcmp(modo, {'verificar','exportar'})));
  campanha = 'CAMINHO_DA_CAMPANHA';
  gate = fullfile(campanha,'Gate5_CNN');
  root = fullfile(campanha,'Data and Scripts','Gate4_Exp03_Run01','Originais_Canonicos');
  if strcmp(modo,'exportar')
    manifesto = fullfile(gate,'dados','manifestos','G5_E03_Manifesto_Canonicos.csv');
  else
    manifesto = fullfile(fileparts(mfilename('fullpath')),'G5_E03_Manifesto_Canonicos.csv');
  endif
  fid = fopen(manifesto,'r'); assert(fid>=0);
  unwind_protect
    m = textscan(fid,'%s%s%d%d%d%d%f%s%s%d','Delimiter',',','HeaderLines',1);
    assert(feof(fid));
  unwind_protect_cleanup
    fclose(fid);
  end_unwind_protect
  ids = {'R01','R02','R04','R05','R07','R08','R10','R11','R03','R09'};
  assert(numel(m{1})==3120 && all(ismember(m{1},ids)));
  x = sin((0:1399)' * 2*pi*1000/25000) * [1,2,0.5];
  y = x; y(982:end,:) = 3*y(982:end,:);
  assert(isequal(tensor(x,980,981,327),tensor(y,980,981,327)));
  for k = 1:numel(ids)
    id = ids{k}; split = 'treino';
    if k>8, split = 'validacao'; endif
    rows = find(strcmp(m{1},id));
    assert(numel(rows)==312 && all(strcmp(m{2}(rows),split)));
    s = load(fullfile(root,[id '_O00.mat']),'registro'); r=s.registro;
    assert(strcmp(r.id,[id '_O00']) && r.Fs_Hz==25000 && r.N==13750);
    assert(isequal(size(r.I_A),[r.N,3]) && all(isfinite(r.I_A(:))));
    stop = double(m{5}(rows));
    event = ceil((r.t_evento_s-r.t0_s)*25000-1e-8);
    assert(isequal(stop,(980:41:r.N-1)') && all(m{6}(rows)==event));
    assert(all(m{3}(rows)==stop-653) && all(m{4}(rows)==stop-980));
    x164 = tensor(r.I_A,stop,654,164);
    x327 = tensor(r.I_A,stop,981,327);
    assert(isequal(size(x164),[3,12,513,312]) && isequal(size(x327),[3,16,513,312]));
    assert(all(isfinite(x164(:))) && all(isfinite(x327(:))));
    if strcmp(modo,'exportar')
      folder = fullfile(gate,'dados','derivados','25kHz',split);
      assert(isfolder(folder));
      path = fullfile(folder,[id '_STFT_v1.mat']);
      assert(!exist(path,'file'), ['Destino já existe: ' path]);
      tmp = [tempname(folder), '.mat'];
      unwind_protect
        save('-mat7-binary',tmp,'x164','x327','stop');
        saved = whos('-file',tmp);
        assert(isequal(saved(strcmp({saved.name},'x164')).size,size(x164)));
        assert(isequal(saved(strcmp({saved.name},'x327')).size,size(x327)));
        assert(!exist(path,'file'));
        [ok,msg] = movefile(tmp,path);
        assert(ok,msg);
      unwind_protect_cleanup
        if exist(tmp,'file'), unlink(tmp); endif
      end_unwind_protect
    endif
    fprintf('%s %s: %d instantes, formas 3x12x513 e 3x16x513\n',id,split,numel(stop));
  endfor
  fprintf('OK: 3120 instantes verificados, modo %s.\n',modo);
endfunction

function out = tensor(signal, ends, n, L)
  % Iniciar pelo quadro mais antigo e terminar exatamente na decisão.
  offsets = 41*(0:floor((n-L)/41));
  offsets = fliplr(offsets);
  w = 0.5-0.5*cos(2*pi*(0:L-1)'/(L-1));
  out = zeros(3,numel(offsets),513,numel(ends),'single');
  for i=1:numel(ends)
    for j=1:numel(offsets)
      last = ends(i) + 1 - offsets(j);
      F = abs(fft(signal(last-L+1:last,:) .* w,1024,1))/sum(w);
      F(2:512,:) *= 2;
      out(:,j,:,i) = reshape(F(1:513,:).',[3,1,513]);
    endfor
  endfor
endfunction
