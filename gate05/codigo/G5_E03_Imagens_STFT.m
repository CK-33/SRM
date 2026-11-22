function G5_E03_Imagens_STFT(gate, teste = false)
  palette = uint8(round(255 * viridis(256)));
  if teste
    x = reshape(single(linspace(0,100,3*12*513)),3,12,513);
    rgb = painel(x,palette);
    path = fullfile(gate,'G5_E03_Imagem_Exemplo_Sintetico.png');
    imwrite(rgb,path);
    assert(isequal(size(rgb),[256,462,3]) && isequal(imread(path),rgb));
    return;
  endif
  ids = {'R01','R02','R04','R05','R07','R08','R10','R11','R03','R09'};
  output = fullfile(gate,'resultados','figuras','E03_STFT_25kHz');
  for k = 1:numel(ids)
    split = 'treino';
    if k > 8, split = 'validacao'; endif
    id = ids{k};
    source = fullfile(gate,'dados','derivados','25kHz',split,[id '_STFT_v1.mat']);
    s = load(source,'x164','x327','stop');
    assert(isequal(size(s.x164),[3,12,513,312]));
    assert(isequal(size(s.x327),[3,16,513,312]));
    assert(isequal(double(s.stop(:)),(980:41:13749)'));
    exportar(s.x164,s.stop,fullfile(output,split,id,'2_ciclos'),id,palette);
    exportar(s.x327,s.stop,fullfile(output,split,id,'3_ciclos'),id,palette);
    fprintf('%s %s: 624 imagens disponíveis.\n',id,split);
  endfor
  fprintf('OK: 6240 imagens disponíveis em %s\n',output);
endfunction

function exportar(x,stop,folder,id,palette)
  if !isfolder(folder), [ok,msg] = mkdir(folder); assert(ok,msg); endif
  for j = 1:numel(stop)
    path = fullfile(folder,sprintf('%s_fim_%05d.png',id,stop(j)));
    if exist(path,'file')
      info = imfinfo(path);
      assert(info.Width==462 && info.Height==256,['PNG existente incompatível: ' path]);
      continue;
    endif
    tmp = [path '.parcial'];
    imwrite(painel(x(:,:,:,j),palette),tmp,'png');
    [ok,msg] = movefile(tmp,path); assert(ok,msg);
  endfor
endfunction

function rgb = painel(x,palette)
  times = round(linspace(1,size(x,2),144));
  bins = round(linspace(513,1,256));
  rgb = repmat(uint8(245),256,462,3);
  for phase = 1:3
    matrix = reshape(x(phase,:,:),size(x,2),513).';
    db = 20*log10(max(matrix(bins,times),1e-3));
    lut = 1 + round(255*min(max((db+60)/100,0),1));
    columns = (phase-1)*150 + (1:144);
    rgb(:,columns,:) = reshape(palette(lut(:),:),256,144,3);
  endfor
  rgb(:,451:462,:) = repmat(reshape(palette(256:-1:1,:),256,1,3),1,12,1);
endfunction
