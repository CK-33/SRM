# SRM Lab — Gate 6

Aplicação Windows para análise experimental das correntes trifásicas de motores de relutância chaveada. O pacote de distribuição inclui os checkpoints da CNN 1D, da ResNet-18 STFT e do híbrido ABC, além do backbone ImageNet que eles usam. Os sinais de demonstração são sintéticos.

## Abrir e experimentar

1. No pacote de distribuição, extraia o ZIP por inteiro e abra **SRM_Lab.exe**. No checkout do GitHub, execute `python app.py` dentro de `gate06` com as dependências instaladas. O executável é um lançador Windows que inicia o aplicativo local com Python. Neste computador ele reaproveita o ambiente de CPU do Gate 5. O arquivo **Abrir_SRM_Lab.cmd** mostra erros no console se for preciso investigar a inicialização.
2. Clique em **Abrir demonstração** ou importe os exemplos disponíveis em **demonstracao**. Confirme as associações de tempo, A, B e C. O exemplo muda a amplitude da fase B perto de 0,37 s.
3. Em **Análise de sinais**, selecione correntes, FFT ou espectrograma. Use a roda do mouse e arraste para mudar o intervalo temporal de entrada. Também é possível digitar o intervalo. Os três canais aparecem juntos na curva e no espectro. O espectrograma é de B.
4. Em **Diagnóstico**, use **Analisar arquivo**. Para testar a chegada progressiva das amostras, use **Iniciar / retomar**, velocidade, pausa, interrupção e reinício. **Arquivo inteiro** retorna ao modo offline. A barra de progresso acompanha a liberação das amostras. O histórico e as decisões do replay são produzidos à medida que as amostras chegam.
5. Para usar IA, escolha um modelo e clique em **Ativar modelo**. Em arquivo, **Inferir agora** avalia a janela mais recente disponível. Durante o replay, a aplicação emite escores auxiliares em cadência reduzida. O escore indica a classe temporal pós-evento das simulações. Ele pode divergir do detector, que usa outra referência. A interface sinaliza divergência e qualidade insuficiente.
6. Em **Relatórios**, exporte PDF, CSV e PNG. O PDF inclui identificação da máquina, intervalo, associação, parâmetros, estado, eventos, modelo quando selecionado e gráficos já produzidos pela sessão. CSV retorna histórico de janelas quando disponível e estatísticas por fase nos demais casos.

## Ambiente offline

A aplicação usa Python 3.13 com Tcl/Tk, NumPy, SciPy e Pillow. PyTorch e torchvision são opcionais. Em outro Windows com dependências disponíveis, instale em um ambiente virtual na pasta da aplicação. Para uma instalação sem rede, use um diretório local de wheels compatíveis.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links CAMINHO_DOS_WHEELS -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-index --find-links CAMINHO_DOS_WHEELS -r requirements-ia.txt
```

O lançador procura a variável SRM_LAB_PYTHON, o ambiente .venv e, por fim, o ambiente do Gate 5 deste computador. É possível indicar outra pasta de modelos com o argumento --modelos ao chamar app.py. O aplicativo não baixa pesos. O diagnóstico determinístico opera sem IA.

## Métodos, unidades e limites

- CSV aceita tempo em segundos e três colunas numéricas, com associação manual. MAT clássico aceita vetores, colunas de matrizes, a estrutura results e o esquema canônico registro. Timestamps decrescentes são rejeitados. Para o formato histórico SRM de 200 kHz, a opção de preparação vem marcada no diálogo. O aplicativo conserva a grade original e, em memória, aceita candidatos a até 5 ns da grade nominal, usando a última amostra nas colisões. Com lacunas, a grade regular não é habilitada. A grade canônica de 25 kHz só é oferecida se blocos de oito reconstruírem exatamente a grade de 200 kHz. O seletor ao lado do nome do arquivo permite alternar a representação. A unidade de corrente em ampères deve ser confirmada para diagnóstico.
- FFT usa magnitude unilateral Hann após remover a média. STFT para inspeção usa magnitude em dB com escala visual ajustada ao intervalo. A IA usa outro contrato com 25 kHz, 654 amostras, STFT 164/41/1024, escala fixa, normalização ImageNet e padronização recuperada dos checkpoints. A CNN 1D usa a média e o desvio de corrente do próprio checkpoint.
- O detector da aplicação usa RMS B de 26,16 ms e cadência nominal de 1,64 ms. A referência é calculada nos primeiros 125 ms do intervalo ou replay, assumidos estáveis. Usa mediana, escala robusta, CUSUM/EWMA e persistência de três decisões. É um detector novo, inspirado nos Gates anteriores. Ele não herda as métricas publicadas no E12.
- RMS, média, desvio, pico, crista e integral quadrática são computados no núcleo. A integral quadrática em A²s não é energia em joules sem resistência. A dispersão RMS não é um índice normativo de desequilíbrio.
- Diagnóstico rejeita dados não finitos, unidades não confirmadas, irregularidade excessiva, canal imóvel e possível saturação. Lacunas e condições adversas podem escapar às heurísticas. A interface devolve indicação de aquisição inválida ou indeterminado quando aplicável.
- Replay separa amostra liberada, relógio simulado e custo de processamento. A IA usa apenas a janela já disponível, em tarefa de fundo e sem fila de chamadas neurais acumuladas. A cadência de tela e a frequência de inferência são inferiores à cadência determinística. A plataforma não oferece garantias de tempo real.
- O bloqueio por nome protege os nomes convencionais R06/R12 e results_6/results_12. Cabe ao operador manter o teste reservado, inclusive quando um arquivo tiver outro nome. Os testes de desenvolvimento usam apenas sinais sintéticos e os pesos salvos.
- Uma pontuação alta denota evidência para o proxy pós-evento dentro da pesquisa. O Gate 5 encontrou separação pelo relógio absoluto e validou principalmente duas origens. O aplicativo não afirma mecanismo físico de falha, gravidade, capacidade industrial ou proteção certificada.

## Verificação de entrega

| Recurso | Estado |
|---|---|
| Aplicação Windows, cinco áreas e associação pela interface | Implementado e testado com CSV sintético |
| MAT padrão e dois esquemas SRM | Implementado e testado com MAT sintéticos no núcleo |
| FFT, STFT, RMS, autocorrelação, correlação cruzada, periodicidade e resíduo periódico | Implementado. FFT/STFT e periodicidade verificados com sinais sintéticos. Comparações exploratórias do resíduo pela interface |
| Diagnóstico determinístico, causas de abstenção e replay causal | Testado por alteração do futuro, execução em lotes e canal preso. Pausa/retomada, navegação e atualizações testadas pela interface |
| PDF, CSV e PNG | Implementados e verificados em execução pela interface |
| CNN 1D, ResNet-18 STFT e híbrido ABC | Checkpoints reais testados em CPU, com comparação ao pipeline do Gate 5 em três janelas sintéticas por modelo |
| HDF5 e Parquet | Adaptadores opcionais para conjuntos numéricos planos. Dependências ausentes no ambiente verificado. Referências complexas de MAT v7.3 e testes de compatibilidade pendentes |
| Lançador .exe | Compilado para Windows. Requer Python e dependências. Distribuição autocontida pendente |
| Núcleo sem IA e instalação | Modo determinístico testado com IA desativada. O percurso de instalação em uma segunda máquina está pendente |

Para repetir os testes, rode test_gate6.py e test_gui.py com o Python configurado. A opção --modelos do teste de paridade espera a pasta de modelos do Gate 5 e o código experimental original em sua pasta irmã codigo. Os logs de execução da aplicação ficam em logs.
