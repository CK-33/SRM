# Validação do SRM Lab

Ambiente verificado: Windows, Python 3.13.0 de 64 bits, NumPy 2.5.5, SciPy 1.18.1, Pillow 13, PyTorch 2.13.0+cpu e torchvision 0.28.0+cpu.

- O lançador Windows foi compilado com o compilador C# local. O teste automatizado de abertura pelo EXE terminou com código 0.
- Os testes do núcleo confirmaram importação de CSV e MAT sintéticos, incluindo results.time/results.signals e o esquema canônico registro. Confirmaram FFT, periodicidade, cálculos, exportação e rejeição do nome reservado.
- O detector produziu alteração depois do degrau sintético. Modificar somente as amostras ainda indisponíveis preservou as decisões anteriores. O processamento em blocos coincidiu com a passagem de referência. Um canal preso gerou indicação de aquisição inválida.
- O teste da interface abriu as cinco áreas, associou canais do CSV pela interface, executou FFT e STFT, selecionou intervalo e observou a detecção. O replay avançou, parou enquanto pausado, retomou e foi interrompido. A interface produziu PDF, CSV e PNG e executou inferência com o checkpoint 1D do pacote.
- Os três modelos reais coincidiram com o pipeline do Gate 5 até seis casas decimais nos escores de três janelas sintéticas por modelo. Nenhum arquivo proprietário de corrente foi usado nos testes.
- Medianas indicativas da chamada de inferência em CPU, três amostras por modelo: CNN 1D 2,21 ms, ResNet-18 STFT 25,47 ms e híbrido 27,17 ms. Pico observado do processo de teste ao final das etapas: 338, 492 e 524 MiB, respectivamente, com alocação cumulativa ao carregar os modelos. Essas medidas não caracterizam percentis operacionais nem qualificam tempo real.

HDF5 e Parquet permanecem opcionais e não foram verificados neste ambiente. O aplicativo foi testado nesta máquina. A instalação em outro Windows e o uso com arquivos proprietários explicitamente selecionados pelo pesquisador ainda devem ser confirmados.
