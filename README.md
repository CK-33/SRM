# SRM Lab — pesquisa e aplicação experimental

Campanha de diagnóstico em motores de relutância chaveada por correntes A/B/C. A documentação acompanha a evolução da exploração dos sinais até uma aplicação desktop Windows.

- [Gate 1 — Exploração dos sinais](gate01/): Caracterização inicial de simulações, marcadores e AWGN. O PDF documenta as janelas de 20 ms e o Welch usado no piloto.
- [Gate 2 — Periodicidade e extração](gate02/): Autocorrelação, defasagem trifásica, referência periódica e preparação de atributos. O ganho do alinhamento retrospectivo é descritivo.
- [Gate 3 — Diagnóstico clássico](gate03/): Regras determinísticas, CUSUM/EWMA, PCA, classificadores e robustez no espaço de atributos.
- [Gate 4 — Engenharia de dados](gate04/): Regularização a 200 kHz, representação canônica a 25 kHz, avaliação do bootstrap e expansão controlada. O PDF disponível cobre E03. O estado de E04 permanece a confirmar.
- [Gate 5 — Deep learning e robustez](gate05/): STFT, transferência, ajuste seletivo, CNN 1D, híbrido, expansão, robustez, interpretabilidade e replay causal de E01 a E12.
- [Gate 6 — SRM Lab para Windows](gate06/): Aplicativo desktop offline para análise de correntes, diagnóstico experimental, replay e exportação. O modo determinístico roda sem checkpoints.

## Escopo científico

Os rótulos pré/pós são proxies temporais em registros simulados. A validação exploratória do Gate 5 usa duas origens e o relógio absoluto separa as janelas puras. R06/R12 ficaram reservados. As métricas não sustentam generalização a novas máquinas ou mecanismos físicos.

## Conteúdo distribuído

Inclui relatórios, figuras selecionadas, código e exemplo sintético. Sinais proprietários, variantes em MAT, checkpoints, logs locais e executáveis ficam fora do repositório. A publicação não estabelece licença de redistribuição. A licença será definida pelo titular.
