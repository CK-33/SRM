# Executar o SRM Lab

No Windows, utilize Python 3.13 com Tcl/Tk. Crie um ambiente virtual e instale as dependências.

```powershell
cd gate06
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --demo
```

O CSV incluído em demonstracao/ é sintético. A aplicação também importa MAT local selecionado pelo operador. O núcleo calcula indicadores, análise espectral, detecção experimental, replay causal e exportação de PDF, CSV e PNG.

Para testar, rode test_gate6.py e test_gui.py com o Python do ambiente. A suíte determinística funciona sem torch. A paridade neural usa --modelos apontando para a pasta local do Gate 5, com seu código original em uma pasta irmã codigo.

A IA é opcional. Instale requirements-ia.txt e informe ao aplicativo uma pasta local autorizada com --modelos. Os identificadores e a hierarquia esperados constam em neural.py. Pesos aleatórios não são usados como substitutos de checkpoints. O software funciona offline após a instalação.

Os resultados classificam o proxy temporal pré/pós da campanha simulada. A aplicação não é um sistema de proteção ou controle. HDF5 e Parquet são opcionais para conjuntos numéricos planos e não foram validados na entrega.
