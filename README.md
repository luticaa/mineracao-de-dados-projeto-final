# Extração Amostral Bounded: Amazon Reviews 2023 (Books)

Pipeline reproduzível e resiliente desenvolvido para extrair **até 20.000 avaliações** da categoria **Books** do repositório oficial **Amazon Reviews 2023** (McAuley Lab / UCSD), com foco estrito em **limitar consumo de rede, eliminar cópia local do arquivo bruto e conter a memória RAM**, sem assumir representatividade estatística infundada.

---

## 1. Visão Geral e Fontes Oficiais

- **Dataset Oficial:** [Hugging Face - McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Portal do Projeto:** [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
- **Arquivo de Avaliações Oficial:** `raw/review_categories/Books.jsonl` (~29,5 milhões de avaliações; ~20,1 GB)
- **Tamanho da Amostra ($N$):** Até 20.000 avaliações
- **Semente Aleatória (*Seed*):** `20261109` (configurável e determinística)
- **Notebook Principal:** [`extrair_amostra_20k_amazon_books.ipynb`](extrair_amostra_20k_amazon_books.ipynb)

---

## 2. Por que NÃO Usar `load_dataset(..., trust_remote_code=True)`?

Nas versões recentes da biblioteca `datasets` do Hugging Face ($\ge 4.0$), **scripts remotos de carregamento (`.py`) foram descontinuados e bloqueados por segurança**, gerando a falha:
```text
RuntimeError: Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```
Conforme estipulado nas regras do projeto, **não dependemos dessa API vulnerável**. Adotamos a rota oficial direta de streaming do arquivo JSONL linha a linha via HTTP nativo do Python (`urllib.request`), garantindo:
- **Zero erros de script ou bloqueio de versão do `datasets`**;
- **Zero erros HTTP 404**;
- **Zero armazenamento de arquivo bruto de 20 GB no disco rígido**;
- **Consumo de memória RAM estritamente limitado a $O(N)$ (< 200 MB)**.

---

## 3. Modos de Operação do Pipeline

O pipeline oferece dois modos de operação na Célula 3:

1. **Modo `bounded_10pct` (Foco em economizar rede):**
   - Interrompe a conexão HTTP voluntariamente ao atingir ~10% das linhas (~2.950.000 registros).
   - Transfere apenas ~1,8 GB a 2,0 GB pela rede (**economia real de ~90% de tráfego**).
   - **Enquadramento metodológico:** A amostra resultante representa **exclusivamente o trecho inicial percorrido** da base e possui viés de cobertura e temporalidade.
2. **Modo `full_population` (Foco em representatividade uniforme global):**
   - Percorre o stream completo até o final com *Reservoir Sampling* (Algoritmo R de Waterman & Vitter).
   - Garante probabilidade uniforme exata ($N/M$) sobre todos os 29,5 milhões de registros, sem guardar o arquivo bruto em disco.

---

## 4. Custos Operacionais e Recursos Previstos

| Recurso | Base Completa (Books) | Modo `bounded_10pct` | Modo `full_population` |
| :--- | :--- | :--- | :--- |
| **Registros Totais** | ~29.500.000 avaliações | 20.000 avaliações | 20.000 avaliações |
| **Linhas Percorridas** | ~29.500.000 linhas | ~2.950.000 linhas | ~29.500.000 linhas |
| **Tráfego de Rede** | ~20,1 GB | ~1,8 GB a 2,0 GB (**~90% economia**) | ~20,1 GB |
| **Armazenamento em Disco Bruto** | ~20,1 GB | **0 GB** (sem arquivo bruto local) | **0 GB** (sem arquivo bruto local) |
| **Memória RAM Utilizada** | Inviável carregar tudo | < 200 MB ($O(N)$) | < 200 MB ($O(N)$) |
| **Representatividade** | População | Apenas trecho inicial percorrido | Uniforme sobre toda a população |

---

## 5. Arquitetura de Resiliência e Escrita Segura

- **Streaming sem Cópia Local:** O processamento decodifica linha por linha diretamente do stream HTTP via `urllib.request` com `json.loads`.
- **Tratamento de Linhas Corrompidas:** Cada registro passa por validação com contagem de anomalias.
- **Escrita Segura e Atômica:** O CSV é gravado primeiramente em arquivo temporário (`.csv.tmp`) e, somente após validação da integridade de escrita, é renomeado atomicamente (`os.replace`) para o arquivo final.
- **Hash Criptográfico SHA-256:** Calculado automaticamente sobre o CSV gerado e registrado nos metadados.

---

## 6. Estrutura do Repositório

```text
projeto_final/
├── README.md                              # Guia completo de metodologia e execução
├── requirements.txt                      # Dependências Python (pandas, numpy, matplotlib, etc.)
├── .gitignore                            # Proteção contra dados massivos e temporários
├── extrair_amostra_20k_amazon_books.ipynb # Pipeline bounded resiliente (streaming direto via HTTP)
└── outputs/                              # Diretório de artefatos finais
    ├── books_reviews_sample_20k_seed_20261109.csv           # Amostra tabular final (UTF-8)
    ├── books_reviews_sample_20k_seed_20261109_metadata.json # Metadados, proveniência e hash SHA-256
    └── diagnosticos_books_reviews_sample_20k_seed_20261109.png # Painel visual de validação técnica
```

---

## 7. Como Executar

### 1. Preparar Dependências
No terminal, na pasta raiz:

```bash
pip install -r requirements.txt
```

### 2. Executar no Jupyter
Abra o notebook [`extrair_amostra_20k_amazon_books.ipynb`](extrair_amostra_20k_amazon_books.ipynb) no VS Code / Jupyter e clique em **Kernel -> Restart & Run All**.
As células executarão sequencialmente sem qualquer erro de dependência ou de script remoto descontinuado.
