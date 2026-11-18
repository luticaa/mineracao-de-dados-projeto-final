# Projeto Integrador de Mineração de Dados (2COP024 - UEL)
**Universidade Estadual de Londrina — Ciência de Dados e Inteligência Artificial**

Projeto completo da disciplina de Mineração de Dados investigando a pergunta de pesquisa:

> **“É possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha?”**

---

## 1. Visão Geral e Fontes de Dados

- **Dataset Oficial:** [Amazon Reviews 2023 (Books)](https://amazon-reviews-2023.github.io/) — McAuley Lab / UCSD.
- **Amostra Analisada:** 20.000 avaliações extraídas com streaming HTTP resiliente (*Algoritmo de Reservatório Bounded*), registradas com integridade SHA-256.
- **Formulação do Alvo Binário:**
  - Notas 1.0 e 2.0 $\rightarrow$ **Avaliação Negativa (Classe 1)**: 1.478 exemplos (8,04%);
  - Notas 4.0 e 5.0 $\rightarrow$ **Avaliação Positiva (Classe 0)**: 16.910 exemplos (91,96%);
  - Nota 3.0 $\rightarrow$ **Excluída da classificação binária** (1.612 avaliações neutras/ambíguas).
  - Base efetiva de modelagem: **18.388 avaliações**.

---

## 2. Estrutura do Repositório

```text
projeto_final/
├── README.md                                  # Documentação completa, metodologia e guia de estudo
├── requirements.txt                          # Dependências (scikit-learn, nltk, pandas, etc.)
├── LICENSE                                   # Licença de código aberto (MIT)
├── contexto/
│   └── PROJETO_INTEGRADOR.pdf                # Diretrizes e rubrica oficial da disciplina
├── projeto_integrador_mineracao_dados.ipynb  # NOTEBOOK PRINCIPAL: Análise, Modelagem e Guia de Estudo
├── extrair_amostra_20k_amazon_books.ipynb     # Notebook de proveniência e amostragem dos dados
└── outputs/                                  # Dados gerados, metadados e figuras
    ├── books_reviews_sample_20k_seed_20261109.csv           # Amostra tabular (UTF-8, N = 20.000)
    ├── books_reviews_sample_20k_seed_20261109_metadata.json # Hash criptográfico SHA-256 e proveniência
    ├── diagnosticos_books_reviews_sample_20k_seed_20261109.png # Painel visual da extração
    ├── figura1_distribuicao_classes_eda.png                 # Distribuição de classes e extensão dos textos
    ├── figura2_diagnostico_divergencias_vader.png           # Diagnóstico léxico de ruído (VADER vs Notas)
    ├── figura3_comparacao_selecao_reducao.png               # Comparação 5-fold CV (Baseline vs Filter vs Embedded vs PCA)
    ├── figura4_avaliacao_matriz_confusao_pr_roc.png         # Matriz de Confusão e curva Precision-Recall no teste
    └── figura5_coeficientes_palavras_chave.png              # Termos de maior peso estatístico (n-grams)
```

---

## 3. Metodologia e Prevenção Estrita de Vazamento (*Data Leakage*)

Para atender com nota máxima aos critérios de rigor metodológico da disciplina:

1. **Separação Estratificada:** Divisão inicial única da base elegível (18.388 exemplos) em **80% Treino** (14.710) e **20% Teste** (3.678), com semente fixa (`random_state=42`) e estratificação pelo alvo binário.
2. **Isolamento Absoluto do Teste:** O conjunto de teste é congelado e reservado exclusivamente para a avaliação final.
3. **Pipelines Encapsulados:** Toda transformação estatística — cálculo de vocabulário, frequências $IDF$, seleção de atributos e projeção via SVD/PCA — é executada **estritamente dentro de objetos `Pipeline` do Scikit-Learn** ajustados apenas sobre os folds de treino.
4. **Métricas Apropriadas para Classes Desbalanceadas:** Devido à proporção de ~11,4 positivas para cada negativa, a acurácia é desconsiderada como critério de decisão. Toda seleção e otimização é conduzida pelo **F1-Score da Classe Negativa (1)**, **Recall** e **PR-AUC (Precision-Recall Area Under Curve)**.

---

## 4. Síntese dos Resultados Experimentais

### 4.1 Validação Cruzada dos Classificadores (5 Folds Estratificados no Treino)

| Classificador | F1-Score Médio (Classe Negativa) | Desvio Padrão | Melhores Parâmetros | Justificativa Técnica |
| :--- | :---: | :---: | :--- | :--- |
| **Regressão Logística** *(Campeão)* | **0,6851** | $\pm 0,0142$ | $C=2.0$, `max_features=5000` | Ponderação `class_weight='balanced'` equilibra o custo do desbalanceamento severo. |
| **Multinomial Naive Bayes** | 0,3985 | $\pm 0,0198$ | $\alpha=0.1$, `max_features=5000` | Sofre com a ausência de ponderação de custo de classe sob forte desbalanceamento. |

### 4.2 Comparação Obrigatória: Seleção e Redução de Atributos (5-Fold CV)

Com a **Regressão Logística** como estimador base, comparamos as 4 versões medidas rigorosamente sob o mesmo protocolo:

| Abordagem | F1-Score Médio (CV) | Retenção de Desempenho | Redução de Atributos | Análise Técnica |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Sem seleção)** | **0,6851** | **100,0%** (Referência) | 0% (5.000 n-grams) | **Melhor versão.** Preserva a rica cauda longa de vocabulário insatisfatório. |
| **Filter (`SelectKBest` $\chi^2$)** | 0,6262 | 91,4% | **70,0%** (1.500 termos) | Mantém >91% do desempenho eliminando 70% dos termos redundantes. |
| **Embedded (`SelectFromModel` L1)** | 0,6129 | 89,5% | **>70,0%** (esparsificado) | Regularização Lasso zera coeficientes irrelevantes de forma automatizada. |
| **Redução PCA (`TruncatedSVD`)** | 0,4775 | 69,7% | 100 componentes densos | **Pior versão.** A compressão ortogonal densa dilui termos negativos críticos e raros no ruído. |

> **Nota Metodológica:** Para matrizes TF-IDF esparsas, o `TruncatedSVD` é o equivalente matematicamente correto do PCA, pois o PCA tradicional exigiria centralizar a matriz na média ($\mathbf{X} - \boldsymbol{\mu}$), destruindo a esparsidade e excedendo a capacidade de memória RAM.

### 4.3 Avaliação Final no Conjunto de Teste Independente ($N = 3.678$)

O modelo campeão foi avaliado uma única vez no teste congelado:

- **Recall da Classe Negativa (1):** **82,43%** (o modelo detecta 8 de cada 10 insatisfações reais);
- **F1-Score da Classe Negativa (1):** **0,6778**;
- **Precisão da Classe Negativa (1):** **0,5755**;
- **Área sob a Curva Precision-Recall (PR-AUC):** **0,7977** (frente a um baseline aleatório de apenas 0,0805);
- **ROC-AUC:** **0,9716**;
- **Acurácia Global (informativa):** 93,69%.

### 4.4 Diagnóstico Léxico de Ruído nos Rótulos e Estudo de Ablação

- **Diagnóstico com VADER:** Sinalizou **887 casos de divergência extrema** (4,82% da base), sendo 371 em avaliações de nota 1-2 com texto altamente elogioso e 516 em avaliações de nota 4-5 com texto altamente crítico.
- **Ablação Experimental:** Removendo os 711 casos suspeitos **exclusivamente do conjunto de treino** e reavaliando no **mesmo teste original**, o F1-score variou discretamente de 0,6778 para 0,6931. Isso comprova que classificadores lineares com TF-IDF já possuem robustez intrínseca a ruídos pontuais, dispensando descarte indiscriminado de dados.

---

## 5. Resposta à Pergunta de Pesquisa

> **Conclusão:** **Sim, é plenamente possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha.**
> Com técnicas clássicas da disciplina (TF-IDF + Regressão Logística ponderada), o sistema identifica 82,4% de todas as avaliações negativas na base de teste, permitindo que plataformas de e-commerce identifiquem insatisfações críticas de clientes de forma automatizada.

---

## 6. Guia de Revisão Rápida para a Apresentação Oral

A Seção 13 do notebook [`projeto_integrador_mineracao_dados.ipynb`](projeto_integrador_mineracao_dados.ipynb) traz o resumo didático dos 7 temas-chave da disciplina:

1. **Binarização do Alvo:** Por que remover nota 3? (Para focar na polaridade de sentimento sem ambiguidade).
2. **Prevenção de Vazamento:** Por que usar `Pipeline`? (Para garantir que TF-IDF e seleção nunca vejam o teste).
3. **TF-IDF:** Por que unigramas + bigramas? (Para capturar negações como *"not good"* e termos como *"waste time"*).
4. **Modelos:** Por que Regressão Logística superou Naive Bayes? (Pela ponderação `class_weight='balanced'`).
5. **Seleção vs Redução:** Por que o PCA/SVD foi pior? (Porque palavras raras e decisivas são diluídas em componentes densos).
6. **Métricas e Custo do Erro:** Por que focar no Recall/F1 e não na acurácia? (Porque o custo de ignorar um cliente insatisfeito é alto e a base tem 92% de positivos).
7. **Ruído de Rótulos:** O ruído foi comprovado? (Não, é uma hipótese tratada como suspeita diagnóstica investigada por ablação).

---

## 7. Como Executar o Projeto

```bash
# 1. Clonar o repositório
git clone https://github.com/luticaa/mineracao-de-dados-projeto-final.git
cd mineracao-de-dados-projeto-final

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar o léxico VADER (se for reexecutar o diagnóstico)
python -c "import nltk; nltk.download('vader_lexicon')"

# 4. Executar o Jupyter
jupyter lab
```

Abra [`projeto_integrador_mineracao_dados.ipynb`](projeto_integrador_mineracao_dados.ipynb) e execute todas as células sequencialmente.
