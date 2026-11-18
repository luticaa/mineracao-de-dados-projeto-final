# Mineração de Texto para Previsão de Avaliações Negativas em E-commerce de Livros: Uma Abordagem com TF-IDF e Support Vector Machines

**Disciplina:** Mineração de Dados (2COP024) — 2º Semestre de 2026  
**Instituição:** Universidade Estadual de Londrina (UEL) — Departamento de Ciência da Computação  
**Integrantes:** Grupo de Trabalho Acadêmico  
**Repositório Oficial:** [github.com/luticaa/mineracao-de-dados-projeto-final](https://github.com/luticaa/mineracao-de-dados-projeto-final.git)

---

## Resumo Executivo

Este relatório apresenta os resultados do Projeto Integrador de Mineração de Dados, cujo objetivo é responder à pergunta: *“É possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha?”*. Utilizando uma amostra controlada de 20.000 avaliações da categoria *Books* do repositório *Amazon Reviews 2023*, formulou-se um problema de classificação binária supervisionada (Notas 1-2 $\rightarrow$ Negativa / Classe 1; Notas 4-5 $\rightarrow$ Positiva / Classe 0; Nota 3 excluída por neutralidade ambivalente). Adotou-se representação vetorial TF-IDF com unigramas e bigramas integrada estritamente em *Pipelines* do Scikit-Learn para prevenção absoluta de vazamento de dados (*data leakage*). Na comparação de modelos exigida pelo regulamento, a Máquina de Vetores de Suporte Linear (*Linear SVM*) superou a Regressão Logística, atingindo F1-Score médio de 0,6874 na validação cruzada 5-fold. Na análise de atributos da Etapa 4, os métodos de seleção (*Filter* via $\chi^2$ e *Embedded* via Lasso L1) retiveram entre 94% e 96% do desempenho com eliminação de mais de 70% dos termos, enquanto a redução dimensional por *TruncatedSVD* (equivalente ao PCA para dados esparsos) sofreu forte degradação (F1 de 0,4862) devido à diluição de termos negativos críticos em vetores densos. No conjunto de teste independente congelado ($N = 3.678$), o modelo campeão alcançou **Recall de 77,70%**, **F1-Score de 0,7044**, **ROC-AUC de 0,9689** e **PR-AUC de 0,7843**, confirmando a viabilidade técnica e a relevância prática da abordagem para mitigação de abandono de clientes (*churn*) em comércio eletrônico.

---

## 1. Introdução e Pergunta de Pesquisa

No comércio eletrônico moderno, plataformas como a Amazon processam dezenas de milhares de avaliações diárias. A leitura manual e individual de cada comentário é operacionalmente inviável para as equipes de atendimento, curadoria editorial e controle de qualidade. Nesse ecossistema, as avaliações com notas baixas (críticas, frustrações com enredo, edições corrompidas, problemas de encadernação ou propaganda enganosa) carregam um peso econômico desproporcional: consumidores insatisfeitos que não recebem resposta rápida tendem ao abandono da plataforma (*churn*), cancelamento de assinaturas e difusão de impressões negativas.

Diante desse desafio prático de inteligência de negócios, formulou-se a seguinte pergunta de pesquisa da disciplina:

> **“É possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha?”**

A tarefa foi modelada como um problema de **classificação binária supervisionada** utilizando processamento de linguagem natural e mineração de texto clássica. O foco principal não é alcançar acurácia ilusória em classes desbalanceadas, mas sim maximizar a capacidade de detecção de avaliações insatisfatórias (*Recall* e *F1-Score da Classe Negativa*), respeitando as diretrizes de integridade metodológica e ausência de vazamento de dados.

---

## 2. Base de Dados, Proveniência e Engenharia da Variável-Alvo

### 2.1 Proveniência dos Dados e Amostragem Resiliente
Os dados brutos provêm do repositório acadêmico oficial **Amazon Reviews 2023** (McAuley Lab / UCSD), categoria `Books`. A população original dessa categoria compreende cerca de 29,5 milhões de avaliações (~20,1 GB em formato JSONL compactado). 

Para atender ao critério de reprodutibilidade e execução eficiente em ambiente local ou Google Colab, foi desenvolvido um pipeline de extração por streaming HTTP direto (`extrair_amostra_20k_amazon_books.ipynb`) utilizando o **Algoritmo de Reservatório Bounded** (Vitter, 1985). O pipeline limitou o consumo de memória RAM a menos de 200 MB ($O(N)$), interrompendo a leitura ao atingir ~10% das linhas do arquivo bruto, extraindo exatamente **20.000 registros** com semente determinística `20261109`. A integridade do arquivo gerado (`outputs/books_reviews_sample_20k_seed_20261109.csv`) foi validada por hash SHA-256 e registrada em metadados formais.

### 2.2 Dicionário de Colunas Utilizadas
| Coluna | Tipo Original | Papel no Pipeline | Descrição Semântica |
| :--- | :--- | :--- | :--- |
| `rating` | `float64` | Definição do Alvo | Nota de 1.0 a 5.0 estrelas atribuída pelo cliente. |
| `title` | `object` | Entrada Textual | Título síntese da resenha. |
| `text` | `object` | Entrada Textual | Conteúdo dissertativo da resenha. |
| `parent_asin` | `object` | Metadado de Controle | Identificador do produto editorial na Amazon. |
| `user_id` | `object` | Metadado de Controle | Identificador do usuário avaliador. |

### 2.3 Formulação da Variável-Alvo (`is_negative`)
A distribuição das notas nas 20.000 avaliações brutas evidenciou o comportamento assimétrico típico de sistemas de recomendação de livros:
- **1.0 estrela:** 737 avaliações (3,68%);
- **2.0 estrelas:** 741 avaliações (3,70%);
- **3.0 estrelas:** 1.612 avaliações (8,06%);
- **4.0 estrelas:** 3.420 avaliações (17,10%);
- **5.0 estrelas:** 13.490 avaliações (67,45%).

Conforme estabelecido na literatura de análise de sentimentos (Pang & Lee, 2008), avaliações de 3 estrelas são ambivalentes e reúnem elogios parciais e ressalvas moderadas, constituindo uma zona cinzenta que prejudica a separabilidade de classes polares. Portanto, a nota 3.0 foi excluída da modelagem (1.612 linhas descartadas).

O alvo binário final foi estabelecido de forma transparente e consistente:
- **Notas 1.0 e 2.0 $\rightarrow$ Classe 1 (Avaliação Negativa):** 1.478 exemplos (**8,04%**);
- **Notas 4.0 e 5.0 $\rightarrow$ Classe 0 (Avaliação Positiva):** 16.910 exemplos (**91,96%**).
- **Base elegível final:** **18.388 exemplos** (proporção de ~11,4 positivas para cada negativa).

---

## 3. Metodologia e Prevenção Estrita de Vazamento (*Data Leakage*)

### 3.1 Protocolo de Partição dos Dados
Em cumprimento rigoroso à Seção 7 do regulamento da disciplina (*Erros que mais tiram nota: Vazamento*), adotou-se o seguinte protocolo:
1. **Separação Estratificada Prévia:** Antes de qualquer etapa de vetorização, vocabulário ou aprendizado, os 18.388 registros foram divididos em **80% para Treino (14.710)** e **20% para Teste (3.678)** utilizando `train_test_split` estratificado pela classe e com semente determinística (`random_state=42`).
2. **Congelamento do Teste:** O conjunto de teste foi completamente isolado, não sendo utilizado em nenhuma escolha de hiperparâmetros, limiares de decisão ou seleção de vocabulário.
3. **Encapsulamento em Pipelines:** Todas as operações estatísticas (cálculo de frequências $IDF$, seleção de atributos por teste Qui-Quadrado, projeção por SVD e ajuste dos classificadores) foram executadas **exclusivamente dentro de objetos `Pipeline` do Scikit-Learn** no âmbito de cada partição de treino da validação cruzada.

### 3.2 Mineração de Texto e TF-IDF
Para maximizar a riqueza informacional, concatenou-se o título e o corpo da resenha (`review_full = title + " " + text`), uma vez que títulos concisos frequentemente resumem o descontentamento (ex.: *"Péssima edição"* ou *"Waste of money"*).

O vetorizador `TfidfVectorizer` foi parametrizado com:
- `ngram_range=(1, 2)`: Captura termos isolados e pares contíguos de palavras, viabilizando o aprendizado de modificadores de negação essenciais (como *"not good"*, *"not recommend"*, *"waste time"*);
- `min_df=3`: Elimina erros ortográficos singulares e palavras com frequência estritamente pontual;
- `max_df=0.90`: Descarta termos que ocorrem em mais de 90% dos documentos (como *"book"*), reduzindo a esparsidade desnecessária;
- `sublinear_tf=True`: Aplica escala logarítmica ($1 + \log(TF)$) para atenuar a influência de termos repetidos excessivamente em resenhas prolixas;
- `max_features=5000`: Limita o espaço dimensional aos 5.000 n-grams mais relevantes no treino.

---

## 4. Modelagem e Validação Cruzada dos Classificadores da Disciplina

Conforme exigido na Etapa 3 da disciplina (*"pelo menos dois classificadores da disciplina (árvore, floresta, boosting, SVM, logística...), ajustados por validação cruzada"*), comparou-se dois métodos com fundamentos matemáticos complementares:

1. **Máquina de Vetores de Suporte Linear (`LinearSVC`):** Método de margem máxima que encontra o hiperplano de separação ótimo minimizando a função de perda *hinge* com penalidade L2. É amplamente consolidado como um dos métodos mais robustos para texto esparso em alta dimensão.
2. **Regressão Logística (`LogisticRegression`):** Modelo linear discriminativo probabilístico baseado na minimização da perda logística (*cross-entropy / log-loss*).

Ambos os modelos foram configurados com o parâmetro `class_weight='balanced'`, que penaliza os erros na classe minoritária proporcionalmente ao inverso da sua frequência na base de treino.

A busca em grade de hiperparâmetros (parâmetro de regularização $C \in \{0.1, 0.5, 1.0, 2.0\}$) foi conduzida via **Validação Cruzada Estratificada com 5 Folds (`StratifiedKFold`)** exclusivamente no conjunto de treino, otimizando o **F1-Score da Classe Negativa (1)**.

### Tabela 1: Resultado da Validação Cruzada Estratificada (5 Folds no Treino)
| Classificador | F1-Score Médio (CV) | Desvio Padrão ($\sigma$) | Melhor Hiperparâmetro |
| :--- | :---: | :---: | :---: |
| **Linear SVM (`LinearSVC`)** *(Campeão)* | **0,6874** | **$\pm 0,0189$** | **$C = 0.5$** |
| **Regressão Logística** | 0,6851 | $\pm 0,0142$ | $C = 2.0$ |

Ambos os modelos apresentaram desempenho competitivo e elevado. O **Linear SVM** alcançou a liderança empírica com F1-Score de **0,6874**, sendo selecionado como o classificador campeão para a etapa de seleção e redução de atributos.

---

## 5. Comparação Obrigatória: Seleção e Redução de Atributos

Atendendo estritamente à Etapa 4 do regulamento (*"com o melhor classificador, comparar três versões medidas da mesma forma: sem nada, com seleção de atributos (pelo menos dois métodos, por exemplo um filter e um embedded) e com redução de dimensionalidade (PCA); dizer qual funcionou melhor e por quê"*), avaliou-se 4 arquiteturas baseadas no **Linear SVM** sob validação cruzada estratificada em 5 folds:

1. **Baseline (Sem nada):** Vetor TF-IDF original completo com 5.000 n-grams;
2. **Método Filter:** `SelectKBest` utilizando o teste estatístico Qui-Quadrado ($\chi^2$), retendo os 1.500 termos mais associados à classe negativa;
3. **Método Embedded:** `SelectFromModel` acoplado a um estimador `LinearSVC` com regularização L1 (Lasso), promovendo a seleção intrínseca de variáveis ao zerar os coeficientes de termos redundantes;
4. **Redução Dimensional (PCA / TruncatedSVD):** `TruncatedSVD` com 100 componentes principais.

> **Fundamentação Matemática do TruncatedSVD:** Em processamento de linguagem natural, matrizes TF-IDF são altamente esparsas (>98% de zeros). A aplicação do PCA convencional requer a centralização dos dados pela média amostral ($\mathbf{X} - \boldsymbol{\mu}$), o que destrói a esparsidade da matriz e causaria estouro imediato de memória RAM. O `TruncatedSVD` calcula a decomposição em valores singulares diretamente sobre a matriz esparsa, constituindo a realização computacional e matematicamente adequada do PCA para texto.

### Tabela 2: Comparação de Seleção e Redução de Atributos (5-Fold CV F1)
| Abordagem | F1-Score Médio | Desvio Padrão | Retenção de Desempenho | Atributos Efetivos |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Sem seleção)** | **0,6874** | **$\pm 0,0189$** | **100,0% (Referência)** | 5.000 termos |
| **Filter (`SelectKBest` $\chi^2$)** | 0,6472 | $\pm 0,0094$ | 94,2% | 1.500 termos (70% redução) |
| **Embedded (`SelectFromModel` L1)** | 0,6614 | $\pm 0,0160$ | 96,2% | ~1.400 termos (>70% redução) |
| **Redução PCA (`TruncatedSVD`)** | 0,4862 | $\pm 0,0129$ | 70,7% | 100 componentes densos |

### Análise Técnica: Qual funcionou melhor e por quê?
- **Melhor Versão:** O **Baseline (Sem nada)** obteve o maior F1-Score (0,6874). Em problemas de análise de sentimentos, a riqueza de vocabulário na cauda longa (palavras raras mas decisivas de insatisfação) confere poder discriminatório que supera o ganho de parcimônia.
- **Eficiência dos Métodos de Seleção:** Tanto o método *Filter* quanto o *Embedded* demonstraram excelente eficiência, preservando entre 94% e 96% do poder preditivo original enquanto eliminaram mais de 70% da dimensionalidade. Em sistemas embarcados ou com restrição estrita de latência, o método *Embedded* representaria a melhor escolha de compromisso.
- **Degradação Acentuada da Redução Dimensional (PCA/SVD):** A projeção em 100 componentes densos sofreu uma queda drástica de quase 30% no F1-Score (0,4862). Isso decorre da natureza intrínseca do sinal semântico textual: enquanto termos esparsos específicos (como *"terrible"*, *"waste"*, *"horrible"*, *"damaged"*) atuam como chaves de alta polaridade, a compressão ortogonal do PCA projeta esses termos em combinações lineares densas que agregam ruído de milhares de termos neutros, diluindo o sinal polarizador da classe minoritária.

---

## 6. Avaliação Final pelo Custo do Erro no Conjunto de Teste

O modelo campeão (**Linear SVM Baseline**) foi ajustado sobre a totalidade do conjunto de treino (14.710 exemplos) e submetido a uma **avaliação única no conjunto de teste independente congelado (3.678 exemplos)**.

### 6.1 Análise do Custo Assimétrico do Erro
Em e-commerce, os custos de classificação incorreta são severamente assimétricos:
- **Falso Negativo (FN):** Um cliente que detestou a compra é rotulado como satisfeito. **Custo Crítico (Alto):** O cliente insatisfeito é ignorado, gerando atrito no pós-venda, avaliações destrutivas na plataforma e perda irreversível do cliente (*churn*).
- **Falso Positivo (FP):** Um cliente satisfeito é sinalizado preventivamente como insatisfeito. **Custo Mínimo (Baixo):** Um atendente ou sistema automatizado revisa a resenha e constata que o usuário estava satisfeito, sem prejuízo ao negócio.

Portanto, o modelo deve priorizar um **Recall elevado para a Classe Negativa**, controlando a Precisão para evitar sobrecarga operacional.

### Tabela 3: Relatório de Desempenho no Conjunto de Teste Independente ($N = 3.678$)
| Classe | Precisão | Recall | F1-Score | Suporte Real |
| :--- | :---: | :---: | :---: | :---: |
| **Positiva (Classe 0: 4-5 estrelas)** | 0,9799 | 0,9624 | 0,9711 | 3.382 |
| **Negativa (Classe 1: 1-2 estrelas)** | **0,6443** | **0,7770** | **0,7044** | **296** |
| **Média Ponderada** | 0,9529 | 0,9475 | 0,9496 | 3.678 |
| **Acurácia Global (Informativa)** | — | — | **0,9475** | 3.678 |

### Métricas Globais da Detecção de Insatisfação:
- **Recall da Classe Negativa:** **77,70%** (detecta aproximadamente 8 em cada 10 insatisfações reais do público);
- **F1-Score da Classe Negativa:** **0,7044**;
- **Área sob a Curva Precision-Recall (PR-AUC):** **0,7843** (superando com folga o piso aleatório de 0,0805);
- **ROC-AUC:** **0,9689**, demonstrando alta capacidade de discriminação ordinal entre as classes.

### Matriz de Confusão no Teste:
- Verdadeiros Positivos da Insatisfação (TP): **230** avaliações negativas identificadas corretamente;
- Falsos Negativos (FN): **66** avaliações negativas não detectadas;
- Verdadeiros Negativos (TN): **3.255** avaliações positivas confirmadas;
- Falsos Positivos (FP): **127** avaliações positivas classificadas preventivamente como negativas.

---

## 7. Diagnóstico Léxico de Ruído nos Rótulos e Estudo de Ablação

### 7.1 Diagnóstico com VADER
Investigou-se se discrepâncias entre o texto redigido e a nota atribuída poderiam representar ruído de rótulo (*label noise*) derivado de toques acidentais (*miss-clicks*) ou críticas direcionadas à entrega. 

Utilizando o analisador léxico **VADER** (`SentimentIntensityAnalyzer`, pacote `nltk`), estabeleceu-se um critério conservador de divergência extrema ($|score| \ge 0,60$):
- **Suspeitas em Negativas (Nota 1-2 com texto léxico $\ge +0.60$):** 371 casos (25,10% das negativas);
- **Suspeitas em Positivas (Nota 4-5 com texto léxico $\le -0.60$):** 516 casos (3,05% das positivas);
- **Total sinalizado:** **887 avaliações (4,82% da base elegível)**.

### 7.2 Estudo de Ablação Experimental
Para avaliar o impacto desses casos sem comprometer a integridade do teste, executou-se um experimento de ablação:
- **Cenário Referência:** Modelo treinado com todos os 14.710 exemplos de treino;
- **Cenário Experimental:** Remoção dos 711 casos suspeitos **estritamente do conjunto de treino** (13.999 exemplos restantes).
- **Ambiente de Teste Comum:** Ambos os modelos foram avaliados no **mesmo conjunto de teste original (3.678 exemplos)**, com os mesmos rótulos intocados.

### Tabela 4: Comparação da Ablação de Ruído no Conjunto de Teste Comum
| Cenário | Exemplos de Treino | F1-Score Negativo (Teste) | PR-AUC (Teste) |
| :--- | :---: | :---: | :---: |
| **Referência (Treino com Ruído Natural)** | 14.710 | **0,7044** | **0,7843** |
| **Experimental (Treino Sem Suspeitas VADER)** | 13.999 | 0,6988 | 0,7778 |

**Conclusão Metodológica Sóbria:** A exclusão dos casos suspeitos no treino não gerou ganho expressivo de desempenho no teste real, apresentando variação residual de 0,7044 para 0,6988. Esse achado comprova que:
1. O classificador Linear SVM com regularização L2 e representação TF-IDF já possui elevada robustez contra ruídos esparsos de rotulação;
2. Descartar avaliações com base em heurísticas léxicas simples pode remover resenhas legítimas ricas em sarcasmo ou enredos trágicos (ex.: livros de terror ou drama descritos como *"horrible"* ou *"terrifying"*, mas com nota 5 estrelas).

---

## 8. Interpretação dos Coeficientes e Termos Mais Informativos

A inspeção dos coeficientes do hiperplano do `LinearSVC` validou a consistência semântica do aprendizado:
- **Termos com Maior Peso para Classe Negativa (Classe 1):** *"waste"*, *"not"*, *"poor"*, *"disappointed"*, *"boring"*, *"return"*, *"waste time"*, *"worst"*, *"terrible"*, *"dont waste"*, *"awful"*.
- **Termos com Maior Peso para Classe Positiva (Classe 0):** *"great"*, *"excellent"*, *"loved"*, *"must read"*, *"best"*, *"highly recommend"*, *"wonderful"*, *"easy read"*, *"beautiful"*, *"amazing"*.

A presença de bigramas explícitos como *"waste time"* e *"dont waste"* entre os termos mais punitivos comprova a importância da escolha de $n$-grams compostos para superar limitações clássicas de modelos *Bag of Words* simples.

---

## 9. Limitações Conhecidas

1. **Escopo da Amostra Bounded:** Para conter consumo de rede e permitir reprodutibilidade ágil, a amostragem percorreu o trecho inicial da base da Amazon, podendo carregar viés cronológico ou de lotes de upload.
2. **Contexto Semântico Profundo:** Modelos baseados em TF-IDF não codificam relações sintáticas de longa distância nem dependências anafóricas complexas presentes em análises literárias aprofundadas.
3. **Generalização Multidomínio:** O vocabulário otimizado é específico para resenhas de livros (*Books*), necessitando de recalibração para categorias com léxicos distintos (como eletrônicos ou vestuário).

---

## 10. Conclusão

Este trabalho respondeu afirmativamente e de forma conclusiva à pergunta central da disciplina:

> **Sim, é perfeitamente possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha.**

Utilizando uma arquitetura clássica e fundamentada de Mineração de Dados — representação TF-IDF acoplada a uma Máquina de Vetores de Suporte Linear (*Linear SVM*) com balanceamento de classes —, o sistema alcançou um **Recall de 77,70%** e **F1-Score de 0,7044** sobre uma base de teste com forte assimetria ($92\%$ positivas), gerando um índice **ROC-AUC de 0,9689**. O protocolo experimental adotou rigor absoluto contra vazamento de dados (*data leakage*), comparou formalmente técnicas de seleção e redução de atributos e demonstrou a viabilidade de aplicar mineração de texto prática e interpretável em problemas reais de comércio eletrônico.

---

## Referências Bibliográficas

1. **Cortes, C., & Vapnik, V. (1995).** *Support-Vector Networks*. Machine Learning, 20(3), pp. 273–297.
2. **Hutto, C. J., & Gilbert, E. E. (2014).** *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. In: Eighth International Conference on Weblogs and Social Media (ICWSM-14), Ann Arbor, MI.
3. **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval*. Cambridge University Press.
4. **McAuley, J., Hou, Y., Li, J., & He, R. (2023).** *Amazon Reviews 2023 Dataset*. McAuley Lab, University of California San Diego. Disponível em: <https://amazon-reviews-2023.github.io/>.
5. **Pang, B., & Lee, L. (2008).** *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval, 2(1–2), pp. 1–135.
6. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, pp. 2825–2830.
7. **Vitter, J. S. (1985).** *Random Sampling with a Reservoir*. ACM Transactions on Mathematical Software (TOMS), 11(1), pp. 37–57.
