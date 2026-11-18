---
marp: true
theme: default
paginate: true
header: "Projeto Integrador — Mineração de Dados (2COP024) | UEL"
footer: "Ciência da Computação / Ciência de Dados e IA — 2026"
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 20px;
    padding: 35px 50px;
  }
  h1 {
    color: #1a365d;
    font-size: 32px;
  }
  h2 {
    color: #2b6cb0;
    font-size: 24px;
    margin-top: 0;
  }
  .highlight {
    background-color: #ebf8ff;
    border-left: 4px solid #3182ce;
    padding: 8px 12px;
    font-weight: 500;
  }
  footer {
    font-size: 12px;
    color: #718096;
  }
---

# Previsão de Avaliações Negativas em E-commerce de Livros
## Mineração de Texto com TF-IDF e Support Vector Machines (SVM)

**Disciplina:** Mineração de Dados (2COP024) — UEL  
**Docente:** Prof. Responsável  
**Integrantes do Grupo:** Equipe de Ciência de Dados e IA  
**Repositório:** `github.com/luticaa/mineracao-de-dados-projeto-final`

---

## 1. O Problema de Negócio e a Pergunta de Pesquisa

* **Contexto de Negócio:**
  * O volume massivo de resenhas em plataformas como a Amazon impede a leitura humana manual de cada comentário.
  * O custo do erro é **assimétrico**: clientes insatisfeitos não atendidos rapidamente cancelam compras, abandonam a plataforma (*churn*) e afetam a reputação da loja.

* **Pergunta Central de Pesquisa:**
  <div class="highlight">
  “É possível prever se uma avaliação de livro da Amazon será negativa com base no texto da resenha?”
  </div>

* **Formulação Técnica:** Classificação binária supervisionada de texto livre com métodos clássicos da disciplina.

---

## 2. Base de Dados e Engenharia da Variável-Alvo

* **Dataset Oficial:** *Amazon Reviews 2023 (Books)* — McAuley Lab / UCSD.
* **Amostragem Resiliente:** 20.000 avaliações extraídas por streaming direto via HTTP com *Algoritmo de Reservatório Bounded* (Seed: `20261109`).

* **Binarização Polarizada do Alvo (`is_negative`):**
  * **Notas 1.0 e 2.0 $\rightarrow$ Classe 1 (Negativa):** 1.478 exemplos (**8,04%**);
  * **Notas 4.0 e 5.0 $\rightarrow$ Classe 0 (Positiva):** 16.910 exemplos (**91,96%**);
  * **Nota 3.0 $\rightarrow$ Excluída (1.612 casos neutros):** Evita fronteiras cinzentas e ambivalência semântica.
* **Base Elegível Final:** **18.388 avaliações** (proporção severa de ~11,4 positivas para cada negativa).

---

## 3. Rigor Metodológico: Prevenção Estrita de Vazamento

* **O Maior Risco de Penalização na Disciplina:** Vazamento de dados (*Data Leakage*).

* **Protocolo Adotado:**
  1. **Divisão Estratificada Prévia:** 80% Treino (14.710) e 20% Teste (3.678) com semente fixa (`random_state=42`).
  2. **Teste 100% Congelado:** O conjunto de teste nunca foi visto por nenhuma etapa de vocabulário, $IDF$, seleção de atributos ou escolha de hiperparâmetros.
  3. **Tudo em `Pipeline` do Scikit-Learn:** O método `fit` de cada transformador rodou estritamente sobre os folds de treino.

* **Métrica Orientadora:**
  * Proibição expressa de usar apenas acurácia (um chute aleatório teria 92% de acurácia).
  * Foco absoluto no **F1-Score da Classe Negativa (1)**, **Recall** e **PR-AUC**.

---

## 4. Mineração de Texto e TF-IDF

* **Concatenação de Texto:** `review_full = title + " " + text` (títulos curtos contêm forte carga emocional).
* **Configurações do `TfidfVectorizer` dentro do Pipeline:**
  * **Unigramas e Bigramas (`ngram_range=(1, 2)`):** Essencial para capturar negações explícitas (*"not good"*, *"waste time"*);
  * **Filtros de Frequência:** `min_df=3` (descarta erros de digitação) e `max_df=0.90` (elimina termos onipresentes como *"book"*);
  * **Escala Sublinear (`sublinear_tf=True`):** $1 + \log(TF)$, suavizando resenhas longas;
  * **Vocabulário:** 5.000 n-grams mais discriminatórios.

---

## 5. Modelagem: Classificadores Clássicos da Disciplina

* **Comparação Obrigatória (Etapa 3):** Dois modelos citados no PDF com fundamentos matemáticos distintos, ajustados com `class_weight='balanced'`:

| Modelo | Fundamentação | F1-Score Médio (5-Fold CV) | Desvio Padrão ($\sigma$) |
| :--- | :--- | :---: | :---: |
| **Linear SVM (`LinearSVC`)** | Margem máxima (*Hinge Loss*) | **0,6874** | $\pm 0,0189$ |
| **Regressão Logística** | Máxima Verossimilhança (*Log-Loss*) | 0,6851 | $\pm 0,0142$ |

* **Classificador Campeão:** **Linear SVM** ($C=0.5$), consagrado na literatura para matrizes esparsas de texto em alta dimensão.

---

## 6. Comparação Obrigatória: Seleção vs. Redução de Atributos

* **Etapa 4 do Regulamento:** Comparar 3 versões no classificador campeão (Linear SVM):

| Versão | Técnica | F1-Score (CV) | Retenção | Análise Técnica |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | Sem seleção (5.000 termos) | **0,6874** | **100%** | **Vencedor:** Preserva a rica cauda longa de vocabulário insatisfatório. |
| **Filter** | `SelectKBest` ($\chi^2$, 1.500 termos) | 0,6472 | 94,2% | Retém >94% do F1 cortando 70% das variáveis. |
| **Embedded** | `SelectFromModel` (Lasso L1) | 0,6614 | 96,2% | Regularização L1 zera termos irrelevantes. |
| **Redução PCA** | `TruncatedSVD` (100 comp.) | 0,4862 | 70,7% | **Pior versão:** Queda acentuada de quase 30%. |

* **Por que o SVD/PCA degradou?** Em NLP esparso, palavras raras e pontuais (*"waste"*, *"horrible"*) têm peso crítico. A compressão em 100 vetores densos dilui esse sinal específico no ruído ortogonal.

---

## 7. Avaliação Final no Conjunto de Teste ($N = 3.678$)

* **Avaliação Única no Teste Independente e Intocado:**

| Métrica | Valor Obtido | Contexto de Negócio |
| :--- | :---: | :--- |
| **Recall da Classe Negativa** | **77,70%** | **Detecta quase 8 de cada 10 insatisfações reais.** |
| **Precisão da Classe Negativa** | **64,43%** | 64% das resenhas alertadas são realmente negativas. |
| **F1-Score Negativo** | **0,7044** | Equilíbrio harmônico sólido sob desbalanceamento. |
| **PR-AUC (Precision-Recall)** | **0,7843** | Frente a apenas 0,0805 de um baseline aleatório. |
| **ROC-AUC** | **0,9689** | Excelente separabilidade ordinal das classes. |

* **Matriz de Confusão:** 230 Verdadeiros Positivos (TP) da insatisfação capturados, contra apenas 66 Falsos Negativos (FN).

---

## 8. Diagnóstico de Ruído nos Rótulos e Estudo de Ablação

* **Diagnóstico Léxico com VADER ($|score| \ge 0,60$):**
  * Sinalizou 887 avaliações (4,82% da base) com aparente contradição entre nota e texto (ex.: 1★ com texto elogioso ou 5★ com críticas severas).
  * **Tratadas estritamente como suspeitas para inspeção**, nunca como "erros comprovados".

* **Estudo de Ablação (Remoção apenas no Treino, Teste Comum):**
  * *Modelo com Treino Normal (14.710 ex.):* F1-Score no teste = **0,7044**;
  * *Modelo com Treino Filtrado (13.999 ex.):* F1-Score no teste = **0,6988**.

* **Conclusão Metodológica Sóbria:** O classificador Linear SVM com TF-IDF já possui estabilidade natural contra ruídos pontuais, dispensando descarte artificial de dados.

---

## 9. Termos Mais Preditivos e Interpretação

* **Inspeção dos Coeficientes do Hiperplano SVM:**
  * **Mais Fortes para Avaliação NEGATIVA (Classe 1):**  
    *"waste"*, *"not"*, *"poor"*, *"disappointed"*, *"boring"*, *"return"*, *"waste time"*, *"worst"*, *"terrible"*, *"dont waste"*, *"awful"*.
  * **Mais Fortes para Avaliação POSITIVA (Classe 0):**  
    *"great"*, *"excellent"*, *"loved"*, *"must read"*, *"best"*, *"highly recommend"*, *"wonderful"*, *"easy read"*, *"beautiful"*.

* **Observação Qualitativa:**  
  Bigramas como *"waste time"* e *"dont waste"* receberam pesos expressivos, provando a eficácia dos $n$-grams sobre palavras isoladas.

---

## 10. Conclusão Final e Resposta à Pergunta

<div class="highlight">
<strong>Resposta Conclusiva:</strong> Sim, é perfeitamente possível prever avaliações negativas de livros a partir do texto da resenha.
</div>

* **Principais Entregas:**
  1. Modelo operacional com **77,7% de Recall** e **0,704 de F1-Score** em base com 92% de desbalanceamento;
  2. Protocolo científico sem qualquer vazamento de dados (*data leakage*);
  3. Comprovação teórica e empírica de que seleção de atributos supera redução dimensional densa (PCA/SVD) para dados esparsos de texto;
  4. Código, dados e documentação 100% reprodutíveis no GitHub e executáveis no Google Colab.

---

## 11. Guia de Arguição: Perguntas Rápidas da Banca

1. **Por que removeram a nota 3.0?**  
   *R: Porque a nota 3 reúne prós e contras ambíguos. O objetivo era separar polaridades nítidas de sentimento.*
2. **Por que não usaram acurácia como critério?**  
   *R: Com 92% de avaliações positivas, um classificador cego teria 92% de acurácia e seria inútil para o negócio.*
3. **Por que o Linear SVM venceu a Regressão Logística?**  
   *R: Ambos foram muito bem com class_weight='balanced'. O SVM teve ligeira vantagem (0,6874 vs 0,6851) por maximizar a margem geométrica de separação.*
4. **Por que o PCA/SVD foi o pior método?**  
   *R: Porque TF-IDF depende de palavras negativas pontuais e raras. O SVD mistura essas palavras em 100 componentes densos e dilui o sinal no ruído.*
5. **O ruído de rótulo foi confirmado?**  
   *R: Não, é tratado como suspeita diagnóstica. A ablação provou que o modelo já é robusto ao ruído natural.*
