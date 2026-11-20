# Classificação de Nódulos Pulmonares em Imagens de TC com CNNs

Benchmark comparativo e estatístico de quatro arquiteturas pré-treinadas de Redes Neurais Convolucionais (**VGG-16, ResNet50, InceptionV3 e MobileNetV2**) aplicadas à classificação binária de nódulos pulmonares em exames de Tomografia Computadorizada (TC).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-1E88E5?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy_%26_Statsmodels-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA_Parallel-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

## 🎯 Contexto e Objetivo

O diagnóstico precoce do câncer de pulmão eleva a taxa de sobrevida em 5 anos de **18% para 56%**, sendo a Tomografia Computadorizada (TC) o exame padrão para triagem e estadiamento. Fundamentado por uma **Revisão Sistemática da Literatura (RSL)** prévia (23 estudos analisados), este trabalho avalia de forma rigorosa o equilíbrio entre **desempenho preditivo** e **custo computacional** das quatro arquiteturas mais consolidadas na oncologia pulmonar, isolando oscilações amostrais inerentes a bases médicas restritas.

---

## 🗂️ Bases de Dados e Estratégias de Balanceamento

Os experimentos foram conduzidos em dois cenários distintos de análise tomográfica, mantendo o particionamento estratificado de **70% Treino, 15% Validação e 15% Teste**:

1. **IQ-OTH/NCCD (Fatias Completas — 512×512 px):**
   * **Composição:** 1.097 slices de 110 pacientes agrupados em classificação binária: *com tumor* (681 imagens: benignos + malignos) vs. *normal* (416 imagens).
   * **Balanceamento:** Aplicação de *Data Augmentation* (rotação entre [-15°, +15°] e espelhamento vertical) **exclusivamente no conjunto de treino** para evitar *data leakage*, atingindo proporção 1:1 (477 positivas / 477 negativas).
   * **Resolução de entrada:** 224×224 px (VGG-16, ResNet50, MobileNetV2) e 299×299 px (InceptionV3).

2. **LUNA16 (Redução de Falsos Positivos — Patches 64×64 px):**
   * **Composição:** Recortes extraídos de 888 exames de TC (1.186 nódulos anotados com consenso ≥ 3 radiologistas).
   * **Balanceamento:** Subamostragem da razão original de **484:1** para **10:1** (1.091 positivos / 10.900 negativos no treino), alinhada à prevalência clínica observada nos estudos brasileiros de rastreamento (BRELT1 e BRELT3).
   * **Resolução de entrada:** 64×64 px (VGG-16, ResNet50, MobileNetV2) e 75×75 px (InceptionV3 — além de um experimento adicional em 299×299 px para análise de custo-benefício).

---

## 🔬 Protocolo Experimental e Otimização

Para garantir reprodutibilidade total diante da natureza estocástica do treinamento profundo:

* **20 Repetições Pareadas:** Cada arquitetura foi executada 20 vezes utilizando sementes fixas (`seed` de `1` a `20`), assegurando que todas as redes fossem treinadas e avaliadas exatamente nas mesmas partições de dados.
* **Otimização Bayesiana (`Optuna` + `GPSampler`):** Em cada rodada, foram executados **50 trials** (limite de 20 épocas por trial) guiados por Processos Gaussianos para minimizar o erro do **F1-Score de validação**.

| Hiperparâmetro | Espaço de Busca |
| :--- | :--- |
| **Learning Rate** | 10<sup>-5</sup> a 10<sup>-2</sup> (escala logarítmica) |
| **Weight Decay** | 10<sup>-5</sup> a 10<sup>-2</sup> (escala logarítmica) |
| **Batch Size** | `64` ou `128` |
| **Camadas Ocultas (MLP)** | `1`, `2` ou `3` |
| **Neurônios por Camada** | `64` a `512` (passos de 64) |
| **Taxa de Dropout** | `0.2` a `0.5` |
| **Fine-Tuning** | `0`, `1`, `2` ou `3` blocos finais descongelados |

* **Treinamento Final:** Re-treinamento com a melhor configuração por até **150 épocas**, aplicando **Early Stopping** (paciência de 20 épocas) e restauração dos melhores pesos para avaliação no conjunto de teste.
* **Infraestrutura e Paralelismo:** Executado em workstation com **Intel Core i9-14900K**, **NVIDIA RTX 4500 Ada (24 GB GDDR6)** e **128 GB DDR5**, utilizando paralelismo de até 4 processos simultâneos na GPU (ganho de ≈ 3,5× no tempo total).

---

## 📊 Resultados Experimentais (Média ± Desvio-Padrão em 20 execuções)

### 1. Dataset IQ-OTH/NCCD
| Arquitetura | Acurácia | F1-Score | AUC-ROC | Tempo Total (20 exec.) |
| :--- | :---: | :---: | :---: | :---: |
| **VGG-16** | **0,9809 ± 0,0104** | **0,9846 ± 0,0085** | **0,9971 ± 0,0024** | 9,46h |
| **ResNet50** | 0,9639 ± 0,0201 | 0,9713 ± 0,0155 | 0,9955 ± 0,0036 | 6,10h |
| **InceptionV3** | 0,9542 ± 0,0161 | 0,9631 ± 0,0128 | 0,9908 ± 0,0061 | 5,60h |
| **MobileNetV2** | 0,8830 ± 0,0307 | 0,9031 ± 0,0250 | 0,9505 ± 0,0279 | **2,02h** |

### 2. Dataset LUNA16
| Arquitetura | Acurácia | F1-Score | AUC-ROC | Tempo Total (20 exec.) |
| :--- | :---: | :---: | :---: | :---: |
| **VGG-16** | **0,9779 ± 0,0029** | **0,8774 ± 0,0156** | **0,9866 ± 0,0033** | 15,99h |
| **ResNet50** | 0,9778 ± 0,0033 | 0,8753 ± 0,0188 | 0,9861 ± 0,0061 | 12,50h |
| **InceptionV3 (75×75)** | 0,9503 ± 0,0029 | 0,6948 ± 0,0187 | 0,9342 ± 0,0109 | 8,52h |
| **InceptionV3 (299×299)*** | 0,9597 ± 0,0043 | 0,7625 ± 0,0252 | 0,9638 ± 0,0047 | 73,29h |
| **MobileNetV2** | 0,9208 ± 0,1198 | 0,6927 ± 0,1862 | 0,9174 ± 0,1260 | **5,14h** |

*\*O upscaling para 299×299 na InceptionV3 aumentou o tempo de máquina em **8,6×** (73,29h vs. 8,52h) para um ganho marginal de +0,0677 no F1-Score, sendo considerado inviável e mantida a resolução 75×75 na inferência estatística.*

---

## 📈 Validação Estatística (DBC com α = 0,05)

O **F1-Score** foi analisado sob um **Delineamento em Blocos Casualizados (DBC)** (Y<sub>ij</sub> = μ + T<sub>i</sub> + β<sub>j</sub> + ε<sub>ij</sub>), utilizando as 20 sementes como blocos para isolar a variabilidade da divisão amostral:

* **IQ-OTH/NCCD (Abordagem Paramétrica):**
  * Pressupostos atendidos: Normalidade via **Shapiro-Wilk** (*p* = 0,1021) e Homocedasticidade via **Levene** (*p* = 0,4072).
  * **ANOVA em DBC:** Diferença significativa entre arquiteturas (*F* = 102,3306; *p* < 0,0001) e ausência de efeito de bloco (*p* = 0,0958).
  * **Pós-teste Tukey HSD:** **VGG-16 e ResNet50 empataram estatisticamente no topo** (*p* = 0,0513), enquanto a MobileNetV2 foi inferior a todas as demais (*p* < 0,0001).

* **LUNA16 (Abordagem Não Paramétrica):**
  * Normalidade violada (**Shapiro-Wilk** *W* = 0,6979; *p* < 0,0001) devido a outliers severos de convergência da MobileNetV2 (F1<sub>mín</sub> = 0,1567; *CV* = 27,58%).
  * **Teste de Friedman:** Diferença altamente significativa entre modelos (χ² = 49,2000; *p* < 0,0001).
  * **Pós-teste Wilcoxon Pareado (correção de Holm):** **VGG-16 e ResNet50 confirmaram equivalência estatística superior** (*p* = 0,6142), superando significativamente InceptionV3 e MobileNetV2 (*p* ≤ 0,0003).

---

## 💡 Principais Conclusões e Análise de Erros

* **Melhor Trade-off Clínico (ResNet50):** Diante do empate estatístico no topo preditivo em ambas as bases e do menor consumo de VRAM e tempo de processamento em relação à VGG-16 (6,10h vs. 9,46h no IQ-OTH/NCCD; 12,50h vs. 15,99h no LUNA16), a **ResNet50** consolida-se como a arquitetura mais equilibrada para sistemas CAD clínicos.
* **Análise Qualitativa de Erros:** Os falsos positivos concentraram-se em cortes transversais de vasos sanguíneos e bordas ósseas curvas densas (devido à ausência de contexto volumétrico 3D nas fatias 2D), enquanto os falsos negativos ocorreram em nódulos de baixo contraste ou morfologia irregular.
* **Trabalhos Futuros:** Fusão de características (*Feature Fusion*) entre VGG-16 e InceptionV3 e extensão para **CNNs 3D** sobre cubos tomográficos.

---

## 📁 Estrutura do Repositório

* `codigos_iq/` — Pipelines de Data Augmentation, otimização bayesiana (Optuna) e treinamento no dataset IQ-OTH/NCCD.
* `codigos_luna16/` — Scripts de extração/subamostragem, otimização e treinamento nos patches do LUNA16.
* `resultados_iq/` — Métricas brutas das 20 execuções, hiperparâmetros calibrados e testes estatísticos (ANOVA/Tukey).
* `resultados_luna16/` — Logs das 20 execuções, métricas consolidadas e testes não paramétricos (Friedman/Wilcoxon).

---

## 📚 Referência

> **ROSA, Paulo Roberto Bomfim.** *Comparação de Arquiteturas de Redes Neurais Convolucionais para Classificação de Nódulos em Imagens Tomográficas.* Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação) — Universidade Estadual do Oeste do Paraná (UNIOESTE), Cascavel, 2026. Orientador: Prof. Dr. André Luiz Brun; Coorientadora: Prof. Dra. Simone Aparecida Miloca.
