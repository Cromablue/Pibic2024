# Classificação de Pneumopatias por Raios-X com Deep Learning Avançado

Este projeto visa desenvolver um sistema robusto e preciso para a classificação de imagens de raios-X do tórax, identificando diferentes tipos de pneumopatias: **COVID-19**, **Normal**, **Pneumonia Bacteriana** e **Pneumonia Viral**. Para lidar com desafios comuns em datasets médicos, como a escassez de dados e o desbalanceamento de classes, o pipeline integra técnicas avançadas de **geração de imagens (GANs e Modelos de Difusão)** e **classificação com redes neurais convolucionais (CNNs) otimizadas**.

-----

## Bases de Dados Utilizadas

A fundamentação deste projeto reside na utilização de bases de dados públicas e amplamente reconhecidas na literatura científica para a tarefa de classificação de condições pulmonares. A seleção dessas bases foi guiada por um mapeamento sistemático da literatura desenvolvido desenvolvido por mim José Santo de Moura Neto (Bolsista), Rafael da Silva Santos (voluntário) em conjunto ao orientador Dr. Vandermi Jõao da Silva (), que identifica essas fontes como as mais referenciadas em pesquisas de inteligência artificial aplicada ao diagnóstico pulmonar. Essa abordagem visa garantir a **transparência** e a **reprodutibilidade** dos resultados obtidos.

### Fontes dos Dados

Para compor o conjunto de dados abrangente necessário para treinar e avaliar o modelo, foram utilizadas as seguintes bases de dados:

  * **Chest X-Ray Images (Pneumonia) - Kaggle:** Uma coleção rica de imagens de raios-X de tórax, contendo casos classificados como normais, pneumonia viral e pneumonia bacteriana.

  * **COVID-19 Radiography Database - Kaggle:** Essencial para a classe COVID-19, este dataset também inclui imagens de indivíduos normais, com pneumonia viral e com opacidade pulmonar.

  * **QaTa-Cov19 Dataset - Kaggle:** Uma base de dados abrangente que incorpora imagens classificadas como normais, COVID-19, pneumonia viral e pneumonia bacteriana, contribuindo para a diversidade das classes.

  * **covid-chestxray-dataset (IEEE8023/Cohen) - GitHub:** Este dataset é uma fonte primordial para casos de COVID-19, complementado com exemplos de outras síndromes respiratórias, enriquecendo a variabilidade dos dados de COVID-19 e outras pneumonias.

### Justificativa para a Seleção das Bases

A predominância de datasets públicos e a escolha específica destas bases está alinhada com as melhores práticas observadas em trabalhos de pesquisa semelhantes. Conforme destacado no mapeamento bibliográfico:

> “Predominância de datasets públicos: O uso extensivo de datasets publicamente disponíveis (Kaggle, GitHub, NIH, Stanford) promove transparência e reprodutibilidade […]. Compreendem majoritariamente CXR e CT, cobrindo COVID-19, pneumonias diversas, tuberculose e casos normais.”

Esta estratégia não só garante a validação externa dos resultados do projeto, mas também fornece um conjunto de dados diversificado e representativo das condições pulmonares abordadas.

-----

## 1\. Organização e Pré-processamento do Dataset

A base de qualquer projeto de Deep Learning reside na qualidade e organização dos dados. Este projeto começa com uma etapa robusta de preparação do dataset, garantindo que as imagens estejam prontas para o treinamento dos modelos.

### 1.1 Coleta e Estrutura do Dataset

As imagens de raios-X originais, juntamente com as imagens sintéticas geradas, são coletadas e organizadas em uma estrutura de diretórios padronizada. Isso facilita o carregamento e o gerenciamento dos dados pelos modelos de aprendizado de máquina.

  * **Estrutura Esperada:** O dataset final é organizado em três conjuntos principais: `train` (treinamento), `val` (validação) e `test` (teste). Dentro de cada um desses conjuntos, as imagens são categorizadas em subdiretórios correspondentes às suas classes: `covid19`, `normal`, `pneumonia_bacterial` e `pneumonia_viral`.

    ```
    .
    ├── dataset_organizado/
    │   ├── train/
    │   │   ├── covid19/
    │   │   ├── normal/
    │   │   ├── pneumonia_bacterial/
    │   │   └── pneumonia_viral/
    │   ├── val/
    │   │   ├── covid19/
    │   │   └── ...
    │   └── test/
    │       ├── covid19/
    │       └── ...
    └── ...
    ```

### 1.2 Pré-processamento de Imagens

Cada imagem passa por uma série de transformações para garantir uniformidade e otimizar a entrada para as redes neurais.

  * **Redimensionamento e Padding:** Todas as imagens são redimensionadas para um tamanho fixo (e.g., 224x224 pixels), que é o tamanho de entrada padrão para muitas arquiteturas de CNN. Para manter a proporção original da imagem e evitar distorções, um processo de padding (preenchimento com pixels pretos) é aplicado, centralizando a imagem no novo tamanho.
  * **Normalização de Pixels:** As intensidades dos pixels são normalizadas para uma faixa específica (e.g., entre -1 e 1 ou 0 e 1, dependendo da transformação `ToTensor()` e `Normalize()`). Isso ajuda a estabilizar o treinamento da rede.
  * **Ajustes de Contraste e Nitidez:** Para realçar características importantes em imagens médicas, ajustes leves de contraste e nitidez são aplicados, padronizando a qualidade visual das imagens.
  * **Conversão para RGB:** Todas as imagens são convertidas para o formato RGB (3 canais), mesmo que sejam originalmente em escala de cinza (1 canal), para compatibilidade com modelos pré-treinados que esperam 3 canais de entrada.

### 1.3 Data Augmentation (Aumento de Dados)

Para aumentar a diversidade do conjunto de treinamento e reduzir o overfitting, técnicas de aumento de dados são aplicadas dinamicamente durante o carregamento das imagens de treino.

  * **Transformações Aplicadas:** Incluem rotações aleatórias leves, inversões horizontais, cortes aleatórios redimensionados (`RandomResizedCrop`), e ajustes de brilho e contraste (`ColorJitter`).
  * **Albumentations (Preferencial):** O uso da biblioteca `Albumentations` é priorizado devido à sua eficiência e à variedade de transformações que oferece, incluindo algumas mais avançadas como `CLAHE` (para realce de contraste adaptativo) e `GaussNoise` (adição de ruído gaussiano), que podem simular variações em exames reais. Caso `Albumentations` não esteja disponível, um fallback para `torchvision.transforms` é utilizado.

-----

## 2\. Balanceamento de Classes com Geração de Imagens Sintéticas

Um dos maiores desafios em datasets médicos é o desbalanceamento de classes, onde algumas condições (e.g., COVID-19 em estágios iniciais) podem ter muito menos exemplos do que outras (e.g., Normal). Isso pode levar o modelo a aprender um viés para as classes majoritárias. Para mitigar esse problema, foram exploradas e implementadas duas abordagens de geração de imagens: **Redes Adversariais Generativas (GANs)** e **Modelos de Difusão**.

### 2.1 Geração de Imagens com GANs (DCGAN)

As GANs são uma classe de algoritmos de aprendizado de máquina que aprendem a gerar novos dados com as mesmas características do conjunto de dados de treinamento.

  * **Arquitetura DCGAN:** Uma **DCGAN (Deep Convolutional Generative Adversarial Network)** foi implementada. Ela consiste em dois componentes principais:
      * **Gerador (`Generator`):** Uma rede neural que aprende a mapear um vetor de ruído aleatório (vetor latente) para imagens realistas. Ele usa camadas convolucionais transpostas para "desconvoluir" o ruído em uma imagem.
      * **Discriminador (`Discriminator`):** Uma rede neural que aprende a distinguir entre imagens reais (do dataset de treinamento) e imagens falsas (geradas pelo Gerador). Ele usa camadas convolucionais para extrair características da imagem e classificá-la como real ou falsa.
  * **Processo de Treinamento:** O Gerador e o Discriminador são treinados em um jogo de soma zero. O Gerador tenta enganar o Discriminador, enquanto o Discriminador tenta não ser enganado. Através desse processo adversarial, o Gerador aprende a produzir imagens cada vez mais convincentes.
  * **Balanceamento:** Para as classes minoritárias (e.g., `covid19`, `pneumonia_bacterial`, `pneumonia_viral`), uma GAN separada é treinada usando apenas imagens dessa classe. Após o treinamento, o Gerador é utilizado para criar imagens sintéticas adicionais, aumentando o número de amostras nessas classes até atingir um volume desejado, balanceando o dataset.

### 2.2 Geração de Imagens com Modelos de Difusão

Os Modelos de Difusão representam uma abordagem mais recente e poderosa para a geração de imagens, conhecida por sua alta qualidade e estabilidade de treinamento.

  * **Processo de Ruído e Reversão:** Um Modelo de Difusão funciona em duas etapas:
      * **Processo de Difusão (Forward):** Ruído gaussiano é gradualmente adicionado a uma imagem original ao longo de vários "timesteps" (passos de tempo), transformando-a em ruído puro.
      * **Processo de Reversão (Backward/Denoising):** O modelo é treinado para aprender a reverter esse processo, ou seja, a remover o ruído de uma imagem ruidosa para reconstruir a imagem original. Ele aprende a prever o ruído adicionado em cada timestep.
  * **Arquitetura U-Net:** Uma arquitetura **U-Net** é utilizada como o modelo principal para prever o ruído. A U-Net é uma rede simétrica de encoder-decoder com conexões de salto (skip connections) que permitem a passagem de informações de baixa resolução para camadas de alta resolução, essencial para gerar imagens detalhadas.
  * **`DiffusionScheduler`:** Um scheduler é implementado para gerenciar os parâmetros do processo de difusão (e.g., `betas`, `alphas`), que controlam a quantidade de ruído adicionado/removido em cada timestep.
  * **Geração de Imagens:** Para gerar uma nova imagem, o modelo começa com um tensor de ruído gaussiano aleatório e, iterativamente, aplica o processo de denoising ao longo dos timesteps, transformando o ruído em uma imagem coerente e realista.
  * **Vantagens sobre GANs:**
      * **Qualidade de Imagem Superior:** Modelos de Difusão geralmente produzem imagens com maior fotorrealismo e detalhes mais finos.
      * **Estabilidade de Treinamento:** São mais fáceis e estáveis de treinar em comparação com GANs, que podem sofrer de problemas como "mode collapse" (onde o gerador produz pouca variedade de imagens).
      * **Maior Diversidade:** Tendem a gerar uma gama mais ampla e diversa de amostras, o que é crucial para balancear efetivamente um dataset e evitar que o modelo de classificação aprenda apenas variações limitadas.
  * **Balanceamento Aprimorado:** As imagens geradas pelos Modelos de Difusão são então integradas ao dataset, proporcionando um balanceamento de classes mais eficaz e um conjunto de dados de treinamento mais rico para o classificador.

-----

## 3\. Classificação com CNNs Avançadas (DenseNet, ResNet, EfficientNet)

Com um dataset balanceado e pré-processado, o foco se volta para o treinamento do modelo de classificação que realizará o diagnóstico das pneumopatias.

### 3.1 Arquitetura da CNN

O projeto emprega arquiteturas de CNNs de última geração, utilizando a poderosa técnica de **Transfer Learning**.

  * **Transfer Learning:** Em vez de treinar uma CNN do zero (o que exigiria um dataset massivo e muito tempo), modelos pré-treinados em grandes bases de dados (como ImageNet) são utilizados como **backbones**. Esses modelos já aprenderam a extrair características visuais complexas (bordas, texturas, padrões) que são úteis para uma ampla gama de tarefas de visão computacional.
      * **DenseNet:** Uma arquitetura notável que promove a reutilização de características e a propagação eficiente de informações por meio de "conexões densas", onde cada camada se conecta a todas as camadas subsequentes de forma direta. Isso ajuda a mitigar o problema do vanishing-gradient e a usar menos parâmetros.
      * **ResNet50:** Uma arquitetura amplamente utilizada e robusta, conhecida por suas **conexões residuais** que ajudam a treinar redes muito profundas sem degradação do desempenho.
      * **EfficientNet-B3:** Uma família de modelos que otimiza a profundidade, largura e resolução da rede de forma eficiente, resultando em alta precisão com menos parâmetros computacionais.
  * **Classificador Personalizado:** A camada final (ou "cabeça") do modelo pré-treinado é substituída por um classificador personalizado, adaptado para as 4 classes específicas do nosso problema. Este classificador geralmente consiste em:
      * **Camadas Lineares (`nn.Linear`):** Para mapear as características extraídas pelo backbone para as probabilidades de cada classe.
      * **Funções de Ativação (`ReLU`):** Introduzem não-linearidade, permitindo que a rede aprenda padrões complexos.
      * **Normalização em Lote (`BatchNorm1d`):** Estabiliza o treinamento e acelera a convergência, normalizando as saídas das camadas.
      * **Dropout (`nn.Dropout`):** Uma técnica de regularização que desativa aleatoriamente um percentual de neurônios durante o treinamento, prevenindo o overfitting e forçando a rede a aprender características mais robustas.

### 3.2 Processo de Treinamento Otimizado

O treinamento do modelo é cuidadosamente configurado para garantir eficiência e alta performance.

  * **`OptimizedCOVIDDataset`:** Uma classe `Dataset` personalizada que:
      * Carrega imagens de forma eficiente, com validação de arquivos e cache opcional em RAM.
      * Calcula **pesos de classe** para o `CrossEntropyLoss`, dando maior penalidade a erros em classes minoritárias, essencial para datasets desbalanceados.
  * **`OptimizedModelTrainer`:** Uma classe abrangente que gerencia o ciclo de treinamento:
      * **Função de Perda (`nn.CrossEntropyLoss`):** Utilizada com os pesos de classe calculados para lidar com o desbalanceamento.
      * **Otimizador (`AdamW`):** Uma escolha popular e eficaz para o treinamento de redes neurais profundas, com bom desempenho e regularização integrada.
      * **Scheduler de Taxa de Aprendizagem (`ReduceLROnPlateau`):** Monitora uma métrica (e.g., perda de validação) e reduz a taxa de aprendizado quando não há melhoria, permitindo que o modelo refine seus pesos.
      * **Mixed Precision Training (AMP):** Utiliza `torch.cuda.amp.GradScaler` e `autocast` para realizar operações em ponto flutuante de 16 bits (FP16) onde possível. Isso acelera o treinamento em GPUs e reduz o consumo de memória, sem perda significativa de precisão.
      * **Early Stopping:** Interrompe o treinamento se a métrica de validação (e.g., acurácia) não melhorar após um número predefinido de épocas (`patience`), evitando o overfitting e economizando recursos computacionais.
      * **Logging Detalhado:** Registra o progresso do treinamento, perdas, acurácias (geral e por classe) e taxa de aprendizado.
      * **Salvamento de Checkpoints:** O melhor modelo (com base na acurácia de validação) é salvo automaticamente.

### 3.3 Avaliação e Análise de Resultados

Após o treinamento, o modelo é rigorosamente avaliado no conjunto de teste para verificar sua capacidade de generalização para dados não vistos.

  * **Métricas de Desempenho:**
      * **Acurácia Geral:** A proporção de predições corretas.
      * **Relatório de Classificação (`classification_report`):** Fornece métricas detalhadas por classe, incluindo **Precisão**, **Recall** e **F1-Score**. Essas métricas são cruciais em contextos médicos, onde o custo de falsos positivos e falsos negativos pode variar.
      * **Matriz de Confusão (`confusion_matrix`):** Uma tabela que visualiza o desempenho do algoritmo, mostrando as contagens de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos para cada classe. É plotada em versões absoluta e normalizada.
      * **Curvas ROC e AUC (`roc_auc_score`):** Para classificação multiclasse, as curvas ROC (Receiver Operating Characteristic) e a Área Sob a Curva (AUC) são calculadas. Elas avaliam a capacidade do modelo de distinguir entre as classes em diferentes limiares de classificação.
  * **Visualização de Predições:** Amostras aleatórias do conjunto de teste são exibidas com suas predições, rótulos verdadeiros e confiança, permitindo uma inspeção visual do desempenho do modelo.
  * **Análise de Classificações Incorretas:** Uma função dedicada ajuda a identificar e detalhar exemplos onde o modelo cometeu erros, permitindo uma análise mais profunda das falhas e potenciais áreas de melhoria.
  * **Salvamento de Informações do Modelo:** Um arquivo JSON é gerado contendo todos os detalhes da arquitetura do modelo, hiperparâmetros de treinamento, histórico de perdas/acurácias e resultados de avaliação, garantindo a rastreabilidade e reprodutibilidade.

-----

## 4\. Resultados do Treinamento e Avaliação

Nesta seção, apresentamos os resultados obtidos com a arquitetura DenseNet, avaliando o impacto das diferentes estratégias de balanceamento de classes (GANs vs. Modelos de Difusão).

### 4.1 Resultados com DenseNet + Imagens Geradas por GANs

Os resultados a seguir correspondem ao treinamento da DenseNet usando um dataset balanceado com imagens geradas por **GANs**.

  * **Progresso do Treinamento (DenseNet + GANs):**

    ```
    Train Loss: 0.4256, Train Acc: 0.8252
    Val Loss: 2.4591, Val Acc: 0.8834
    Novo melhor modelo salvo! Val Acc: 0.8834
    ------------------------------------------------------------
    Época 9/50
    Train Loss: 0.2938, Train Acc: 0.8840
    Val Loss: 0.2251, Val Acc: 0.9027
    Novo melhor modelo salvo! Val Acc: 0.9027
    ------------------------------------------------------------
    Época 10/50
    Train Loss: 0.2447, Train Acc: 0.9061
    Val Loss: 0.1869, Val Acc: 0.9222
    Novo melhor modelo salvo! Val Acc: 0.9222
    Checkpoint salvo na época 10
    ------------------------------------------------------------
    Época 11/50
    Train Loss: 0.2108, Train Acc: 0.9189
    Val Loss: 0.1801, Val Acc: 0.9295
    Novo melhor modelo salvo! Val Acc: 0.9295
    ------------------------------------------------------------
    Época 12/50
    Train Loss: 0.1908, Train Acc: 0.9259
    Val Loss: 0.1445, Val Acc: 0.9428
    Novo melhor modelo salvo! Val Acc: 0.9428
    ------------------------------------------------------------
    Época 13/50
    Train Loss: 0.1759, Train Acc: 0.9326
    Val Loss: 0.1795, Val Acc: 0.9290
    ------------------------------------------------------------
    Época 14/50
    Train Loss: 0.1679, Train Acc: 0.9352
    Val Loss: 0.2098, Val Acc: 0.9158
    ------------------------------------------------------------
    Época 15/50
    Train Loss: 0.1577, Train Acc: 0.9397
    Val Loss: 0.1842, Val Acc: 0.9263
    Checkpoint salvo na época 15
    ------------------------------------------------------------
    Época 16/50
    Train Loss: 0.1528, Train Acc: 0.9412
    Val Loss: 0.1363, Val Acc: 0.9476
    Novo melhor modelo salvo! Val Acc: 0.9476
    ------------------------------------------------------------
    Época 17/50
    Train Loss: 0.1489, Train Acc: 0.9430
    Val Loss: 0.1549, Val Acc: 0.9397
    ------------------------------------------------------------
    Época 18/50
    Train Loss: 0.1477, Train Acc: 0.9437
    Val Loss: 0.1670, Val Acc: 0.9301
    ------------------------------------------------------------
    Época 19/50
    Train Loss: 0.1360, Train Acc: 0.9476
    Val Loss: 0.2629, Val Acc: 0.9000
    ------------------------------------------------------------
    Época 20/50
    Train Loss: 0.1365, Train Acc: 0.9473
    Val Loss: 0.1087, Val Acc: 0.9574
    Novo melhor modelo salvo! Val Acc: 0.9574
    Checkpoint salvo na época 20
    ------------------------------------------------------------
    Época 21/50
    Train Loss: 0.1283, Train Acc: 0.9512
    Val Loss: 0.1143, Val Acc: 0.9538
    ------------------------------------------------------------
    Época 22/50
    Train Loss: 0.1236, Train Acc: 0.9513
    Val Loss: 0.1152, Val Acc: 0.9535
    ------------------------------------------------------------
    Época 23/50
    Train Loss: 0.1266, Train Acc: 0.9516
    Val Loss: 0.1389, Val Acc: 0.9448
    ------------------------------------------------------------
    Época 24/50
    Train Loss: 0.1339, Train Acc: 0.9487
    Val Loss: 0.1327, Val Acc: 0.9460
    ------------------------------------------------------------
    Época 25/50
    Train Loss: 0.1193, Train Acc: 0.9536
    Val Loss: 0.1303, Val Acc: 0.9475
    Checkpoint salvo na época 25
    ------------------------------------------------------------
    Época 26/50
    Train Loss: 0.1163, Train Acc: 0.9549
    Val Loss: 0.1453, Val Acc: 0.9443
    ------------------------------------------------------------
    Época 27/50
    Train Loss: 0.0976, Train Acc: 0.9617
    Val Loss: 0.1002, Val Acc: 0.9594
    Novo melhor modelo salvo! Val Acc: 0.9594
    ------------------------------------------------------------
    Época 28/50 (interrompida)
    ```

  * **Melhor Acurácia de Validação Alcançada (DenseNet + GANs):** **0.9594 (95.94%)** na Época 27.

  * **Análise:** O modelo demonstrou um aprendizado robusto, com a acurácia de treinamento e validação crescendo consistentemente ao longo das épocas. Atingir quase 96% de acurácia em validação com imagens geradas por GANs é um resultado muito promissor, indicando que as imagens sintéticas contribuíram efetivamente para o desempenho do modelo.

### 4.2 Resultados com DenseNet + Imagens Geradas por Modelos de Difusão

Os resultados a seguir correspondem ao treinamento da DenseNet utilizando um dataset balanceado com imagens geradas por **Modelos de Difusão**.

  * **Distribuição do Dataset (Exemplo de log):**

    ```
    Dataset carregado: 64082 amostras de ./diffusion_dataset_organizado/train
      covid19: 11497 amostras
      normal: 15700 amostras
      pneumonia_bacterial: 11510 amostras
      pneumonia_viral: 11457 amostras
    Dataset carregado: 5432 amostras de ./diffusion_dataset_organizado/val
      covid19: 2284 amostras
      normal: 3140 amostras
      pneumonia_bacterial: 2301 amostras
      pneumonia_viral: 2293 amostras
    Dataset carregado: 5437 amostras de ./diffusion_dataset_organizado/test
      covid19: 2285 amostras
      normal: 3140 amostras
      pneumonia_bacterial: 2301 amostras
      pneumonia_viral: 2293 amostras
    ```

    *Nota: A classe 'normal' já possui um grande número de amostras originalmente, por isso não é aumentada pelas técnicas de geração, enquanto as outras classes minoritárias são balanceadas para um volume similar.*

  * **Pesos das Classes:**

    ```
    Pesos das classes: tensor([0.9998, 1.0008, 0.9974, 0.9988], device='cuda:0')
    ```

    *Nota: Os pesos são próximos de 1.0, o que indica que as classes estão bem balanceadas após a inclusão das imagens geradas por difusão.*

  * **Matriz de Confusão (Normalizada) e Contagens:**
    *(Esta informação é visualizada nas imagens fornecidas e seria incluída aqui em formato textual/gráfico em um relatório completo, mostrando o desempenho por classe: TP, FP, FN, TN. Um exemplo de interpretação para "pneumonia\_viral" na matriz é que **2743** amostras foram corretamente classificadas como Pneumonia Viral, enquanto **3** foram classificadas incorretamente como COVID-19, **13** como Normal e **86** como Pneumonia Bacteriana.)*

    ```
    (Gráfico da Matriz de Confusão será inserido aqui)
    ```

  * **Resultados AUC-ROC (Avaliação Final no Conjunto de Teste):**

    ```
    AUC covid19: 0.9993
    AUC normal: 0.9996
    AUC pneumonia bacterial: 0.9971
    AUC pneumonia viral: 0.9964
    AUC Médio: 0.9978
    Acurácia de validação final: 0.9595
    ```

  * **Melhor Acurácia de Validação (DenseNet + Difusão):** **0.9595 (95.95%)**.

  * **Análise:** Os valores de AUC (Área Sob a Curva ROC) são extremamente altos para todas as classes (próximos de 0.999), e o AUC Médio de **0.9978** é um resultado excepcional. Isso sugere que o modelo tem uma excelente capacidade de distinguir entre as diferentes classes, mesmo em um cenário multiclasse. A acurácia de validação final de **0.9595** é consistente com os resultados de AUC, reforçando a alta performance do modelo quando treinado com dados aumentados por modelos de difusão.

### 4.3 Comparativo e Conclusões Preliminares

Ambas as abordagens de balanceamento de classes (GANs e Modelos de Difusão) combinadas com a arquitetura DenseNet produziram resultados de alta acurácia de validação (aproximadamente 95.9% a 96%). No entanto, a alta pontuação de AUC dos resultados com Modelos de Difusão, em conjunto com a matriz de confusão mais detalhada, sugere que **os Modelos de Difusão podem ter proporcionado imagens sintéticas de maior qualidade e diversidade**, otimizando a capacidade discriminatória do modelo para todas as classes e resultando em um desempenho ligeiramente superior e mais robusto, especialmente na separação das classes.

-----

## Estrutura do Projeto

A organização do código e dos dados segue uma estrutura modular e clara:

```
.
├── dataset_organizado/           # Dataset final, balanceado, pré-processado e dividido
│   ├── train/
│   │   ├── covid19/
│   │   ├── normal/
│   │   ├── pneumonia_bacterial/
│   │   └── pneumonia_viral/
│   ├── val/
│   └── test/
├── dataSetPibic/                 # Dataset original (entrada para GANs/Difusão)
├── generated_covid19/            # Imagens geradas por GAN (se usadas)
├── diffusion_generated_covid19/  # Imagens geradas por Modelos de Difusão (se usadas)
├── models/                       # Modelos treinados (.pth) e checkpoints
│   └── best_covid_model_optimized.pth
├── plots/                        # Gráficos gerados (histórico, matriz de confusão, ROC)
│   ├── training_history_optimized.png
│   ├── confusion_matrix_optimized.png
│   └── roc_curves_optimized.png
├── reports/                      # Relatórios de informações do modelo
│   └── model_info_optimized.json
├── main_script.py                # Script principal que orquestra todo o pipeline
└── README.md                     # Este arquivo
```

-----

## Próximos Passos e Melhorias Contínuas

O projeto atual estabelece uma base sólida para a classificação de pneumopatias. Futuras melhorias e expansões podem incluir:

  * **Avaliação com ResNet50 e EfficientNet-B3:** Treinar e comparar o desempenho da ResNet50 e EfficientNet-B3 usando *ambas* as estratégias de balanceamento (GANs e Difusão) para uma análise comparativa mais completa e identificar qual arquitetura/estratégia de geração é a mais eficaz.
  * **Otimização de Hiperparâmetros:** Realizar uma busca mais exaustiva pelos melhores hiperparâmetros (Learning Rate, Dropout, etc.) para o classificador.
  * **Técnicas de Interpretabilidade:** Implementar métodos como Grad-CAM ou LIME para entender quais regiões das imagens o modelo considera mais importantes para suas decisões, aumentando a confiança e a transparência em um contexto médico.
  * **Validação Cruzada (Cross-Validation):** Para uma avaliação mais robusta do desempenho do modelo em datasets menores.
  * **Desenvolvimento de Interface:** Criar uma interface gráfica (web ou desktop) para facilitar a interação de usuários não técnicos com o modelo treinado.
  * **Integração de Segmentação Pulmonar:** Explorar a segmentação pulmonar como um pré-processamento para focar o modelo apenas nas áreas de interesse.

-----

## 📌 Versão Atual

**v1.0.0** - Pipeline completo com geração de imagens sintéticas, classificação com CNNs modernas, e avaliação detalhada.

-----