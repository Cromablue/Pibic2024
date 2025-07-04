# Classificação de Pneumopatias por Raios-X com Deep Learning

Este projeto visa desenvolver um sistema robusto e preciso para a classificação de imagens de raios-X do tórax, identificando diferentes tipos de pneumopatias: COVID-19, Normal, Pneumonia Bacteriana e Pneumonia Viral. Para lidar com desafios comuns em datasets médicos, como o desbalanceamento de classes e a escassez de dados, o pipeline integra técnicas avançadas de geração de imagens (GANs e Modelos de Difusão) e classificação com redes neurais convolucionais (CNNs) otimizadas.

## 1. Organização e Pré-processamento do Dataset

O primeiro passo crucial é a preparação dos dados. O dataset original de raios-X é processado e organizado em uma estrutura padronizada, fundamental para o treinamento de modelos de Deep Learning.

Funcionalidades Chave:

    Estrutura Padronizada: O dataset é organizado em diretórios train, val (validação) e test, com subdiretórios para cada uma das quatro classes (covid19, normal, pneumonia_bacterial, pneumonia_viral).

    Redimensionamento e Padding: Todas as imagens são redimensionadas para um tamanho alvo consistente (224x224 pixels), mantendo a proporção original e adicionando preenchimento (padding) quando necessário. Isso evita distorções e garante que todas as entradas para a CNN tenham o mesmo formato.

    Normalização Básica: Ajustes de contraste e nitidez são aplicados para padronizar a aparência das imagens e realçar características importantes, facilitando a aprendizagem do modelo.

    Data Augmentation (Aumento de Dados): Para enriquecer o conjunto de treinamento e melhorar a capacidade de generalização do modelo, são aplicadas aumentações nas imagens existentes, como rotações leves, ajustes de brilho/contraste e inversões horizontais. Isso cria novas variações das imagens, reduzindo o overfitting.

    Divisão Estratificada: As imagens (originais e geradas) são divididas nos conjuntos de treino, validação e teste (70%/15%/15%) de forma estratificada, garantindo que a proporção de classes seja mantida em cada split, crucial para datasets desbalanceados.

## 2. Balanceamento de Classes com Geração de Imagens

Um dos maiores desafios em datasets médicos é o desbalanceamento de classes, onde algumas classes têm muito mais exemplos do que outras. Isso pode levar o modelo a ter um desempenho ruim nas classes minoritárias. Para combater isso, foram exploradas e implementadas duas abordagens de geração de imagens: Redes Adversariais Generativas (GANs) e Modelos de Difusão.

### 2.1 Geração de Imagens com GANs (DCGAN)

Para as classes com menor número de amostras, foram treinadas GANs separadamente para gerar imagens sintéticas realistas e, assim, balancear o dataset.

    Arquitetura DCGAN: Uma arquitetura DCGAN (Deep Convolutional Generative Adversarial Network) foi implementada com um Gerador e um Discriminador baseados em convoluções.

        Gerador: Aprende a mapear um vetor de ruído aleatório para imagens que se assemelham às imagens de treinamento da classe específica.

        Discriminador: Aprende a distinguir entre imagens reais (do dataset) e imagens falsas (geradas pelo Gerador).

    Treinamento por Classe: Uma GAN é treinada para cada classe minoritária (covid19, pneumonia_bacterial, pneumonia_viral), permitindo que cada Gerador se especialize na geração de imagens da sua respectiva classe.

    Imagens Sintéticas: Após o treinamento, o Gerador é utilizado para criar novas imagens, aumentando o número de amostras nas classes minoritárias até que o dataset esteja mais balanceado.

### 2.2 Geração de Imagens com Modelos de Difusão

Além das GANs, os Modelos de Difusão representam uma abordagem mais recente e poderosa para a geração de imagens de alta qualidade.

    Processo de Difusão: Esses modelos aprendem a reverter um processo de "ruído" gradual. Durante o treinamento, ruído é adicionado às imagens ao longo de vários timesteps. O modelo aprende a remover esse ruído passo a passo para reconstruir a imagem original.

    Geração de Imagens: Na inferência, o modelo começa com ruído aleatório e, ao longo dos timesteps, remove o ruído para gerar imagens novas e realistas.

    Vantagens sobre GANs: Modelos de Difusão são conhecidos por:

        Qualidade Superior: Produzem imagens com detalhes mais finos e maior fotorrealismo.

        Estabilidade de Treinamento: São geralmente mais estáveis para treinar do que as GANs, que podem sofrer de "mode collapse" (gerar pouca diversidade de imagens).

        Maior Diversidade: Geram uma gama mais ampla de imagens, evitando que o modelo "memorize" poucas variações.

    Balanceamento Aprimorado: As imagens geradas pelos modelos de difusão são então incorporadas ao dataset para balancear as classes, complementando ou substituindo as imagens geradas por GANs, dependendo dos resultados de qualidade.

## 3. Classificação com CNNs Avançadas (Densenet/ResNet)

Com um dataset balanceado e pré-processado, o próximo estágio é treinar o modelo de classificação que realmente irá diagnosticar as pneumopatias.

    Arquitetura de Rede Neural Convolucional (CNN): O coração do sistema de classificação é uma CNN, capaz de aprender automaticamente características complexas das imagens de raios-X.

    Transfer Learning: Para acelerar o treinamento e melhorar o desempenho em um dataset médico (que, embora aumentado, ainda pode ser limitado em comparação com datasets de imagens genéricas), utilizamos a técnica de Transfer Learning. Modelos como ResNet50 e EfficientNet-B3, pré-treinados em milhões de imagens (ImageNet), são usados como backbones. As camadas iniciais desses modelos, que aprenderam a detectar características genéricas, são mantidas, e apenas as camadas finais são adaptadas para o nosso problema de 4 classes.

    Classificador Personalizado: Sobre o backbone pré-treinado, um classificador customizado é adicionado. Ele consiste em camadas lineares, ativações ReLU, normalização em lote (BatchNorm1d) e camadas de dropout.

        BatchNorm1d: Ajuda a estabilizar o treinamento e acelerar a convergência.

        Dropout: Regulariza o modelo, "desligando" aleatoriamente alguns neurônios durante o treinamento para evitar overfitting.

    Otimização e Treinamento:

        Função de Perda (CrossEntropyLoss): Utilizada com pesos de classe para dar mais importância aos erros nas classes minoritárias, reforçando o balanceamento feito na etapa de geração de imagens.

        Otimizador (AdamW): Uma variante robusta do Adam que incorpora decaimento de peso para regularização.

        Scheduler de Taxa de Aprendizagem (ReduceLROnPlateau): Ajusta automaticamente a taxa de aprendizado durante o treinamento. Se a perda de validação não melhorar após um certo número de épocas, a taxa de aprendizado é reduzida, permitindo que o modelo refine seus pesos.

        Mixed Precision Training (AMP): Utilizado para acelerar o treinamento em GPUs compatíveis, combinando cálculos de ponto flutuante de 16 e 32 bits, o que economiza memória e tempo de computação.

        Early Stopping: Interrompe o treinamento se a acurácia de validação não melhorar após um número predefinido de épocas, prevenindo o overfitting.

    Métricas de Avaliação Abrangentes: Após o treinamento, o modelo é avaliado usando:

        Acurácia Geral: Percentual de predições corretas.

        Relatório de Classificação: Métricas por classe (Precisão, Recall, F1-Score), essenciais para entender o desempenho em cada tipo de pneumopatia.

        Matriz de Confusão: Visualiza as classificações corretas e incorretas para cada classe.

        Curvas ROC e AUC: Medem a capacidade do modelo de distinguir entre as classes, especialmente importante para cenários médicos.

## Estrutura do Projeto (em resumo)

```
├── dataset_organizado/          # Dataset final organizado e pré-processado
│   ├── train/
│   │   ├── covid19/
│   │   ├── normal/
│   │   ├── pneumonia_bacterial/
│   │   └── pneumonia_viral/
│   ├── val/
│   └── test/
├── dataSetPibic/                 # Dataset original (entrada para GANs/Difusão)
│   ├── train/
│   │   ├── covid19/
│   │   ├── normal/
│   │   ├── pneumonia_bacterial/
│   │   └── pneumonia_viral/
├── generated_covid19/            # Imagens geradas por GAN para COVID-19
├── generated_pneumonia_bacterial/ # Imagens geradas por GAN para Pneumonia Bacteriana
├── generated_pneumonia_viral/    # Imagens geradas por GAN para Pneumonia Viral
├── diffusion_generated_covid19/  # Imagens geradas por Difusão para COVID-19
├── diffusion_generated_pneumonia_bacterial/ # Imagens geradas por Difusão para Pneumonia Bacteriana
├── diffusion_generated_pneumonia_viral/ # Imagens geradas por Difusão para Pneumonia Viral
├── models/                       # Modelos treinados (.pth) e checkpoints
├── plots/                        # Gráficos gerados (histórico, matriz de confusão, ROC)
├── scripts/                      # (Opcional) Scripts auxiliares
│   ├── organize_dataset.py
│   ├── train_gans.py
│   ├── train_diffusion.py
│   └── train_classifier.py
└── README.md                     # Este arquivo
```

## Próximos Passos

O projeto está em um estágio avançado, com um pipeline completo desde a preparação de dados até a avaliação do modelo. Os próximos passos podem incluir:

    Otimização de Hiperparâmetros: Realizar uma busca mais exaustiva pelos melhores hiperparâmetros (Learning Rate, Dropout, etc.) para o classificador.

    Avaliação de Outras Arquiteturas: Experimentar outros backbones pré-treinados (ex: Inception, VGG, Vision Transformers) para comparar o desempenho.

    Testes com Segmentação Pulmonar: Integrar e testar a segmentação pulmonar como um passo de pré-processamento para ver se focar apenas nas regiões de interesse melhora a acurácia.

    Interpretabilidade do Modelo: Utilizar técnicas como Grad-CAM para visualizar quais partes da imagem o modelo está usando para fazer suas predições, o que é crucial em aplicações médicas.

    Integração com Interface: Desenvolver uma interface (web ou desktop) para que médicos e pesquisadores possam usar o modelo facilmente.