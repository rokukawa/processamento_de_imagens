
# Projeto de Extração e Classificação de Características em Imagens

O projeto será composto pela implementação de script Python para classificação de imagens raio-X dividias em duas classes: covid e normal.

O projeto lida com a base de imagens com três descritores de forma: Hu Moments, Local Binary Pattern e Gray Histogram.




## Equipe

Gustavo Rokukawa Camargo
## Descrição dos Descritores Implementados

O projeto implementa a extração de características de imagens utilizando descritores Hu Moments, Local Binary Pattern e Gray Histogram. 

O Hu Moments é descritor de forma invariantes à escala, rotação e translação. É derivado dos momentos de imagem, que são medidas estatísticas que descrevem características geométricas de uma forma.

O Local Binary Pattern é uma técnica usada para descrever a textura de uma imagem.

O Gray Histogram é uma representação gráfica da distribuição de intensidades de tons de cinza em uma imagem.
## Link

[![GitHub](https://img.shields.io/badge/github-1DA1F2?style=for-the-badge&logo=github&logoColor=white&color=black)](https://github.com/rokukawa/processamento_de_imagens)


## Classificador e acurácia

#### Extracting LBP features 

``` [INFO] Acurácia do classificador no conjunto de teste: 91.18% ```

``` [INFO] Matriz de Confusão: [[18  2][1 13]] ```


#### Extracting Hu Moments features 

``` [INFO] Acurácia do classificador no conjunto de teste: 58.82% ```

``` [INFO] Matriz de Confusão: [[20  0][14  0]] ```
## Instruções de uso

Antes de executar o código, você precisará garantir que tenha o Python 3.10 instalado e, em seguida, instalar as bibliotecas necessárias. Certifique-se de ter imagens disponíveis em seu sistema local ou adaptar o código para baixar as imagens do Kaggle
 - [Kaggle](https://www.kaggle.com/datasets/tarandeep97/covid19-normal-posteroanteriorpa-xrays)


#### Instalação do Python 3.10:

Certifique-se de ter o Python 3.10 instalado em seu sistema. Você pode baixá-lo em: 
 - [Python](https://www.python.org/downloads/)

#### Configuração do Ambiente Virtual:

``` python3 -m venv venv ```

#### Instalação das Bibliotecas Necessárias:

``` pip install opencv-python numpy scikit-learn progress scikit-image ```

Execução do Código:

``` python3 lbp_FeatureExtration.py ``` ou ``` python3 huMoments_FeatureExtration.py ```
