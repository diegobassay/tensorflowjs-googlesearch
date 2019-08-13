# tensorflowjs-googlesearch
Demonstração de como usar o TensorflowJS para fazer pesquisas no Google usando a melhor predição de uma image obtida através de redes neurais com arquitetura VGG19 (ImageNet).

## Pré-requisitos

* Python = v2.7
* NodeJs = v10.15
* Microsoft Build Tools 2019

## Preparando o projeto
Para instalar as dependências executar o seguinte comando na raiz do projeto:
```
npm i
```
## Para executar o projeto
Depois de executar o passo acima executar o seguinte comando na raiz do projeto:
```
npm start
```
## Para testar
Usar o seguinte endereço:
```
http://localhost:8081/
```

## Para executar
Escolher uma imagem (png, bmp ou jpg), clicar em upload a imagem será submetida ao Tensorflow e a classe com peso mais relevante será usada para obter resultados na pesquisa do Google.