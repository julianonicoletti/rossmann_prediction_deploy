# Projeto de Predição de Faturamento da Rede de Farmácias Rossmann
<img src="images\rossmann.jpg">

# RESUMO

Usando um conjunto de dados público disponível no site [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales) e técnicas de análises de dados como ETL (Extraction, Transformation e Load) conseguimos montar um modelo preditivo de machine learning com a proposta de prever o faturamento de cada loja da rede de farmácias Rossmann (1115 lojas ao total) para as próximas 16 semanas. Usamos a linguagem Python para manipular esses dados e suas diversas bibliotecas, como Pandas e Numpy (Data), Inflection (String Manipulation), Boruta(Variable Selection), Scikit-learn (Predictive Data Analysis) e Flask (API e WEB). Ao final foi entregue uma previsão robusta com indicadores bem estabelecidos da faixa de faturamento no período que podem ser acessados tanto pela API quanto por mensagem usando o aplicativo Telegram.

# 1. A Empresa
Segundo a [Wikipedia](https://en.wikipedia.org/wiki/Rossmann_(company)) a empresa Rossmann é uma rede de lojas farmacêuticas da Europa com mais de 56.200 empregadores em mais de 4000 lojas. Em 2019 a rede teve um volume de negócios acima de 10 Bilhões de Euros e está presente em países como Alemanha, Polônia, Hungria, República Tcheca, Turquia, Albania e Espanha.

# 2. O Problema de Negócio
Com o objetivo de prever o faturamento de cada uma das lojas num prazo futuro de 16 semanas, fomos incumbidos de desenvolver um modelo de machine learning. A motivação principal é que os gerentes de cada loja da rede possam ter, com ampla faixa de confiança, de uma previsão do faturamento e planejar possíveis ações futuras para melhoria ou até ampliação das lojas.
## 2.1 Premissas do Negócio
- A previsão considera apenas as lojas que tiveram vendas superiores a 0 nos dados disponíveis.
- Dias em que as lojas estiveram fechadas são excluídos da análise
- A consulta a previsão estará disponível 24 horas, 7 dias da semana.
- Lojas sem informações sobre a distância com competidores próximos terão esse dado fixado em 200.000 metros

# 3. Estratégia para Solução
Adotamos uma estratégia cíclica muito usada na mineração de dados, o CRISP-DM (Cross Industry Standard Process for Data Mining) que nos permite agilidade e eficiência na entrega e agregando valor para a empresa. O método permite que a cada ciclo, seja possível o aprimoramento do modelo e uma entrega mais rápida de um produto minimamente confiável com futuras melhorias.
<img src="images\PDCA.jpg">

# 4. Insights durante a Análise dos Dados
Previamente a uma análise pormenorizada dos dados, foi feito um Brainstormming de hipóteses que poderiam influenciar nas vendas de cada loja para facilitar a análise exploratória.

<img src="images\Vendas_Diarias_das_Lojas_mindmap.png">
Após a criação desse mindmap foi feito a análise de cada uma das hipóteses utilizando os dados e ferramentas do Python e obtivemos essas como principais constatações:

## 4.1 Insight n.01: No Feriado de Natal as lojas vendem mais do que nos outros feriados e em dias regulares
Hipótese Verdadeira. Com os dados podemos concluir que nos feriados o faturamento das lojas é maior e que, especialmente o feriado de Natal é o principal responsável por esse aumento.
<img src="images\Vendas_por_feriado.png">

## 4.2 Insight n.02: O faturamento aumenta com o passar dos anos.
Hipótese Verdadeira. O faturamento geral da rede aumenta com o passar dos anos e isso foi uma variável altamente correlacionada com o valor a se predizer "Sales".
<img src="images\correlacao_vendas_ano.png">

## 4.3 Insight n.03: O tamanho da loja e variedade de produtos não tem relação com o faturamento.
Hipótese Falsa. O faturamento de cada loja tem ligação direta com o tamanho e sortimento de produtos. No gráfico temos 3 tipos, o básico (menor) vende menos que os outros 2 tipos (Extended e Extra).
<img src="images\Vendas_por_feriado.png">

# 5. Trabalho com as variáveis
A partir do dataset original precisamos fazer algumas derivações, criando algumas variáveis e excluindo outras que não teriam imortância para o processo de Machine Learning.
## 5.1 Normalização da Variável Alvo
A variável alvo é 'Sales' (volume de vendas por dia por loja). Quando analisado essa variável em sua distribuição, detecta-se uma assimetria positiva, que, segundo a literatura (Aurélien Geron) sugere-se a aplicação de uma correção com função quadrática ou ainda logarítmica.
<img src="images\reescaling_variavel_alvo.png">
Por inspecão dos gráficos optou-se pela distribuição logaritmica por apresentar uma distribuição mais 'normal'.
## 5.2 Transformação e reescalonamento de variáveis numéricas
O reescalonamento de variáveis numéricas tem por objetivo levar todas essas variáveis a terem uma excursão de fundo de escala semelhantes, por exemplo, entre 0 e 1. Sendo assim avaliou-se que 3 variáveis precisavam de algum tipo de transformação e reescalonamento. O exemplo abaixo mostra o caso da variável competition_distance que por ter sido retirado seus valores 'NaN' e substituido por 200.000 m, apresentaria muito 'outliers'. Nesse caso foi feito um reescalonamento para a forma logarítmica.
<img src="images\competition_distance.png">
## 5.3 Transformação de variáveis categóricas
As variáveis categóricas 'state_holiday', 'store_type' e 'assortment' receberam o seguinte tratamento:

- 'state_holiday': one hot encoding
- 'store_type': label encoding
- 'assortment': ordinal encoding

Por sua vez, as variáveis representativas de dados sazonais, por possuírem natureza circular, receberam codificação trigonométrica6, 7. Essa codificação foi aplicada às variáveis: dia, mês, semana do ano, dia da semana, trimestre, bimestre, e quinzena do ano.

# 6. Modelo de Previsão
Dentro do processo de elaboração do modelo de Machine Learning foram feitos testes de caráter classificatório em alguns dos modelos mais comumentemente usados para modelos de Regressão.
- Average Model
- Linear Regression
- Linear Regression categorizado - LASSO
- Random Forest
- XGBoost 
## 6.1 Escolha da métrica
Na escolha entre os algoritmos, utilizamos a métrica MAPE (Mean Absolute Percentage Error), que é uma medida de erro que expressa a porcentagem média do erro em relação ao valor real. Optamos por essa métrica porque ela é mais compreensível para a equipe de negócios e o CEO, uma vez que fornece uma representação percentual do erro em relação ao valor médio. Dessa forma, é mais fácil interpretar e comunicar o desempenho dos algoritmos selecionados.
Assim, todos os modelos foram testados e obtivemos os seguintes resultados:
<img src="images\metricas_modelos.png">

## 6.2 Escolha do Modelo
O modelo escolhido foi o XGBoost, que apesar de, a depender dos hiperparâmetros escolhidos (ver item 6.3), demora mais, a diferença do resultado do MAPE foi considerável.

## 6.3 Ajuste de Hiperparâmetros
Após a escolha do modelo foi feito um ajuste fino dos hiperparâmetros utilizando Random Search e Bayesian Search(que utiliza um processo de 'aprendizado' após cada teste de hiperparâmetro para decidir qual o próximo a testar). Ao final os parâmetros escolhidos foram esses:

<img src="images\hiperparametros_xgboost.png">

 ## 6.4 Performance do Modelo
 Nos gráficos abaixos temos uma idéia da performance muito boa do modelo, visto que conseguiu reproduzir o padrão de vendas ao longo dos anos estudados. Nos 2 gráficos finais há uma distribuição das predições com base no erro. Repare que a grande maioria esteve ao redor do erro 0, tendo alguns 'outliers' que a predição apresentou alguma dificuldade.

 <img src="images\performance_modelo_graficos.png">
 ![gráficos com os resultados do modelo preditivo](/Ds_em_producao\images\performance_modelo_graficos.png "gráficos com os resultados do modelo preditivo")

 # 7. Aplicação no Telegram
 Foi feito uma integração de uma API com o aplicativo de mensagens Telegram para mostrar a predição de cada uma das 1115 lojas em tempo real e em segundos. Abaixo um GIF da demonstração dessa aplicação:

<img src="images\teste_telegram_bot.gif" width="220">

# 8. Conclusões
Conseguimos atender a demanda do CFO da empresa Rossmann com um modelo preditivo de vendas para as próximas 16 semanas que funciona em tempo real, 24/7 via mensagem do Telegram. Isso facilitará a tomada de decisão tanto de cada gerente regional, de loja quanto do alto escalão da empresa.

# 9. Lições Aprendidas
- A forma CRISP-DM proporciona entregar valor mais rapidamente e também que seja sempre ajustada e melhorada a cada ciclo. Pode-se alterar parâmetros, escolhas de variáveis com base nos resultados anteriores e com no melhor entendimento do negócio.
- O estudo do negócio e de suas particularidades é muito eficaz para saber utilizar as variáveis disponíveis para melhor tratamento e performance do modelo de predição.
- A construção de um bot para o telegram fornece rapidez na entrega dos resultados e permite acesso rápido para qualquer interessado.

# 10. Próximos Passos
- Realizar um estudo aprofundado no modelo e nos dataframes para elucidar a variação na predição para algumas lojas.
- Desenvolver uma aplicação web usando StreamLit para acesso a essa previsão.
- Implementar testes unitários para avaliar a funcionalidade das funções e classes desenvolvidas e garantir estabilidade e segurança.
