# Projeto de Cross-Validation

<center><img alt="Colaboratory logo" width="50%" src="https://miro.medium.com/max/720/0*XJRuatmI2xNAeo3G"></center>

No fim de um projeto de machine learning sempre há uma dúvida sobre a performance do modelo estabelecido. Para isso é necessário usar as métricas para essa avaliação. Na biblioteca scikit-learn é encontrada uma variedade de métricas para uso. Essas métricas são utilizadas após o uso do modelo nos dados de testes.

No momento após uma análise dos dados e antes do modelo é realizada uma separação dos dados. Os dados são separados em dados de treino, esses usados para o treinamento do modelo, e dados de testes usados para avaliação do modelo. É necessário adotar a filosofia de que os dados de teste apenas podem ter contato com o modelo no fim.

Cross-validation é uma técnica realizada para avaliação do desempenho do modelo. A diferença está que ele avalia o modelo particionando os dados de treino. Ele participa dos dados de treino em dados de treino e validação. Utilizando-se do método K-fold os dados de treino são divididos em subconjuntos. Os dados são divididos em k subconjuntos que a cada iteração um subconjunto é usado para avaliação. Assim é garantido que cada subconjunto seja avaliado.

O problema aqui abordado será a classificação de medicamentos. Aqui vamos aplicar três modelos de classificadores e avaliá-los por Cross-Validation. Os modelos são: RandomForest, K-neighbors e LogisticRegression. A patir dos resultados decidimos qual o melhor modelo para ser usado nos dados de testes.
