# DESAF-O-caso-1
Desafío para postulación caja los andes.

library(ROSE)
library(xgboost)
library(caret)
library(pROC)
library(MLmetrics)

# LECTURA BBDD
base <- read.csv("C:")

# Convertir a factor para poder aplicar REMUESTREO
base$x3 <- as.factor(base$x3)
base$x4 <- as.factor(base$x4)
base$x5 <- as.factor(base$x5)
base$x6 <- as.factor(base$x6)
base$x7 <- as.factor(base$x7)
base$target <- as.factor(base$target)

# REMUESTREO
base_balanceada <- ROSE(target ~ ., data = base, seed = 123)$data

# Pasar a numérico para aplicar correctamente XGBoost
base_balanceada$x3 <- as.numeric(base_balanceada$x3)
base_balanceada$x4 <- as.numeric(base_balanceada$x4)
base_balanceada$x5 <- as.numeric(base_balanceada$x5)
base_balanceada$x6 <- as.numeric(base_balanceada$x6)
base_balanceada$x7 <- as.numeric(base_balanceada$x7)

set.seed(123)  # SEMILLA PARA REPRODUCIBILIDAD
# VALIDACIÓN Y ENTRENAMIENTO
sep <- createDataPartition(base_balanceada$target, p = 0.8, list = FALSE)
train_base <- base_balanceada[sep, ]
test_base <- base_balanceada[-sep, ]

# CATEGORÍAS Y DATOS (X DATOS; Y CATEGORÍA)
train_x <- as.matrix(train_base[, -ncol(train_base)])
train_y <- train_base$target
test_x <- as.matrix(test_base[, -ncol(test_base)])
test_y <- test_base$target

# CONVERTIR TARGET A NUMÉRICO
train_y <- as.numeric(as.character(train_y))
test_y <- as.numeric(as.character(test_y))

# DEFINICIÓN DE DATOS PARA XGBOOST
train <- xgb.DMatrix(data = train_x, label = train_y)
test <- xgb.DMatrix(data = test_x, label = test_y)

# PARÁMETROS DEL MODELO
parametros <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = 1,
  max_depth = 3,
  eta = 0.03,
  lambda = 2,
  alpha = 1
)

# ENTRENAR MODELO
model <- xgb.train(params = parametros, data = train, nrounds = 50, 
                   watchlist = list(train = train, test = test))

# PREDICCIONES
train_pred <- predict(model, train)
test_pred <- predict(model, test)

# CONVERTIR PROBABILIDADES A CATEGORÍAS
train_pred_ <- ifelse(train_pred > 0.5, 1, 0)
test_pred_ <- ifelse(test_pred > 0.5, 1, 0)

# MÉTRICAS
auc(train_y, train_pred)
auc(test_y, test_pred)
F1_Score(y_pred = test_pred_, y_true = test_y)

