### AHR lo tomo de base para armar la Demo con Mariano.
### Era: Aplicar  XGBoost  a los datos a predecir


#limpio la memoria
rm( list=ls() )
gc()


library("xgboost")
library("Matrix")
library("randomForest")  #solo se usa para imputar nulos
library("readr") # Me la pide para importar
library("data.table")
library("dplyr") # Si quiero buscar duplicadosy limpiar.
library("caret") # Para la matriz de comfusion
library("lattice") # Para dividir
library("ggplot2")
library("stringr")

#Parametros de entrada
directorio_trabajo    <-  "C:/Users/grecs/Desktop/DATOS/"   
archivo_entrada       <- "Base para Mariano Greco.csv"
archivo_prediccion    <- "salida_xgboost_prediccion_guido_v12.txt"
#archivo_competencia   <- "salida_xgboost_prediccion_mejores_guido_v12.txt"
#archivo_perfiles      <- "salida_xgboost_prediccion_perfiles_guido_v12.txt"

campo_clase           <- "clase"
valor_clase_positivo  <- "1"
formula               <- formula(paste(campo_clase, "~ ."))
campo_id              <- "ID_REGISTRO"
#campo_tipo            <- "TIPO"

setwd( directorio_trabajo )

#leo el dataset
base <- read_delim(archivo_entrada,";", escape_double = FALSE, trim_ws = TRUE)
# Es un dataframe y tambien dos tipos mas, mejor pasarlo a dataframe.
data<- data.frame(base)
str(data)
nrow(data)
#Ahora si es dataframe.

# Veo cunatos NA hay.
table(is.na(data$Score))
#Hay 1085
#Los reemplazo por 0
data$Score[is.na(data$Score)] <- 0
data$Empleador[is.na(data$Empleador)] <- 0
data$Provincia[is.na(data$Provincia)] <- 0
data$Score[is.na(data$Score)] <- 0
#Pasa el "Tipo de cliente" a categorica 
clase<-data[,4]
########################
for (i in 1:length(clase)){
  if (clase[i]=="Bueno" ) {clase[i]=1} else
    if (clase[i]== "Sin comportamiento aun") {clase[i]=1} else
      if (clase[i]=="Cancelada") {clase[i]=0} else
      {clase[i]=0}}
########################

table(clase)# totales por cada clase#

# agrego la columna con la clase binaria
data_con<- cbind(data, clase)

#Borro las variables que no me interesan
data_con <- data_con[ , c(1,7,9,10,14,15,19,20)]

data_new<-str_extract(data_con$Score, "[^A-Za-z]+")

#data_con$Score<- as.numeric(as.character(data_con$Score))

datanew<- as.numeric(as.character(data_con$Score))
datanew[is.na(datanew)] <- 0
data_con<- cbind(data_con, datanew)
View(data_con)
#### Le saque la columana TIPO asi usamos todo el dataset
#Veo registros duplicados
#table(duplicated(data_con[,-1]))
#Estos 259 lo tendriamos que sacar

#Cambio nombre y sigo con el scrip viejo
dataset_unido  <- data_con
nrow( dataset_unido )
str(dataset_unido$clase)
str(dataset_unido)


#dejo la clase en 0/1
dataset_unido[,campo_clase]  <- as.numeric( dataset_unido[,campo_clase] == valor_clase_positivo  )
# Ya estaba hecho? No, era un factor antes, ahora es numerico


### Aca Marian puso a mano 0 en la columna TIPO hasta el 648 y el resto con 1, lo hizo para forzar a que funcione.
#test_desde   <- which.max(  dataset_unido[  ,campo_tipo ] )
#test_hasta   <- nrow( dataset_unido )
#train_desde  <- 1
#train_hasta  <- test_desde -1 
### Esto sirve para tener datos a evaluar junto a los de entrenamiento del modelo y separarlos.

#Agrego todo 1 al campo Tipo para que entre en el analisis
#y luego lo parto con la libreria

#Acomodo para dejar solo un 0.25 sin analizar 
#dataset_unido$TIPO[1:(2161-540)]<-0
#table(dataset_unido$TIPO)

#Identifico el dataset original del dataset sin clase
#Me dice el maximo valor al cambiar de valor. Es raro porque no le dije nunca que es un 0 o un 1 lo que busca. 
#examen_desde   <- which.max(  dataset_unido[  ,campo_tipo ] )
#examen_hasta   <- nrow( dataset_unido )
#base_desde  <- 1
#base_hasta  <- examen_desde -1

#Armo subconjuntos train y testing del dataset original
#base <- dataset_unido[ base_desde:base_hasta, ]
particion <- createDataPartition(dataset_unido$clase, p = 0.75, list = FALSE)
train <- (dataset_unido[particion,])
test <- (dataset_unido[-particion,])

str(test)
dim(base)
dim(train)
dim(test)

#armo un vector con los numeros de registros que pertenecen a train y test
trainids = rownames(train)
testids = rownames(test)


#armo el dataset SPARSE
formula  <- formula(paste(campo_clase, "~ .-1"))
dataset_unido_sparse  <- sparse.model.matrix( formula, data = dataset_unido )
nrow(  dataset_unido_sparse  )
ncol(  dataset_unido_sparse  )

dataset_unido_sparse[  1 ]



#el dataset para entrenar, que automaticamente descarta los registros con clase nula
dtrain       <- dataset_unido_sparse[ trainids, ]
vector_clase <- dataset_unido[ trainids, campo_clase ]
nrow( dtrain ) 
length( vector_clase )




#EJEMPLO  los valores optimos encontrados en la busqueda bayesiana
#Atencion que estos valores NO son los ganadores en la realidad !!!!
psubsample         <-   1.0
pcolsample_bytree  <-   1.0
peta               <-   0.103
pmin_child_weight  <-   1.0
pmax_depth         <-   6L
palpha             <-   0.582
plambda            <-   0.209
pgamma             <-   0.65
pnround            <-   5000 





set.seed( 899981)
modelo   <- xgboost( 
			data  = dtrain,  
      label = vector_clase,
      missing = 0 ,
 			eta = peta, 
 			subsample = psubsample, 
 			colsample_bytree = pcolsample_bytree, 
 			min_child_weight = pmin_child_weight, 
 			max_depth = pmax_depth,
 			alpha = palpha, 
      lambda = plambda,
      gamma  = pgamma,
 			nround = pnround, 
 			objective="binary:logistic",
			early_stopping_rounds = 100,
			#Sume
			#silent = 0,
			#seed = 899981,
			tree_method = "exact", 
			verbose = TRUE
		#	,booster = "dart",
		#	sample_type = "weighted",
		#	normalize_type = "forest"
                   )

#los datos de test
taplicar    <- dataset_unido_sparse[ testids, ]
dim(taplicar)
nrow(test) 

#aplico el modelo a los datos test
predicciont  <- predict(  modelo, taplicar, missing=0 )
predicciont
#Hago binaria la variable predicha
rpredictiont <- round(predicciont)
#Armo un dataset con las variables originales y la predicha 
nuevodataset = cbind(test,prediccion  = predicciont, pred_clase = rpredictiont)
dim(nuevodataset)
#Matriz de confusion
confu<-confusionMatrix(as.factor(nuevodataset$clase),as.factor(nuevodataset$pred_clase), positive = "1")

plot(confu$table, col=c(11,12))
confu


#view variable importance plot
mat <- xgb.importance (feature_names = colnames(dataset_unido_sparse),model = modelo)
xgb.plot.importance (importance_matrix = mat[1:15], col= "green") 



#grabo en un archivo la prediccion
tx <- data.frame( row.names( taplicar ),  predicciont, rpredictiont, test$clase )
colnames( tx )  <-  c( "ID", "Probabilidad (+)", "Prediccion", "Real" )
write.table( tx, file=archivo_prediccion, row.names=FALSE, col.names=TRUE, quote=FALSE, sep="\t", eol = "\n")

modelo$evaluation_log
plot(predicciont,type="c",  col="green")

mean(predicciont)
rm( list=ls() )
gc()

#quit( save="no" )



