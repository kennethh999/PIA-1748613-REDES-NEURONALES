# NEURAL_NETWORK_NEWS.PY KENNETH ESPARZA RAMOS 1748613
#IMPORTANDO LIBRERIAS NECESARIAS
import pandas as pd 
import os
from pandas import Series
import numpy as numpy
import matplotlib.pyplot as pyplot
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import csv


######################################################## Imprecion de MENU ###########################
print("\t\t PIA REDES NEURONALES MARTES ")
print("\t\tKENNNETH SAJID ESPARZA RAMOS 1748613\n ")
print("===MENU DE OPCIONES PARA APRENDIZAJE SUPERVISADO=== \n")
print("0.-INSTRUCCIONES \n ")
print("1.-GRAFICAS DE LA BASE DE DATOS \n ")
print("2.-ENTRENAR LA RED NEURONAL \n ")
print("3.-REALIZAR UNA PREDICCIÓN \n\n")
print("4.-Salir\n\n")

opcionMenu = input("Inserta un numero valor ->")
#Damos Las Instrucciones del programa################################################################
if opcionMenu=="0":
	print("La base de datos real se encuentra en: haberman.csv \n\n")
	print("En caso de que se quieran introducir datos para su predicción se debe usar: Nuevos.csv\n")
	print("y los resultados apareceran en: predict.csv \n\n")
	print("La base de datos tiene como objetivo almacenar los resultados de personas que")
	print("sobrevivieron a una operación de\nextraccion de cáncer de mama entre los años 1958 y 1969.\n")
	print("Ésta cuenta con 4 columnas:\n   Edad del paciente.\n   Año de la operación.\n   Número de tumores adicionales encontrados. \n     1.-El paciente sobrevivió mas de 5 años. \n     2.-El paciente murió antes de 5 años.")  
	print("")
	print("")
   


# ANALISIS Y GRAFICADO DE DATOS ########################################################
elif opcionMenu=="1":
	#Mostramos 4 gráficas 2x2=4
	fig, axes = pyplot.subplots(nrows=2, ncols=2) 
	#abrimos el archivo .csv y hacemos el cambio de codificacion para que no cause problemas
	data=pd.read_csv('haberman.csv', encoding = "ISO-8859-1") 

	######## HACEMOS LA GRAFICA DE SOBREVIVENCIA ########
    #Leemos la columna survival
	dfsurvival= pd.DataFrame(data, columns=['SURVIVAL']) 
	#realizamos una lista con la veces que se repiten los datos
	count = dfsurvival.pivot_table(index=['SURVIVAL'], aggfunc='size') 
    #creamos una grafica con 'x' , 'y,' y la ponemos en la casilla 0,0  del todo conjunto de graficas
	ax=count.plot.bar(x='SURVIVAL', y='count', ax=axes[1,1]) 

    ######## HACEMOS LA GRAFICA DE EDAD DE PACIENTES ##########
	#Leemos la columna edad
	dfAge = pd.DataFrame(data, columns=['AGE'])
	#realizamos una lista con la veces que se repiten los datos						
	countAge = dfAge.pivot_table(index=['AGE'], aggfunc='size')
	#creamos una grafica con 'x' , 'y,' y la ponemos en la casilla 0,1  del todo conjunto de graficas
	axAge=countAge.plot(x='dfAge', y='countAge', ax=axes[1,0])

    ######## HACEMOS LA GRAFICA AÑO EN QUE SE REALIZO LA OPERACIÓN ########
	#Leemos la columna operacion
	dfOPERATION = pd.DataFrame(data, columns=['OPERATION']) 
	#realizamos una lista con la veces que se repiten los datos	                
	countOPERATION = dfOPERATION.pivot_table(index=['OPERATION'], aggfunc='size')
	#creamos una grafica con 'x' , 'y,' y la ponemos en la casilla 1,0  del todo conjunto de graficas
	axOPERATION=countOPERATION.plot(x='dfOPERATION', y='countOPERATION', ax=axes[0,0])

   ######## HACEMOS LA GRAFICA NUMERO DE TUMORES ADICIONALES ENCONTRADOS ########
   #Leemos la columna tumores
	dfNodes = pd.DataFrame(data, columns=['NODES'])
	#realizamos una lista con la veces que se repiten los datos	                      
	countNodes = dfNodes.pivot_table(index=['NODES'], aggfunc='size')
	#creamos una grafica con 'x' , 'y,' y la ponemos en la casilla 1,1  del todo conjunto de graficas
	axNodes=countNodes.plot(x='dfNodes', y='countNodes', ax=axes[0,1])

	pyplot.show() #mostrar graficas

# ENTRENAMIENTO DE LA RED NEURONAL ###################################################################
elif opcionMenu=="2":
	#X = input
	X=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(0,1,2), skiprows=1)
	#Y = output columna D en haberman.csv
	Y=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(3), skiprows=1, dtype='str') 
    # se divide para pronosticar el 30% de la base de datos
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=3) 
    #Entrenamos con 4 neuronas y 3 capas y con el metodo de newton
	clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(4, ), random_state=1, verbose=True) 
	clf.fit(X_train, Y_train) 

    #Realizamos las predicciones sobre el 30% que se dividió de la base de datos.
	predicted = clf.predict(X_test) 
	#Calculamos el error               
	error= 1-accuracy_score(Y_test, predicted)		
    #imprimimos el error
	print('\nError en el conjunto de prueba')
	print(error)
    #guardamos el modelo con pickle
	filename = 'model.sav'								
	pickle.dump(clf, open(filename, 'wb'))
	print('\n\n Modelo gruardado en "model.sav"')

# PREDICCION ###################################################################
elif opcionMenu=="3":

	 #Cargamos el modelo previo que guardamos
	loaded_model = pickle.load(open('model.sav', 'rb'))
     #X=input sacamos de los nuevos candidatos para la prediccion que son años,edad y nodos
	X=numpy.loadtxt('Nuevospacientes.csv', delimiter=',', usecols=(0,1,2), skiprows=1) 
	#Y=output del modelo de aberman.csv de supervivencia
	Y=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(3), skiprows=1, dtype='str')
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.38, random_state=3)

    #Creamos las predicciones de Y que son las supervivencias
	DecisionTreeClassifierModel = DecisionTreeClassifier()  
	DecisionTreeClassifierModel.fit(X_train, Y_train)
	Y_Pred = DecisionTreeClassifierModel.predict(X_test)
	#Imprimimos las predicciones que se nos dan
	print("\n Los datos pronosticados son:")              
	print(Y_Pred)
	print("Los cuales se almacenan en predict.csv")

    #guardamos las preddiciones en predict.csv para poder acceder al archivo posteriormente
	f = open('predict.csv', 'w',newline='')              

	with f:
	    writer = csv.writer(f)
	    writer.writerows(Y_Pred)

	predicted = loaded_model.predict(X_test)			

	Y_test = numpy.loadtxt('predict.csv', encoding='ISO-8859-1', delimiter=',', usecols=(0), dtype='str')
	error= 1-accuracy_score(Y_test, predicted)
    #imprimimos el error
	print('\nError esperado en la prueba:')
	print(error)


# TERMINA EL PROGRAMA ###################################################################
else:
	DoNothing=0
#Kenneth Sajid Esparza Ramos 1748613	