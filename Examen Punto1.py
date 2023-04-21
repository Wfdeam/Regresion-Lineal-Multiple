#!/usr/bin/env python
# coding: utf-8

# En el archivo auto.csv se encuentran los siguientes datos de diferentes automóviles:
# 
# Cilindros
# Cilindrada
# Potencia
# Peso
# Aceleración
# Año del coche
# Origen
# Consumo (mpg)
# Las unidades de las características de los automóviles no se encuentran en el sistema internacional. La variable “origen” es un código que identifica al país de origen.
# 
# Crea un modelo con él para que se pueda estimar el consumo de un vehículo a partir del resto de las variables.

# Importar Librerias

# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures


# Lectura de los datos

# In[4]:


auto = pd.read_csv("C:\\Users\\DEAM\\Downloads\\auto.csv")
auto


# Extraccion de variables independientes

# In[5]:


x_0 = auto.loc[:, auto.columns != 'mpg']
x_0


# Extracion de variables numericas

# In[6]:


x_01 = x_0.loc[:, x_0.columns != 'origin']
x_01


# Normalizacion de variables numericas

# In[7]:


scaler = StandardScaler().fit_transform(x_01)
x_01 = pd.DataFrame(scaler, columns=x_01.columns)
x_01


# Extraccion de variable categorica y conversion a dummie

# In[8]:


x_cat = auto[['origin']]
x_cat = pd.get_dummies(x_cat['origin'],prefix='origin')
x_cat


# Union de categoricas con numericas normalizadas

# In[9]:


new_x = pd.concat([x_01,x_cat], axis =1)
new_x

Extraccion de variable dependiente
# In[10]:


y = auto[['mpg']]
y


# Division de conjuntos de entrenamiento y prueba

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(new_x,y, test_size= 0.3, random_state = 42)
var_th = VarianceThreshold(threshold = 0.1)
x_var_train = var_th.fit_transform(x_train)
x_var_test = var_th.transform(x_test)

print("Variables originales ", x_train.shape[1])
print("Variables finales ", x_var_train.shape[1])
#print("Listado de variables ", np.asarray(list(x_var_test))[var_th.get_support()])


# Creacion del modelo

# In[12]:


poly_2 = PolynomialFeatures(degree = 2, include_bias=True)
x_2 = poly_2.fit_transform(x_var_train)
x_2test = poly_2.fit_transform(x_var_test)

modelo_entrenamiento = LinearRegression(fit_intercept=False)

modelo_entrenamiento.fit(x_2,y_train)

predict_train = modelo_entrenamiento.predict(x_2)
predict_test = modelo_entrenamiento.predict(x_2test)

print("R2 train", modelo_entrenamiento.score(x_2,y_train))
print("R2 test", modelo_entrenamiento.score(x_2test,y_test))

