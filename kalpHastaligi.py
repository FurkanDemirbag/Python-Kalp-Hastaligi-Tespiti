# -*- coding: utf-8 -*-
"""
Red Hair Shanks
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#Veri setini yükle
veriSeti=pd.read_csv("kalp_veriseti.csv")

x=veriSeti.iloc[:,0:13].values  #Girdi Katmanlarım
print(x)

y=veriSeti.iloc[:,-1].values  #Çıktı Katmanım
print(y)

#Kategorik veriler düzenlenmiş halde verilmiş, bu adımı uygulamaya gerek yok.

#veriSeti'mizi eğitim ve test olarak bölüyoruz.
#Anacondada herşeyi güncelleyince cross_validation yerine model_selection gelmiş.
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#Özellik ölçeklendirme yapıyoruz.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Aşama 2 ; Yapay Sinir Ağı İşlemleri

#Yapay Sinir ağını başlat
siniflandirici=Sequential()

#Girdi Katmanım 13 oldugundan input_dim'e 13 verdim , nöronada 6 verdim

#1.Gizli Katman
siniflandirici.add(Dense(activation="relu" , input_dim=13,units=6,kernel_initializer="uniform"))

#2.Gizli Katman
siniflandirici.add(Dense(activation="relu",units=6))

#Çıktı Katmanı
siniflandirici.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))

#Yapay Sinir ağını çalıştır     
siniflandirici.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
#Ağı Eğit
#Batch_size = 2 nin katları olmazsa ani düşüşler yaşanabilir.
siniflandirici.fit(x_train,y_train,batch_size=10,epochs=300)

#Test Aşaması
y_tahmin=siniflandirici.predict(x_test)
print("Accuracy: %2.f%%" %(y_tahmin[1]*100))
y_tahmin=(y_tahmin>0.5)

#Accuracy Hesaplama
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_tahmin)









