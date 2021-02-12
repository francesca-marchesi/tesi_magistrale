import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import numpy as np
#from pythonProject import NN.py
def main():
        #COSTRUZIONE E ORGANIZZAZIONE DEI DATI
    data = pd.read_excel('/Users/Francesca/Desktop/tesi magistrale/originale.xlsx')
    df = pd.DataFrame(data)
        #costruisco df solo con variabili mediche
    df_medico = df[["nome", "dim", "HU_bas", "HU_art", "HU_ven", "HU_tard", "HU_rem", "HU_art/HU_bas", "HU_ven/HU_bas", "HU_tard/HU_bas", "necrosi", "ENH_perif", "ipodenso_pan", "ipodenso_ven", "linfadenopatie", "fat_stranding", "est_duo", "est_bil", "est_lam", "est_ven", "est_art", "inf_organi", "est_ext", "plexus1", "plexus2", "ante_path", "root", "n_plessi", "Age", "Sex", "Ca", "CaStrat", "size_tum", "inv_peri", "inv_vas", "grading", "G1G2", "AJCC8", "LNP", "LNP_LNM", "LNR", "Followup", "Lenght_Followup", "LOCAL_REL", "Time_Local", "DIST_REL", "Time_Dist", "LOCAL_DIST", "Time_LD"]]

        #tolgo le righe che non hanno dei valori per le varibili fondamentali per la nostra analisi
    df_2 = df_medico.drop(df_medico[(df_medico['DIST_REL'].notna() == False)].index)
    df_2 = df_2.drop(df_2[(df_2['Time_Dist'].notna() == False)].index)

        #tolgo tutte le variabili che hanno dei missing values
    for column in df_2:
        if df_2[column].isnull().any() == True: #la sintassi è: df.isnull().any() cerca i missing in tutto il df, qui fissiamo la colonna e con .any() fa si che la risposta sia TRUE/FALSE
            df_2.drop(column, axis=1, inplace=True)

        #aggiungiamo la colonna det target
        #creo un df intermedio che riempio solo con il target che mi serve
    df_prova = pd.DataFrame(columns=['target'])
    df_prova.target=(df_2.Time_Dist>9)
        #creo il df contentente la variabile target binaria
    df_dummies = pd.get_dummies(df_prova['target'])
        #aggiungo solo una delle due colonne al df
    df_2['target'] = df_dummies.iloc[:, 0]
    df_2 = df_2.drop(['LOCAL_REL', 'Time_Local', 'nome', 'DIST_REL', 'Time_Dist'], axis=1)

        #devo eliminare i due dataframe che non uso più?
        #COSTRUZIONE DELLA RETE NEURALE: creo X_train, Y_train, X_val, Y_val; poi costruisco il modello
    X_train, X_val, Y_train, Y_val = train_test_split(df_2.iloc[:,0:10],
                                                      df_2.iloc[:,10],
                                                      test_size=0.33,
                                                      random_state=42)

    model = Sequential()
    model.add(Dense(10, activation ='relu', input_shape = (10,)))
    model.add(Dense(5, activation ='relu'))
    model.add(Dense(1, activation ='sigmoid'))

        # FASE DI ADDESTRAMENTO: utilizzo del metodo compile
    model.compile(optimizer='sgd', loss = tf.keras.losses.BinaryCrossentropy())
    model.summary()
    history = model.fit(X_train,
                        Y_train,
                        batch_size = 32,
                        epochs = 10,
                        shuffle = True,
                        validation_data = (X_val, Y_val)
                        )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()

if __name__ == "__main__":
    main()
