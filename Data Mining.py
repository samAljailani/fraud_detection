#loading the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/natek/Downloads/card_transdata.csv')
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('fraud', axis=1), df['fraud'], test_size=0.3, stratify=df['fraud'])
print('training set proportion: ', len(X_train) / (len(df)))
print('testing set proportion: ', len(X_test) / (len(df)))


print(df.head())
print(df.isna().any())

from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
def data_preprocessing(X, show_hist=False):
    X = X.copy()
    cols = ['ratio_to_median_purchase_price', 'distance_from_last_transaction', 'distance_from_home']
    for i, col in enumerate(cols):
        # Adding an offset to the log tranformation to prevent taking the log of zero
        min_value = X[col].min()
        offset = 1 - min_value if min_value <= 0 else 0
        
        if col != 1:
            X[col] = np.log(X[col] + offset) 
        else:
            transformed_data, _ = boxcox(X[col] + offset) 
            X[col] = transformed_data
            
##    for col in cols:
##        scaler = StandardScaler()
##        X[col] = scaler.fit_transform(X[[col]])
##
##        if show_hist:
##            X[col].hist(bins=100, figsize=(6, 4))
##            plt.title(f'Distribution of {col}')
##            plt.xlabel('Value')
##            plt.ylabel('Frequency')
##            plt.show()

    return X
data_preprocessing(X_train, True).describe()
data_preprocessing(X_test, True).describe()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
def build_model(neurons, dropout_rate, activation, optimizerC):
    model = Sequential()
    
    model.add(Dense(units=neurons[0], input_shape=(7,), activation=activation[0]))
    model.add(Dropout(dropout_rate[0]))
    model.add(Dense(units=neurons[1], activation=activation[0]))
    model.add(Dropout(dropout_rate[0]))
    model.add(Dense(units=neurons[2], activation=activation[0]))

    #output layer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizerC[0], metrics=['accuracy', ''])
    return model

params = {
    'neurons': [[512, 256, 128]],
    'dropout_rate': [0.4],
    'activation': ['relu'],
    'optimizer': [Adam(lr=0.001), Adam(lr=0.0001)]
}



model = KerasClassifier(build_fn=build_model, activation = params['activation'],optimizerC=params['optimizer'],dropout_rate=params['dropout_rate'], neurons = params['neurons'],verbose = 0)

search=GridSearchCV(model, params, scoring='accuracy')
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=2, callbacks=[early_stopping])
#model.save('my_model.h5')
res = search.fit(X_train, y_train)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.show()
