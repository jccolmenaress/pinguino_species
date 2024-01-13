import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
#labels
output = penguin_df['species']
features =penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
#create a one hot encoder
features = pd.get_dummies(features)
print(features.head())
#output = labels codificados, uniques es un objeto que contiene el listado de los valores categoricos originales
output, uniques = pd.factorize(output)
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
#random_state es una semilla esto para que de los mismos resultados, ya que los modelos de ml o deeplearning son modelos estocasticos y puede cambiar los resultados cuando se ejecutan nuevamente
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print(f'our acuraccy score for this model is {score}')
#export the model
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
#export the classes
output_pickle = open('ouput_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()