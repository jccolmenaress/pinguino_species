import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import pickle




st.title('Penguin Classifier')
st.write(" esta app usa 6 valores como entrada para predecir las especies de pinguinos usando un modelo de randomforest")
penguin_df = pd.read_csv("penguins.csv")
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

#enviar un formulario con la accion de un btn
with st.form('user_inputs'):
    island = st.selectbox('Penguin island', options = ["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgeson = 0,0,0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Togerson':
    island_torgeson = 1

sex_female, sex_male = 0,0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream, island_torgeson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
#user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
st.write(f"We predict your penguin is of the {prediction_species} species")

st.subheader(" Prediciendo  la especie de pinguino")
#st.image("feature_importance.png")

st.write("los siguientes histogramas muestran cada variable continua separada por cada especie de pinguino, el eje Y representa los valores de entrada")
fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_length_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("bill length by species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_depth_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("bill depth by species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["flipper_length_mm"],
    hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("flipper_length by species")
st.pyplot(ax)