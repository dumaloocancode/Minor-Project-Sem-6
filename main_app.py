import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import base64

st.title("Traffic Flow Prediction")

# this is the trained weights jo MLP regressor mein humne train kiya tha
with open('weights.pkl', 'rb') as f:
    data = pickle.load(f)


#opening the updated data, jisme saare feature values numeric ha
with open('dump_updated.pkl', 'rb') as f:
    dump = pickle.load(f)


features = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month', 'weather_type', 'weather_description']


st.subheader('The Plotted Data Looks Like:')

def ploted_dataset(features, dataset):
    metrics = ['month', 'month_day', 'weekday', 'hour']
    fig = plt.figure(figsize=(8, 4*len(metrics)))
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), 1, i+1)
        ax.plot(dataset.groupby(metric)['traffic_volume'].mean(), '-o')
        ax.set_xlabel(metric)
        ax.set_ylabel("Mean Traffic")
        ax.set_title(f"Traffic Trend by {metric}")
    plt.tight_layout()
    plt.show()
    
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = ploted_dataset(features, dump)
st.pyplot(fig)


#yaha se hum weather type aur description ke texts ko number de rahe hai jisse hume jab 
#prediciton nikalna hai toh numerical values hone chahiye, usme help karega

dict1 = {'Rain': 1, 'Clear': 2, 'Clouds': 3, 'Drizzle': 4, 'Mist': 5, 'Haze': 6, 'Fog': 7, 'Snow': 8, 'Thunderstorm': 9, 'Smoke': 10, 'Squall': 11}


dict2 = {
    'light rain': 1,
    'sky is clear': 2,
    'broken clouds': 3,
    'overcast clouds': 4,
    'drizzle': 5,
    'mist': 6,
    'haze': 7,
    'fog': 8,
    'light snow': 9,
    'thunderstorm': 10,
    'heavy snow': 11,
    'Sky is Clear': 12,
    'heavy intensity rain': 13,
    'moderate rain': 14,
    'scattered clouds': 15,
    'few clouds': 16,
    'very heavy rain': 17,
    'light intensity drizzle': 18,
    'thunderstorm with heavy rain': 19,
    'snow': 20,
    'proximity thunderstorm': 21,
    'proximity thunderstorm with rain': 22,
    'thunderstorm with light rain': 23,
    'proximity shower rain': 24,
    'thunderstorm with rain': 25,
    'heavy intensity drizzle': 26,
    'thunderstorm with drizzle': 27,
    'smoke': 28,
    'sleet': 29,
    'light rain and snow': 30,
    'thunderstorm with light drizzle': 31,
    'proximity thunderstorm with drizzle': 32,
    'SQUALLS': 33,
    'shower drizzle': 34,
    'freezing rain': 35,
    'shower snow': 36
}




X = dump[features]
target = ['traffic_volume']
Y = dump[target]
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y).flatten()

with st.form("Take Input parameters : ", border=True):

    
    col1, col2, col3 = st.columns(3)

    with col1:
        is_holiday = st.number_input("Is it a holiday", value=None, placeholder="0 for No, 1 for Yes")
        temperature = st.number_input("What is the temperature", value=None, placeholder="ranging from 0 to 47")
        weekday = st.number_input("What day is it today", value=None, placeholder="kaunsa din hai hafte ka")
    with col2:
        hour = st.number_input("Hour of the day?", value=None, placeholder="in the format of 24h")
        month_day = st.number_input("Date?", value=None, placeholder="what is the date")
        year = st.number_input("Year?", value=None, placeholder="Enter the year")
    with col3:
        month = st.number_input("Month?", value=None, placeholder="Enter the month")

        weather_type_value = option = st.selectbox(
            'What is the current weather?',
            ('Rain', 'Clear', 'Clouds', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Snow', 'Thunderstorm', 'Smoke', 'Squall'))
        weather_description_value = option = st.selectbox(
            'How\'d you describe the weather?',
            ('light rain', 'sky is clear', 'broken clouds', 'overcast clouds', 
            'drizzle', 'mist', 'haze', 'fog', 'light snow', 'thunderstorm', 'heavy snow', 
            'Sky is Clear', 'heavy intensity rain', 'moderate rain', 'scattered clouds', 'few clouds', 
            'very heavy rain', 'light intensity drizzle', 'thunderstorm with heavy rain', 'snow', 'proximity thunderstorm', 
            'proximity thunderstorm with rain', 'thunderstorm with light rain', 'proximity shower rain', 'thunderstorm with rain',
            'heavy intensity drizzle', 'thunderstorm with drizzle', 'smoke', 'sleet', 'light rain and snow', 'thunderstorm with light drizzle', 
            'proximity thunderstorm with drizzle', 'SQUALLS', 'shower drizzle', 'freezing rain', 'shower snow'))

    weather_type_input = dict1.get(weather_type_value)
    weather_description_input = dict2.get(weather_description_value)
    weather_type_input = dict1.get(weather_type_value)
    weather_description_input = dict2.get(weather_description_value)
    # submitted = st.form_submit_button("Submit")
    predict_karo = st.form_submit_button("Predict the traffic")
    test_vector = [is_holiday, temperature, weekday, hour, month_day, year, month, weather_type_input, weather_description_input]
   
    test_vector1 = []

    for i in test_vector:
        test_vector1.append(int(i))
    

    if predict_karo:
       
    #    st.write("Data is submitted, please wait")
    
       #call this function
       def result_final():
    
        unseen_data = x_scaler.transform([test_vector1])

        results = data.predict(unseen_data)
    
        predicted = y_scaler.inverse_transform([results])

        # st.text(predicted)
    

        if predicted<=1300:
            st.write("No Traffic at all, enjoy your journey. Be safe :) ")
        elif predicted > 1300 and predicted <= 1700:
            st.write("Seems a little bit busy, but you'll get there :P ")
        elif predicted > 1700 and predicted <= 3500:
            st.write("Might as well carry your headphones or keep your playlist ready, heavy traffic ahead :( ")
        else:
            st.write("Whoops! Consider cancelling your plans today, or delay it a bit; it's a roadblock")

       result_final()    

with open('metric_dump.pkl', 'rb') as f:
    metric_dict = pickle.load(f)

st.write('The Root Mean Squared Error for the following model is:', metric_dict['Root Mean Squared Error'])
st.write('The Mean Squared Error for the following model is:', metric_dict['Mean Squared Error'])

st.markdown(
    "<h1>Traffic Prediction<sup style='font-size:.8em;'>©</sup> done by: <br>Parthsarathi<br>Shivam<br>Rajnish<br>Abhijeet<br>under the guidance of <br>Ms Sonali Kapoor</h1>",
    unsafe_allow_html=True,
)
