# Import all of the dependenciesR
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

from googletrans import Translator
from moviepy.editor import VideoFileClip

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')



# Translation Code
from googletrans import Translator

def translate_to_language(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Example usage for translation to multiple languages:
def translate_english_to_multiple_languages(english_text):
    kannada_translation = translate_to_language(english_text, 'kn')
    tamil_translation = translate_to_language(english_text, 'ta')
    telugu_translation = translate_to_language(english_text, 'te')
    hindi_translation = translate_to_language(english_text, 'hi')
    return kannada_translation, tamil_translation, telugu_translation, hindi_translation



# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipRead and Language Translation Model')

st.title('Lip Reading Recognition on Deaf and Dumb using Machine Learning') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        output_path = 'converted_video.mp4'
        clip = VideoFileClip(file_path)
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, preset='ultrafast')
        # Rendering inside of the app
        print(file_path, "this was the path of video")
        
        video = open('converted_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('Preprocessing and feature extraction of lips using machine learning for prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        ##---------------------------
        st.info('The output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decoding the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        print(converted_prediction,"this was the text output of lip detection model")
        kannada, tamil, telugu, hindi = translate_english_to_multiple_languages(converted_prediction)
        print(kannada, tamil, telugu, hindi)
        total_text = str(converted_prediction) + str(hindi) + str(kannada) + str(tamil) + str(telugu) 
        st.text("ENGLISH-   " + str(converted_prediction))

        st.info('Translated Output of the Sentence')
        st.text("HINDI-   " + str(hindi))
        st.text("KANNADA-   " + str(kannada))
        st.text("TAMIL-   " + str(tamil))
        st.text("TELUGU-   " + str(telugu))
        
        