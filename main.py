
from telegram import (Bot, ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)
import requests
import re
import nltk
import os
import numpy as np
import random
from PIL import Image
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import subprocess
import time
import cv2
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint




#modfinal=load_model("C:/Users/jhonp/speechmod/CNN_FINAL-20-0.798.h5")

#if not modfinal:
#    a=1
#else:
#    a=0
    

classes=['cat', 'dog', 'eight', 'five', 'four', 'nine', 'no', 'off', 'on', 'one', 'seven', 'six', 'stop', 'three', 'tree', 'two', 'yes', 'zero', '_background_noise_']
DIR_TEST='C:/Users/jhonp/imagebot/'


def get_image(DIR_TEST):
    paths=os.listdir(os.path.dirname(DIR_TEST))
    random.choice(paths)       
    return(DIR_TEST+random.choice(paths))

############################################################AUDIO HANDLER FUNCTIONS######################################################
#########################################################################################################################################

###FUNCTION TO READ WAV FILES AND PLOT SPECTOGRAM CREDIT TO: https://www.kaggle.com/davids1992/speech-representation-and-data-exploration



def pred_voice(vpath):
    
        
    def log_specgram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    
    def oggtospec(src_filename = "D:/Data Science/test.ogg",dest_filename = "C:/Users/jhonp/test2w.wav",img_size=100):

        img_array=[]
    
        process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
        output_subdir="C:/Users/jhonp"
        im="test2w.wav"
        rate,data=wavfile.read(output_subdir+"/"+im)
        _,_,specgram=log_specgram(data,rate)
        path='%s.png' % (output_subdir+"\\"+im.split(".wav")[0])
        plt.imsave(path, cv2.resize(specgram,(img_size,img_size)))
        os.remove(dest_filename)     ###delete wav file to avoid conflicts
        os.remove(src_filename)
        
        ##read specgram
        
        img=Image.open(path).convert('L')
        img_ar=np.asarray(img)/255
        #plt.imshow(img)
        img_array.append(img_ar)
        os.remove(path)
                        
        return (img_array)

 
    img_test=oggtospec(src_filename = vpath,dest_filename = "C:/Users/jhonp/test2w.wav",img_size=100)
    prediction=np.argmax(modfinal.predict(np.array(img_test).reshape(-1,100,100,1)),axis=1)
    classp=classes[prediction[0]]
    #os.remove(vpath)
    return(classp)
    

##################################################################################################################################

def greeting(sentence):

    GREETING_INPUTS = ("hola", "hi", "hablame", "tnces", "saludos","saludo")
    GREETING_RESPONSES = ["Hola", "Buen día", "Hey :)", "hi", "hello","estoy muy bien, qué deseas saber", "Hola, pregúntame algo"]    
   
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        else:
            return "NA"


f=open('C:/Users/jhonp/botcorpus.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()

#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words("spanish"))
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]  ###biggest cosine similarity from the matrix excluding user_response itself 
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"No puedo contestar tu pregunta"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url

def req_pic(user_response):
    
    activators=('foto','picture','pack','imagen')    
    user_response.split()
    
    for word in user_response.split():
        if word.lower() in activators:
            return get_image(DIR_TEST)
        else:
            return "NA"


def dog(bot, update):
    url = get_url()
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=url)

def foto(bot, update):
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=open(get_image(DIR_TEST),"rb"),timeout=100)
    
def cat(bot, update):
    catimg=requests.get('http://aws.random.cat/meow').json()
    url = catimg['file']
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=url)

def hi(bot,update):
    salute="hi, how you doing?"
    chat_id = update.message.chat_id
    bot.send_message(chat_id=chat_id, text=salute)

def modtest(bot,update):
    
    img=Image.open("D:/Data Science/spectograms/eight/00b01445_nohash_0.png").convert('L')
    img_ar=np.asarray(img)/255
    img_ar=img_ar.reshape(-1,100,100,1)
    prtest=np.argmax(modfinal.predict(img_ar),axis=1)    
    chat_id = update.message.chat_id
    if classes[prtest[0]]=='eight':
        bot.send_message(chat_id=chat_id, text="verd",timeout=500)
    else:
        bot.send_message(chat_id=chat_id, text="error",timeout=500)
            
    

def responder(bot,update):
    user_response=update.message.text
    
        
    if (greeting(user_response)=="NA"):
        update.message.reply_text(response(user_response),reply_markup=ReplyKeyboardRemove())
        sent_tokens.remove(user_response)
    
#    elif (greeting(user_response)=="NA" & req_pic(user_response)!="NA"):
#        chat_id = update.message.chat_id
#        bot.send_photo(chat_id=chat_id, photo=req_pic(user_response))
    
    else:
        update.message.reply_text(greeting(user_response),reply_markup=ReplyKeyboardRemove())
# =============================================================================
#     
# def gender(update, context):
#     user = update.message.from_user
#     logger.info("Gender of %s: %s", user.first_name, update.message.text)
#     update.message.reply_text('I see! Please send me a photo of yourself, '
#                               'so I know what you look like, or send /skip if you don\'t want to.',
#                               reply_markup=ReplyKeyboardRemove())
# =============================================================================


def voicemsg(bot, update):
    file = bot.getFile(update.message.voice.file_id)
    dest="C:/Users/jhonp/"
    file.download("{}vsent.ogg".format(dest))
    #prec='you said {}'.format("{}vsent.ogg".format(dest))
    prec=pred_voice("{}vsent.ogg".format(dest))
    update.message.reply_text(prec,reply_markup=ReplyKeyboardRemove())
        
    
# =============================================================================
#     try:
#         prec='you said ' + pred_voice(dest+"vsent.ogg")
#         update.message.reply_text(prec,reply_markup=ReplyKeyboardRemove())
#         
#     except Exception as e:
#         update.message.reply_text("error",reply_markup=ReplyKeyboardRemove())
#         
# =============================================================================


def echo(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    
def start(bot, update):
    chat_id = update.message.chat_id
    howare="Hola! soy StatsBot, pregúntame algo de la Cohorte 2012 de estadística UV. usa los comandos /foto o /cat o /dog "
    bot.send_message(chat_id=chat_id, text=howare)
    

def howareu(bot, update):
    chat_id = update.message.chat_id
    howare="I'm fine, how you doin?"
    bot.send_message(chat_id=chat_id, text=howare)

def main():
    updater = Updater('661340966:AAEK38q9twh8Sgnmig64I78cFg19ZCJOmbg')
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('dog',dog))
    dp.add_handler(CommandHandler('start',start))
    dp.add_handler(CommandHandler('hi',hi))
    dp.add_handler(CommandHandler('howare',howareu))
    dp.add_handler(CommandHandler('cat',cat))
    dp.add_handler(CommandHandler('foto',foto))
    dp.add_handler(CommandHandler('modtest',modtest))
    resp_handler = MessageHandler(Filters.text, responder)
    dp.add_handler(resp_handler)
    dp.add_handler(MessageHandler(Filters.voice, voicemsg))
    
    updater.start_polling()
    updater.idle()
    
   
    
if __name__ == '__main__':
    main()