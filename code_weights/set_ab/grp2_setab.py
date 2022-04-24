#%% 
import pandas as pd
import librosa
import numpy as np
import random
import csv

#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils.np_utils import to_categorical
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools

#%% load dataset info
# a_data = pd.read_csv("/content/set_a.csv")
# b_data = pd.read_csv("/content/set_b.csv")
a_data = pd.read_csv("set_a.csv")
b_data = pd.read_csv("set_b.csv")

#%% merge data as maindata
data1 = [a_data, b_data]
maindata = pd.concat(data1)
maindata.drop(["sublabel","dataset"],axis="columns",inplace=True)
maindata = maindata.dropna()

#%% standardize filename
maindata = maindata.reset_index()
maindata.drop("index",axis="columns",inplace=True)

for index, row in maindata.iterrows():
  if index >= 124:
    maindata.at[index,'fname'] = maindata.at[index,'fname'][:6]+ maindata.at[index,'fname'][16:]
    
for index, row in maindata.iterrows():
  if index >= 436:
    maindata.at[index,'fname'] = maindata.at[index,'fname'][:12]+ maindata.at[index,'fname'][22:]

for index, row in maindata.iterrows():
  if index >= 124 and index <170:
    maindata.at[index,'fname'] = maindata.at[index,'fname'][:16]+"_"+ maindata.at[index,'fname'][16:]

for index, row in maindata.iterrows():
  if index >= 170 and index <236:
    maindata.at[index,'fname'] = maindata.at[index,'fname'][:12]+"_"+ maindata.at[index,'fname'][12:]
    
for index, row in maindata.iterrows():
  if index >= 236 and index <436:
    maindata.at[index,'fname'] = maindata.at[index,'fname'][:12]+"_"+ maindata.at[index,'fname'][12:]
    
#%% Save in a clean MainData.csv with new fname
f = open("MainData.csv", "w")
f.truncate()
f.close()

write_data = []
for index,row in maindata.iterrows():
    write_data.append(row)
with open("MainData.csv", 'w', newline="") as f: 
    writer = csv.writer(f)
    writer.writerows(write_data)
    
#%% MFCCs extraction, input dim = 40
def export_function(path, duration=12, sr=16000):
  input_length=sr*duration
  data, sr = librosa.load(path, res_type='kaiser_fast')
  dur = librosa.get_duration(data, sr)
  
  # pad audio file same duration
  if (round(dur) < duration):
      #print ("fixing audio lenght :", path)
      data = librosa.util.fix_length(data, input_length)  
  
  mfccs = np.mean(librosa.feature.mfcc(data, sr, n_mfcc=40).T,axis=0) 
  mfccs1 = np.array(mfccs).reshape([-1,1])
  return mfccs1

#%% Division of dataset  // To RESHUFFLE START FROM HERE
# x: audio data as nparray(float32), y: labels as str
x_Train = []
y_Train = []
x_Val = []
y_Val = []
x_Test = []
y_Test = []


total_num = 585
# we use roughly 80% for training and 10% each for validation and testing
training_num = 465
val_num = 60
test_num = 60

training_list = random.sample(range(total_num),training_num)
val_test_list = list(range(total_num))

for i in training_list:
    val_test_list.remove(i)
    
random.shuffle(val_test_list)
val_list = val_test_list[:60]
test_list = val_test_list[60:]

#%%   
CLASSES = ['murmur','normal','artifact','extrahls','extrastole']

# Map integer value to text labels to save labels as int
label_to_int = {k:v for v,k in enumerate(CLASSES)}

counter = 0

for path,label in zip(maindata.fname,maindata.label): 
    if(counter in training_list):
        output = export_function(path) #mfccs
        x_Train.append(output)
        y_Train.append(label_to_int[label])
    elif(counter in val_list):
        output = export_function(path) #mfccs
        x_Val.append(output)
        y_Val.append(label_to_int[label])
    else: 
        output = export_function(path) #mfccs
        x_Test.append(output)
        y_Test.append(label_to_int[label])
      
    counter += 1

    #%% Model structure
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.20, return_sequences=True,input_shape = (40,1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))
model.add(Dense(len(CLASSES), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['acc','mse', 'mae', 'mape','cosine_proximity'])
model.summary()

# name of model: lstmModel.hdf5
model.save('C:/Users/zhuda/Desktop/ELECTIVES/DL/project/absets/lstmModel.hdf5')

#%%
# convert labels from list to ndarray
y_train = np.array(to_categorical(y_Train, len(CLASSES)))
y_test = np.array(to_categorical(y_Test, len(CLASSES)))
y_val = np.array(to_categorical(y_Val, len(CLASSES)))

# reshape audio data into 3d array
x_train = np.reshape(x_Train,(465,40,1))
x_val = np.reshape(x_Val,(60,40,1))
x_test = np.reshape(x_Test,(60,40,1))

#%% TRAIN

# saved model checkpoint file
modelpath="C:/Users/zhuda/Desktop/ELECTIVES/DL/project/absets/lstmModel.hdf5"

MAX_PATIENT=12
MAX_EPOCHS=100
MAX_BATCH=32

# callbacks
callback=[ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1),
          ModelCheckpoint(filepath=modelpath, monitor='loss', verbose=1, save_best_only=True)]

print ("Training started..... please wait.")
# training
history=model.fit(x_train, y_train, 
                  batch_size=MAX_BATCH, 
                  epochs=MAX_EPOCHS,
                  verbose=0,
                  validation_data=(x_val, y_val),
                  callbacks=callback) 

print ("Training finised")

#%% TEST
# evaluate accuracy againtst all three datasets: training, validtion, testing
score = model.evaluate(x_train, y_train, verbose=0) 
print ("model training data score       : ",round(score[1]*100) , "%")

score = model.evaluate(x_val, y_val, verbose=0) 
print ("model validation data score     : ", round(score[1]*100), "%")

score = model.evaluate(x_test, y_test, verbose=0) 
print ("model testing data score        : ",round(score[1]*100) , "%")
