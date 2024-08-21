# Import Library
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

# Initialising the CNN
# Build Model
model = Sequential()
# Build the First Layer 
# input Layer
model.add(Convolution2D(filters=32, kernel_size=3, strides=1, padding='same', input_shape=(224,224, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2)) 
   
# Build the Second Layer
model.add(Convolution2D(filters=64,  kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2)) 
 
# Build the Three Layer
model.add(Convolution2D(filters=128,  kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))    

# Build the Four Layer
model.add(Convolution2D(filters=256,  kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))    
model.add(Dropout(0.25))

# Build the Five Layer
model.add(Convolution2D(filters=256,  kernel_size=3, strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))    
model.add(Dropout(0.25))

# Prepare the ANN
model.add(Flatten())
# Full Connection
# First FC Layer 
model.add(Dense(512, activation='relu', name='fc_'+str(1)))
model.add(Dropout(0.5))
# Second FC Layer
model.add(Dense(256, activation='relu', name='fc_'+str(2)))
model.add(Dropout(0.5))
# Three FC Layer
# Output Layer
model.add(Dense(24, activation='softmax', name='fc_'+str(3)))

# Compile The Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary the model
model.summary()

# Encode The Labels
code = {'banana':0 , 'black jeans':1, 'blue dress':2, 'blue shirt':3, 'bottle':4,
        'bus':5, 'car':6, 'chair':7, 'fork':8, 'glass':9, 'keyboard':10,'knife':11,
        'laptop screen':12, 'lemon':13, 'mouse':14, 'men':15, 'orange':16,
        'red dress ':17, 'red skirt':18,'smart phone':19, 'sofa':20,
        'table':21,'tomato':22, 'women':23}
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x 
       
# Import Library
import glob as gb
import os
import cv2     

# Read The Data For Training set
dataPath = 'F:/zGraduation_Project/Project/dataset/' 
X_train = []
y_train = []
for folder in  os.listdir(dataPath +'training_set') : 
    files = gb.glob(pathname= str( dataPath +'training_set//' + folder + '/*.*g'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (224,224))
        X_train.append(list(image_array))
        y_train.append(code[folder])     

# Show Some Data From Training without Predict
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),25))) : 
    plt.subplot(5,5,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))
  
# Read The Data For test set 
X_test = []
y_test = []
for folder in  os.listdir(dataPath +'test_set') : 
    files = gb.glob(pathname= str(dataPath + 'test_set//' + folder + '/*.*g'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (224,224))
        X_test.append(list(image_array))
        y_test.append(code[folder])

# Show Some Data From test without Predict
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),25))) : 
    plt.subplot(5,5,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))

# Convert Dataset To Array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Shap The Dataset
print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')


# Read the DataSet For Training And Testing
# Pre-processing For Training Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)

# Pre-processing For test Data
test_datagen  = ImageDataGenerator(rescale = 1./255)

# Training data
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')


# Test data
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')

# Checkpoint...Save the model After Each Epoch
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('Checkpoint.hdf5',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min')

# For Prevent Show the Warning
import warnings
warnings.filterwarnings('ignore')

# Fitting the Model
history = model.fit_generator(training_set,
                              validation_data=test_set,
                              epochs = 25,
                              steps_per_epoch=len(training_set),
                              validation_steps=len(test_set),
                              callbacks = [checkpoint])


# Save the Model
model.load_weights('Checkpoint.hdf5')
model.save('model.h5')

# Convert Model keras to Tensorflow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)

#Plot The Training And Testing
# summarize history for accuracy  
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.ylim([0.75,1])
plt.legend(['train', 'test'], loc='lower right')  
plt.savefig('plot1')  



# summarize history for loss  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.ylim([0,15])
plt.legend(['train', 'test'], loc='upper right')  
plt.savefig('plot2') 

# Import Library
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from imutils import paths
import imutils
import random
import cv2

# Get the images
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
IMAGE_DIMS = (224, 224, 3)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process,and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

# Scale the data and Calculate the Size
data = np.array(data, dtype="float") / 255.0
print("data Size : {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# Classes
labels = [['banana', 'black jeans', 'blue dress', 'blue shirt', 'bottle',
        'bus', 'car', 'chair', 'fork', 'glass', 'keyboard','knife',
        'laptop screen', 'lemon', 'mouse', 'men', 'orange','red dress',
        'red skirt','smart phone', 'sofa','table','tomato', 'women']]

# encode the classes
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Show the Labels
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
    
# After Encode 
print('show encode classes \n' ,labels)

# load the Model
from tensorflow.keras.models import load_model
print("loading the Model....")
model =load_model('model.h5') 

1

# load the image
image = cv2.imread('images/aaaaaa.jpg')
output = imutils.resize(image, width=400)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
# preprocess the image for classification
image = cv2.resize(image, (224,224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# classify the input image
print("classifying image") 
proba = model.predict(image)[0]
idxs = np.argsort(proba)
print(proba)
print(idxs)

# Convert Text to sound
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")

# loop over the indexes of the high confidence class labels
arr1 = []
arr2 = []
for (i, j) in enumerate(idxs):
    y = mlb.classes_[i]
    arr1.append(y)
    x = proba[i] * 100
    arr2.append(x)
    
index = np.argmax(arr2)  
k = arr1[index]
txt = "{}: {:.2f}%".format(k, proba[index] * 100)

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))

# Print And Speak For The Object
print('..............................\n')
print('you are look at the '+str(k))
speak.speak('you are look at the '+str(k))
   
# show the output image
cv2.putText(output, txt, (10, (0 * 30) + 25),
             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.waitKey(0)
