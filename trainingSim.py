print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import importDataInfo, balanceData, loadData, batchGen, createModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. DATA
path = 'myData'
data = importDataInfo(path)
data = balanceData(data, display=False) # Turn off if you trust it now
imagesPath, steerings = loadData(path, data)

# 2. SPLIT
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images:', len(xTrain))
print('Total Validation Images:', len(xVal))

# 3. MODEL
model = createModel()

# 4. PARAMS
batch_size = 128
epochs = 30
steps_per_epoch = len(xTrain) // batch_size
validation_steps = len(xVal) // batch_size

# 5. CALLBACKS
checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# 6. TRAIN
history = model.fit(batchGen(xTrain, yTrain, batch_size, 1),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=batchGen(xVal, yVal, batch_size, 0),
                    validation_steps=validation_steps,
                    callbacks=[checkpoint, early_stop])

# 7. SAVE & PLOT
model.save("model.h5")
print("Model Saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()