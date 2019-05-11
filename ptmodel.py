import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import SGD, Adam

data = pd.read_csv('./df_final.csv')

height = 224
width = 224

mask = np.random.rand(len(data)) < 0.8
train = data[mask].reset_index(drop=True)
test = data[~mask].reset_index(drop=True)

pretrainmod = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(height, width, 3))

gen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, vertical_flip=True, validation_split=0.2)

train_gen = gen.flow_from_dataframe(train,
                                    directory='./images/comp/',
                                    x_col='fileid',
                                    y_col='Type1',
                                    class_mode='categorical',
                                    target_size=(height, width),
                                    subset='training')


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


class_list = list(data.Type1.unique())
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(pretrainmod,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))


NUM_EPOCHS = 10
BATCH_SIZE = 16
num_train_images = 15000

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath = "./checkpoints/" + "ResNet50" + "_model_weights.h5"
histpath = "./checkpoints/" + "ResNet50" + "_model_history.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
history = History()
callbacks_list = [checkpoint, history]

history = finetune_model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8,
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
                                       shuffle=True, callbacks=callbacks_list)


#Plot the training and validation loss + accuracy
#history = load_model('./checkpoints/ResNet50_model_weights.h5')


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


plot_training(history)
plt.savefig('acc_vs_epochs.png')
