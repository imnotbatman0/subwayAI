import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import psutil

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Using cuDNN:", tf.test.is_built_with_cuda())


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)]  # Adjust as needed
#         )
#         print("GPU memory limit set successfully.")
#     except RuntimeError as e:
#         print(e)

# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.threading.set_inter_op_parallelism_threads(8)


data_dir = pathlib.Path("/home/thanuj/Semester4/aiml/myown/subwayAI/dataset")

batch_size = 32
img_height = 200
img_width = 200

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    # validation_split = 0.1,
    # subset= "training",
    seed= 19,
    image_size=(img_height, img_width),
    batch_size= batch_size,
    label_mode='categorical'
)

# test_ds = keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.1,
#     subset="validation",
#     seed=19,
#     image_size=(img_height, img_width),
#     batch_size= batch_size,
#     label_mode='categorical'  
# )

class_names = train_ds.class_names
normalization_layer = tf.keras.layers.Rescaling(1./255)  # Scale from [0, 255] → [0, 1]
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))  # Apply only once!
# test = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Apply only once!

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE).cache(filename='./train_cache')

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="tanh", input_shape=(img_height, img_width, 3)
        ),        
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="tanh"  # input_shape=(31, 31, 32)
        ),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
])

#prev_weights = "model.weights.h5"
#model.load_weights(prev_weights)



model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    # validation_data=test_ds,
    epochs=250
)

batch_size = 16
model.fit(
    train_ds,
    # validation_data=test_ds,
    epochs=150
)
# batch_size = 8
# model.fit(
#     train_ds,
#     # validation_data=test_ds,
#     epochs=75
# )

# batch_size = 4
# model.fit(
#     train_ds,
#     # validation_data=test_ds,
#     epochs=50
# )
model.save("subai.keras")
