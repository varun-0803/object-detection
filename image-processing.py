import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set path to your dataset folder
dataset_path = r"C:\Users\KML\Downloads\dogsandcat\train"
r"C:\Users\KML\Downloads\dogsandcat\test"

# Set image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load dataset with ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Visualize some training images
images, labels = next(train_data)

plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(f"Label: {train_data.class_indices}")
    plt.axis("off")
plt.tight_layout()
plt.show()
from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')  # Use softmax for multi-class
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
    

)
import time
from tensorflow.keras.callbacks import Callback

class EpochTimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        self.epoch_times.append(duration)
        print(f"Epoch {epoch + 1} took {duration:.2f} seconds")

timing_callback = EpochTimingCallback()

history = model.fit(
    train_data,
    validation_data=val_data,
    
    callbacks=[timing_callback]
)


import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to your test image
test_image_path = r"C:\Users\KML\Downloads\SampleJPGImage_50kbmb.jpg"

# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(128, 128))  # Match training image size
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Map index to class label
class_labels = list(train_data.class_indices.keys())
predicted_label = class_labels[predicted_class]

# Show result
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()

print(f" Predicted class: {predicted_label}")
