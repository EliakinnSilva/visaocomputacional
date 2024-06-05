import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Caminho para o conjunto de dados
dataset_path = 'dataset/'


# Configurações do gerador de dados
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255,
                             rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Criação do modelo
num_classes = len(train_generator.class_indices)
base_model = MobileNetV2(weights='imagenet', include_top=False)
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])


# Compilação do modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Callbacks
checkpoint = ModelCheckpoint('user_recognition_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# Treinamento do modelo
model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint, early_stopping])


# Salvar o modelo treinado
model.save('user_recognition_model_final.keras')


# Função para prever a classe de uma imagem e marcar como desconhecido se não for confiante
def predict_user(image_path, model, class_indices, threshold=0.7):
    from tensorflow.keras.preprocessing import image
    
    # Carregar a imagem
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Fazer a previsão
    predictions = model.predict(img_array)
    max_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    # Se a confiança for menor que o limiar, marcar como desconhecido
    if max_confidence < threshold:
        return "Desconhecido"
    else:
        for class_name, class_index in class_indices.items():
            if class_index == predicted_class:
                return class_name


# Carregar o modelo treinado e prever uma nova imagem
model = load_model('user_recognition_model_final.keras')
class_indices = train_generator.class_indices

image_path = 'path_to_new_image.jpg'  # caminho para a nova imagem
user = predict_user(image_path, model, class_indices)
print(f'O usuário é: {user}')
