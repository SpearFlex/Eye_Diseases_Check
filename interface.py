import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, Rescaling
# from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.src.layers import Rescaling, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from matplotlib.image import imread
import keras

# Параметры модели
img_height = 256
img_width = 256
num_classes = 4  # Замените на количество ваших классов

# Функция для создания модели с той же архитектурой
def create_model(input_shape):
    # Входной слой
    input_layer = Input(shape=input_shape)

    # Нормализация данных от 0 до 1
    normalized_layer = Rescaling(1./255)(input_layer)

    # 1 блок свёртки
    conv_1 = Conv2D(filters=40, kernel_size=(3, 3), activation='relu', padding='same')(normalized_layer)
    pool_1 = MaxPooling2D(2, 2)(conv_1)

    # 2 блок свёртки
    conv_2 = Conv2D(50, (3, 3), activation='relu', padding='same')(pool_1)
    drop_2 = Dropout(0.2)(conv_2)

    # 3 блок свёртки
    conv_3 = Conv2D(60, (3, 3), activation='relu', padding='same')(drop_2)
    pool_3 = MaxPooling2D(2, 2)(conv_3)
    drop_3 = Dropout(0.1)(pool_3)

    conv_4 = Conv2D(70, (3, 3), kernel_regularizer=keras.regularizers.l2(1e-4))(drop_3)
    pool_4 = MaxPooling2D(2, 2)(conv_4)

    averaged_features = GlobalAveragePooling2D()(pool_4)
    flatten_features = Flatten()(averaged_features)
    output = Dense(num_classes, activation='softmax')(flatten_features)

    return Model(inputs=input_layer, outputs=output)

# Компиляция модели
def compile_model():
    model = create_model((img_height, img_width, 3))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

# Воссоздание модели и загрузка весов
model = compile_model()
model.load_weights('best_model.weights.h5')



class EyeDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Определение глазных заболеваний')
        self.setGeometry(100, 100, 800, 600)

        # Основной виджет и макет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Метка для отображения изображения
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Кнопка для загрузки изображения
        self.load_button = QPushButton('Загрузить изображение', self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Поле для вывода результата
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            try:
                # Отображение изображения
                pixmap = QPixmap(file_name)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

                # Обработка изображения и получение результата
                result = self.predict_image(file_name, self.result_text)
                self.result_text.setText(f"Результат: {result}")
            except Exception as e:
                self.result_text.setText(f"Ошибка при обработке изображения: {str(e)}")

    @staticmethod
    def predict_image(image_path, result_text):
        try:
            # Загрузка и предварительная обработка изображения
            result_text.setText(f"Результат: Начало1")
            img = Image.open(image_path)  # Загрузка изображения с помощью matplotlib
            result_text.setText(f"Результат: Начало2")
            img = img.resize((img_height, img_width), Image.Resampling.LANCZOS)
            result_text.setText(f"Результат: Начало3")
            # Изменение размера изображения
            I = np.asarray(img)
            result_text.setText(f"Результат: Начало4")
            # img_array = I / 255.0  # Нормализация
            img_array = np.expand_dims(I, axis=0)  # Добавление размерности батча
            result_text.setText(f"Результат: Начало5")

            # Предсказание с использованием модели
            predictions = model.predict(img_array)
            result_text.setText(f"Результат: Начало6")
            predicted_class_index = np.argmax(predictions, axis=-1)[0]
            result_text.setText(f"Результат: Начало7")

            # Здесь можно добавить список классов, если они известны
            class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            result_text.setText(f"Результат: Начало8")
            predicted_class_label = class_names[predicted_class_index]
            result_text.setText(f"Результат: Начало9")

            return predicted_class_label
        except Exception as e:
            return f"Ошибка: {str(e)}"



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EyeDiseaseApp()
    ex.show()
    sys.exit(app.exec_())