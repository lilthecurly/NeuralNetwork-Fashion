# импорт необходимых библиотек и модулей
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.textinput import TextInput
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist


class FashionApp(App):
    def choice_number(self):
        a = self.txt1.text
        if a.isdigit():
            if 0 < int(a) <= 60000:

                x = int(a) - 1
                # загрузка набора данных fashion.MNIST

                (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
                # прописываем имена т.к. они не включены в набор данных
                class_names = ['Футболка', 'Штаны', 'Пуловер', 'Платье', 'Пальто', 'Сандалии', 'Рубашка', 'Кроссовки',
                               'Сумка', 'Ботинки']
                # нормализация данных для улучшения алгоритма оптимизации обучения сетей
                x_train = x_train / 255
                x_test = x_test / 255
                # создаем последовательную модель нейронной сети
                model = keras.Sequential([
                    keras.layers.Flatten(input_shape=(28, 28)),  # преобразование 2-мерного массива данных в 1-мерный
                    # массив
                    keras.layers.Dense(128, activation="relu"),
                    # для входного слоя используем 128 нейронов, используем функцию
                    # активации "relu" т.к она хороша в простых сетях
                    keras.layers.Dense(10, activation="softmax")
                    # в выходном слое используем 10 нейронов (по числу классов) и ф-ю
                    # активации "softmax" которая вернет 10 вероятных оценок
                ])
                # компилируем модель по стахостическому градиентому спуску, в качестве функции ошибки используем
                # категориальную перекрестную энтропию т.к классов > 2
                model.compile(optimizer=tensorflow.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                # обучаем сеть с помощью фунции fit
                model.fit(x_train, y_train, epochs=10)
                # точность в 99,99% - 100% была получена при достижении 715 эпохи

                # проверка точности предсказания для новых для сети изображений
                test_loss, test_acc = model.evaluate(x_test, y_test)
                print('Test accuracy:', test_acc)

                # получим предсказание от сети
                predictions = model.predict(x_train)
                # получим самое вероятное из предсказаний
                np.argmax(predictions[x])
                # выведем картинку на экран
                plt.figure()
                plt.imshow(x_train[x])
                plt.colorbar()
                plt.grid(False)
                # получим название класса
                var = class_names[np.argmax(predictions[x])]
                print(var)

                self.lbl1.text = "Это " + var
            else:
                self.lbl1.text = "Ошибка, введите целое число от 1 до 60000"
        else:
            self.lbl1.text = "Ошибка, введите целое число от 1 до 60000"

    def build(self):

        al = AnchorLayout()
        bl = BoxLayout(orientation='vertical', size_hint=[.5, .5])

        self.txt1 = TextInput(text="Введите целое число от 1 до 60000", multiline=False)
        bl.add_widget(self.txt1)
        btn1 = Button(text="Предсказать!")
        btn1.bind(on_press=self.choice_number)
        bl.add_widget(btn1)
        self.lbl1 = Label(text="")
        bl.add_widget(self.lbl1)

        al.add_widget(bl)
        return al


if __name__ == "__main__":
    FashionApp().run()
