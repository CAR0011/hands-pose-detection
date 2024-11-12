from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Cargar datos de EMNIST (Extended MNIST) que incluye letras
dataset, metadata = tfds.load('emnist/byclass', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Extraer el número de clases únicas en las etiquetas
num_classes = metadata.features['label'].num_classes

# Normalizar: Números de 0 a 255, que sean de 0 a 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Aplicar normalización al conjunto de entrenamiento y prueba
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Estructura de la red
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)  # Número de clases dinámico
])

# Indicar las funciones a utilizar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(metadata.splits['train'].num_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

# Realizar el entrenamiento
model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=metadata.splits['train'].num_examples // BATCHSIZE
)

# Clase para definir el servidor HTTP. Solo recibe solicitudes POST.
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")

        # Obtener datos de la petición y limpiar los datos
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)

        # Realizar transformación para dejar igual que los ejemplos que usa EMNIST
        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(28, 28)
        arr = np.array(arr)
        arr = arr.reshape(1, 28, 28, 1)

        # Realizar y obtener la predicción
        prediction_values = model.predict(arr, batch_size=1)
        prediction = str(np.argmax(prediction_values))
        print("Predicción final: " + prediction)

        # Regresar respuesta a la petición HTTP
        self.send_response(200)
        # Evitar problemas con CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(prediction.encode())

# Iniciar el servidor en el puerto 8000 y escuchar por siempre
# Si se queda colgado, en el administrador de tareas buscar la tarea de python y finalizar tarea
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()