using Tensorflow;
using static Tensorflow.Binding;

// Inicializar TensorFlow.NET
var graph = new Graph();
var session = new Session(graph);

// Crear los datos de entrada (por ejemplo, una matriz de características)
var xData = new float[,] {
            { 0.0f, 0.0f },
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 1.0f, 1.0f }
        };

var yData = new float[,] {
            { 0.0f },
            { 1.0f },
            { 1.0f },
            { 0.0f }
        };

// Notice it is a xor gate

// Crear tensores de entrada y salida (Placeholders)
var x = tf.placeholder(TF_DataType.TF_FLOAT);
var y = tf.placeholder(TF_DataType.TF_FLOAT);

// Crear la red neuronal: una capa oculta con  con 5 capas ocultas y una capa de salida con 1
var weights1 = tf.Variable(tf.random.normal(new int[] { 2, 4 }, dtype: TF_DataType.TF_FLOAT));  // Capa 1: 2 entradas -> 4 neuronas
var bias1 = tf.Variable(tf.zeros(new int[] { 4 }, dtype: TF_DataType.TF_FLOAT));
var layer1 = tf.add(tf.matmul(x, weights1), bias1);
layer1 = tf.nn.relu(layer1);  // Activación ReLU

var weights2 = tf.Variable(tf.random.normal(new int[] { 4, 4 }, dtype: TF_DataType.TF_FLOAT));  // Capa 2: 4 entradas -> 4 neuronas
var bias2 = tf.Variable(tf.zeros(new int[] { 4 }, dtype: TF_DataType.TF_FLOAT));
var layer2 = tf.add(tf.matmul(layer1, weights2), bias2);
layer2 = tf.nn.relu(layer2);  // Activación ReLU

var weights3 = tf.Variable(tf.random.normal(new int[] { 4, 4 }, dtype: TF_DataType.TF_FLOAT));  // Capa 3: 4 entradas -> 4 neuronas
var bias3 = tf.Variable(tf.zeros(new int[] { 4 }, dtype: TF_DataType.TF_FLOAT));
var layer3 = tf.add(tf.matmul(layer2, weights3), bias3);
layer3 = tf.nn.relu(layer3);  // Activación ReLU

var weights4 = tf.Variable(tf.random.normal(new int[] { 4, 4 }, dtype: TF_DataType.TF_FLOAT));  // Capa 4: 4 entradas -> 4 neuronas
var bias4 = tf.Variable(tf.zeros(new int[] { 4 }, dtype: TF_DataType.TF_FLOAT));
var layer4 = tf.add(tf.matmul(layer3, weights4), bias4);
layer4 = tf.nn.relu(layer4);  // Activación ReLU

var weights5 = tf.Variable(tf.random.normal(new int[] { 4, 4 }, dtype: TF_DataType.TF_FLOAT));  // Capa 5: 4 entradas -> 4 neuronas
var bias5 = tf.Variable(tf.zeros(new int[] { 4 }, dtype: TF_DataType.TF_FLOAT));
var layer5 = tf.add(tf.matmul(layer4, weights5), bias5);
layer5 = tf.nn.relu(layer5);  // Activación ReLU

var weights6 = tf.Variable(tf.random.normal(new int[] { 4, 1 }, dtype: TF_DataType.TF_FLOAT));  // Capa de salida: 4 entradas -> 1 salida
var bias6 = tf.Variable(tf.zeros(new int[] { 1 }, dtype: TF_DataType.TF_FLOAT));
var output = tf.add(tf.matmul(layer5, weights6), bias6);  // Capa de salida

// Definir la función de pérdida (error cuadrático para clasificación)
var loss = tf.reduce_mean(tf.square(output - y));

// Optimización (Descenso de Gradiente)
var optimizer = tf.train.GradientDescentOptimizer(learning_rate: 0.1f).minimize(loss);

// Inicializar variables
var init = tf.compat.v1.global_variables_initializer();
session.run(init);

// Entrenar la red neuronal
for (int i = 0; i < 1000; i++)
{
    // Ejecutar una iteración de entrenamiento
    session.run(optimizer, new FeedItem(x, xData), new FeedItem(y, yData));

    if (i % 100 == 0)
    {
        var currentLoss = session.run(loss, new FeedItem(x, xData), new FeedItem(y, yData));
        Console.WriteLine($"Epoch {i}, Loss: {currentLoss[0]}");
    }
}

// Guardar el modelo entrenado
var saver = tf.train.Saver();
saver.save(session, "model.ckpt"); // copiar modelo model.ckpt al consumidor en su carpeta
Console.WriteLine("Modelo entrenado y guardado.");
Console.ReadLine();