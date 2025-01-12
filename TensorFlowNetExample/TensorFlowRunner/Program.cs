using Tensorflow;
using static Tensorflow.Binding;

// Inicializar TensorFlow.NET
var graph = new Graph();
var session = new Session(graph);

// Crear los tensores de entrada (Placeholders)
var x = tf.placeholder(TF_DataType.TF_FLOAT);
var y = tf.placeholder(TF_DataType.TF_FLOAT);

// Redefinir las mismas variables que en el programa de entrenamiento
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

// Inicializar las variables
var init = tf.compat.v1.global_variables_initializer();
session.run(init);

// Restaurar el modelo entrenado
var saver = tf.train.Saver();
saver.restore(session, "model.ckpt");  // Restaura el modelo desde el checkpoint
Console.WriteLine("Modelo restaurado.");


// Crear los datos de entrada para la predicción
var newData = new float[,] {
            { 0.9f, 0.1f },
            { 1.0f, 0.0f },
            { 1.1f, 0.1f },
            { 0.1f, 0.1f },
            { 0.2f, 0.0f },
            { 0.2f, 0.2f },
        };

// Realizar predicciones con el modelo restaurado
var result = session.run(output, new FeedItem(x, newData));

// Mostrar las predicciones
Console.WriteLine("Predicciones:");
foreach (var item in result)
{
    Console.WriteLine((int)item);
}

Console.ReadLine();