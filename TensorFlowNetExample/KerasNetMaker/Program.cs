using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Numpy;


// Define the input and output data
var xData = np.array(new float[,]
{
            { 0.0f, 0.0f },
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 1.0f, 1.0f }
});

var yData = np.array(new float[,]
{
            { 0.0f },
            { 1.0f },
            { 1.0f },
            { 0.0f }
});

// Notice it is a xor gate

// Define the model using Keras Sequential API
var model = new Sequential();

// Add layers
model.Add(new Dense(4, input_dim: 2, activation: "relu"));  // Layer 1
model.Add(new Dense(4, activation: "relu"));               // Layer 2
model.Add(new Dense(4, activation: "relu"));               // Layer 3
model.Add(new Dense(4, activation: "relu"));               // Layer 4
model.Add(new Dense(4, activation: "relu"));               // Layer 5
model.Add(new Dense(1, activation: "linear"));             // Output layer

// Compile the model
model.Compile(optimizer: new SGD(lr: 0.1f), // learning rate
              loss: "mean_squared_error");

// Train the model
model.Fit(xData, yData, batch_size: 4, epochs: 1000, verbose: 1);

// Save the trained model
model.Save("model.h5");
Console.WriteLine("Modelo entrenado y guardado.");

Console.ReadLine();