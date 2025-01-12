using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Numpy;

// Define the model using Keras Sequential API
var model = new Sequential();

// Add layers to the model
model.Add(new Dense(4, input_dim: 2, activation: "relu"));  // Layer 1
model.Add(new Dense(4, activation: "relu"));               // Layer 2
model.Add(new Dense(4, activation: "relu"));               // Layer 3
model.Add(new Dense(4, activation: "relu"));               // Layer 4
model.Add(new Dense(4, activation: "relu"));               // Layer 5
model.Add(new Dense(1, activation: "linear"));             // Output layer

// Compile the model (optional if only for inference, but recommended)
model.Compile(optimizer: new Adam(),
              loss: "mse",
              metrics: new string[] { "mae" });

// Load pre-trained weights
model.LoadWeight("model.h5");
Console.WriteLine("Model weights restored.");

// Define new input data for prediction
var newData = np.array(new float[,]
{
            { 0.9f, 0.1f },
            { 1.0f, 0.0f },
            { 1.1f, 0.1f },
            { 0.1f, 0.1f },
            { 0.2f, 0.0f },
            { 0.2f, 0.2f },
});

// Perform predictions
var predictions = model.Predict(newData);

// Display predictions
Console.WriteLine("Predictions:");
foreach (var prediction in predictions.GetData<float>())
{
    Console.WriteLine(prediction);
}

Console.ReadLine();