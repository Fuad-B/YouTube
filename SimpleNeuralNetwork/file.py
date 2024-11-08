import numpy as np

   # Define a simple neural network with one hidden layer
class NeuralNetwork:
       def __init__(self, input_size, hidden_size, output_size):
           self.weights_input_hidden = np.random.rand(input_size, hidden_size)
           self.weights_hidden_output = np.random.rand(hidden_size, output_size)

       def forward(self, inputs):
           hidden_layer = np.dot(inputs, self.weights_input_hidden)
           output_layer = np.dot(hidden_layer, self.weights_hidden_output)
           return output_layer

   # Example usage
input_data = np.array([[0.1, 0.2]])
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
prediction = nn.forward(input_data)
print("Prediction:", prediction)