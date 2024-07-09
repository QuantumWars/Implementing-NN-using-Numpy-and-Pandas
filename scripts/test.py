import sys
import matplotlib.pyplot as plt
sys.path.append('../src')

from data_processing import DataProcessor
from neural_network import NeuralNetwork

def main():
    processor = DataProcessor('../data/train.csv')
    X_train, Y_train, _, _, _ = processor.process_data()

    nn = NeuralNetwork()
    nn.gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)

    index = 0  # Change the index to test different images
    prediction, label, image = nn.test_prediction(X_train, Y_train, index)
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")

    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()
