import sys
sys.path.append('../src')

from data_processing import DataProcessor
from neural_network import NeuralNetwork

def main():
    processor = DataProcessor('../data/train.csv')
    X_train, Y_train, X_dev, Y_dev, m_train = processor.process_data()

    nn = NeuralNetwork()
    nn.gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)

    dev_predictions = nn.make_predictions(X_dev)
    print("Development set accuracy:", nn.get_accuracy(dev_predictions, Y_dev))

if __name__ == "__main__":
    main()
