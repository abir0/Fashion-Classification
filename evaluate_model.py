import pathlib
from fastai.vision.all import *
from sklearn.metrics import classification_report, accuracy_score


def numpy_array_to_image(arr):
    return Image.fromarray(arr.astype('uint8'))


def get_test_data(data_path):
    test_df = pd.read_csv(data_path)
    test_X = test_df.drop('label', axis=1)
    test_Y = test_df['label']
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    test_X = test_X.reshape(-1, 28, 28)
    return test_X, test_Y


def evaluate_model(model_path, data_path):
    # Load the saved model
    model = load_learner(model_path)

    # Get the test data
    test_X, test_Y = get_test_data(data_path)

    # Create a DataLoader for the test data
    test_data = L(zip(list(map(numpy_array_to_image, test_X)), test_Y))
    test_dl = model.dls.test_dl(test_data, with_labels=True)

    # Get predictions on the test data
    preds = model.get_preds(dl=test_dl, with_decoded=True)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_Y, preds[0].argmax(dim=1).numpy())
    classification_rep = classification_report(test_Y, preds[0].argmax(dim=1).numpy())

    return accuracy, classification_rep


def generate_report(accuracy, classification_rep):
    # Get the architecture summary
    with open("model_summary.txt", "r") as f:
        architecture_summary = "".join(f.readlines())

    # Write the results to the output file
    with open("output.txt", "w") as f:
        f.write("\nEvaluation Metrics:\n")
        f.write(f"\nAccuracy: {accuracy}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_rep)
        f.write("\n\nModel Architecture Summary:\n\n")
        f.write(architecture_summary)


def main():
    pathlib.PosixPath = pathlib.WindowsPath
    # Get the model and data paths
    model_path = pathlib.Path("models/model_v2.pkl")
    data_path = pathlib.Path("dataset/fashion-mnist_test.csv")

    # Check if the model and data files exist
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
    if not data_path.exists():
        print(f"Error: Data file '{data_path}' does not exist.")
        sys.exit(1)
    
    # Evaluate the model
    accuracy, classification_rep = evaluate_model(model_path, data_path)

    # Generate the evaluation report
    generate_report(accuracy, classification_rep)
    
    print("Evaluation report saved to 'output.txt'.")

if __name__ == "__main__":
    main()
