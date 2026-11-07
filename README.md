# Iris Flower Classification 🌸

## Project Overview
This project uses Machine Learning to classify Iris flowers into three different species based on their physical measurements.

The model predicts whether a flower belongs to:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Dataset
The Iris dataset contains 150 records with the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable is `Species`.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Machine Learning Algorithm
**Decision Tree Classifier**

The model is trained using the Iris dataset and evaluated using testing data.

## Project Workflow
1. Load the Iris dataset.
2. Preprocess the dataset.
3. Separate input features and target variable.
4. Split the dataset into training and testing sets.
5. Train the Decision Tree Classifier.
6. Predict flower species.
7. Evaluate model performance.
8. Generate a confusion matrix.
9. Predict the species of a new flower.

## Model Performance

- Training samples: 120
- Testing samples: 30
- Accuracy: 93.33%

### Confusion Matrix

The confusion matrix is saved as `confusion_matrix.png`.

## Sample Prediction

Input flower measurements are passed to the trained model.

**Predicted Species:** Iris-setosa

## How to Run the Project

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

3. Run the Python program:

```bash
python iris_classification.py
```

## Project Files

- `iris_classification.py` – Main Python program.
- `Iris.csv` – Dataset.
- `confusion_matrix.png` – Model evaluation visualization.
- `README.md` – Project documentation.

## Conclusion

The project demonstrates how Machine Learning can be used to classify Iris flowers based on their measurements. The Decision Tree Classifier achieved an accuracy of 93.33% on the test dataset.