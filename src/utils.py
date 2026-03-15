import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def sentence_vector(tokens: list, model):
    """
    Convert a list of tokens into a single sentence vector using a word embedding model.

    The function filters tokens that exist in the embedding vocabulary and
    computes the average of their word vectors to represent the sentence.
    If none of the tokens are found in the model vocabulary, a zero vector
    with the same dimensionality as the model's embeddings is returned.

    Parameters
    ----------
    tokens : list of str
        A list of tokenized words representing a sentence.
    model : gensim.models.Word2Vec or similar
        A trained word embedding model that contains word vectors accessible
        via `model.wv` and has a `vector_size` attribute.

    Returns
    -------
    numpy.ndarray
        A 1D vector representing the sentence embedding with shape
        `(model.vector_size,)`.

    Notes
    -----
    This method uses mean pooling of word vectors to obtain the
    sentence representation.

    Examples
    --------
    >>> tokens = ["machine", "learning", "is", "fun"]
    >>> vec = sentence_vector(tokens, w2v_model)
    >>> vec.shape
    (300,)
    """
    words = [word for word in tokens if word in model.wv]

    if len(words) == 0:
        return np.zeros(model.vector_size)

    return np.mean(model.wv[words], axis=0)


def glove_sentence_vector(tokens: list, model):
    """
    Convert a list of tokens into a sentence vector using a GloVe embedding model.

    The function filters tokens that exist in the embedding vocabulary and
    computes the average of their word vectors to represent the sentence.
    If none of the tokens are found in the embedding model, a zero vector
    with the same dimensionality as the embeddings is returned.

    Parameters
    ----------
    tokens : list of str
        A list of tokenized words representing a sentence.
    model : dict-like
        A pre-trained GloVe embedding model or dictionary-like object
        that maps words to their corresponding embedding vectors and
        provides a `vector_size` attribute.

    Returns
    -------
    numpy.ndarray
        A one-dimensional vector representing the sentence embedding
        with shape `(model.vector_size,)`.

    Notes
    -----
    The sentence representation is obtained using mean pooling
    (the average of all word vectors in the sentence that exist in
    the embedding vocabulary).

    Examples
    --------
    >>> tokens = ["natural", "language", "processing"]
    >>> vec = glove_sentence_vector(tokens, glove_model)
    >>> vec.shape
    (300,)
    """
    words = [word for word in tokens if word in model]

    if len(words) == 0:
        return np.zeros(model.vector_size)

    return np.mean(model[words], axis=0)


def plot_history(history):
    """
    Plot training and validation metrics from a model training history.

    This function visualizes the learning progress of a trained model by
    plotting two subplots:

    1. Training and validation loss over epochs.
    2. Training and validation accuracy over epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        A history object returned by the `fit()` method of a Keras model.
        The object must contain a `history` attribute with the following keys:
        ``loss``, ``val_loss``, ``accuracy``, and ``val_accuracy``.

    Returns
    -------
    None

    Notes
    -----
    This function displays a matplotlib figure with two subplots (loss curve
    and accuracy curve) as a side effect and does not return any value.

    Examples
    --------
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
    >>> plot_history(history)
    """
    plt.figure(figsize=(12, 5))

    # subplot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], color="#FF7F50")
    plt.plot(history.history["val_loss"], color="#35C692")

    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    # subplot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], color="#FF7F50")
    plt.plot(history.history["val_accuracy"], color="#35C692")

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")

    plt.tight_layout()
    plt.show()


def model_eval(model, test_data, test_labels, scaler, encoder):
    """
    Evaluate a trained classification model using test data.

    This function preprocesses the test features using the provided scaler,
    generates predictions from the trained model, converts encoded labels
    back to their original form using the given encoder, displays a confusion
    matrix, and returns common evaluation metrics in a dataframe.

    Parameters
    ----------
    model : estimator object
        A trained classification model that implements a `predict()` method.
        The model may return either class labels or class probabilities.
    test_data : array-like or sparse matrix
        Feature matrix used for testing the model before scaling.
    test_labels : array-like
        Encoded true labels corresponding to `test_data`.
    scaler : sklearn.preprocessing.StandardScaler or similar
        Scaler used during training to normalize or standardize features.
        Must implement the `transform()` method to scale the test data
        consistently with the training data.
    encoder : sklearn.preprocessing.LabelEncoder or similar
        Encoder used to transform labels during preprocessing.
        Must implement `inverse_transform()` and contain the `classes_`
        attribute for mapping encoded labels back to their original form.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the following evaluation metrics:
        Accuracy, Precision (weighted), Recall (weighted), and F1 Score (macro).

    Notes
    -----
    This function displays a confusion matrix plot using matplotlib as a
    side effect before returning the evaluation metrics dataframe.

    Examples
    --------
    >>> df_metrics = model_eval(model, X_test, y_test, scaler, encoder)
    >>> print(df_metrics)
    """
    test_data = scaler.transform(test_data)

    y_pred = model.predict(test_data)

    if np.ndim(y_pred) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    test_labels = encoder.inverse_transform(test_labels)
    y_pred = encoder.inverse_transform(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot()
    plt.show()

    # Evaluation Metrics
    df = pd.DataFrame(
        {
            "Accuracy": [accuracy_score(test_labels, y_pred)],
            "Precision": [precision_score(test_labels, y_pred, average="weighted")],
            "Recall": [recall_score(test_labels, y_pred, average="weighted")],
            "F1 Score": [f1_score(test_labels, y_pred, average="macro")],
        }
    )

    return df
