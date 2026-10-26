import os
import pickle


def predict_product_category(title):
    model_path = os.path.join('..', 'models', 'product_classifier_model.pkl')
    vectorizer_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path):
        model_path = 'product_classifier_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'

    # Load artifacts
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Transform and predict
    title_vec = vectorizer.transform([title])
    prediction = model.predict(title_vec)

    return prediction[0]


if __name__ == '__main__':
    sample_title = "apple iphone 11 pro max 256gb"
    predicted_category = predict_product_category(sample_title)
    print(f"Product Title: '{sample_title}'")
    print(f"Predicted Category: {predicted_category}")
