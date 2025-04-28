import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

EMBEDDINGS_FILE = 'embeddings.pkl'
CLASSIFIER_FILE = 'svm_classifier.pkl'

def train_classifier():
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings, labels, label_encoder = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier test accuracy: {accuracy*100:.2f}%")

    with open(CLASSIFIER_FILE, 'wb') as f:
        pickle.dump((classifier, label_encoder), f)
    print(f"Saved trained classifier to {CLASSIFIER_FILE}")

if __name__ == "__main__":
    train_classifier()
