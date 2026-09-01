import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = "dataset/train_data.txt"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean movie descriptions."""

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("MOVIE GENRE CLASSIFICATION")
print("=" * 70)

print("\nLoading training dataset...")

df = pd.read_csv(
    TRAIN_FILE,
    sep=":::",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"]
)

print("Dataset loaded successfully!")

print("\nNumber of movies:", len(df))


# ============================================================
# DATA CLEANING
# ============================================================

print("\nCleaning dataset...")

# Remove missing values
df = df.dropna(
    subset=["genre", "description"]
)

# Remove duplicate descriptions
df = df.drop_duplicates(
    subset=["description"]
)

# Clean genre
df["genre"] = df["genre"].str.strip().str.lower()

# Clean descriptions
df["description"] = df["description"].apply(
    clean_text
)

# Remove empty descriptions
df = df[
    df["description"].str.len() > 0
]


print(
    "Movies after cleaning:",
    len(df)
)


# ============================================================
# SHOW GENRE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("GENRE DISTRIBUTION")
print("=" * 70)

genre_counts = df["genre"].value_counts()

print(genre_counts)

print(
    "\nTotal genres:",
    df["genre"].nunique()
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df["description"]
y = df["genre"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF testing shape:",
    X_test_tfidf.shape
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Linear SVM":
        LinearSVC(
            C=1.5,
            class_weight="balanced"
        )
}


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

results = {}

best_model = None
best_model_name = None
best_accuracy = 0


print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(
        f"{name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, accuracy in results.items():

    print(
        f"{name:<25} "
        f"{accuracy * 100:.2f}%"
    )


print("\nBest Model:", best_model_name)

print(
    f"Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# BEST MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 70)

best_predictions = best_model.predict(
    X_test_tfidf
)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "genre_model.pkl"
)

encoder_path = os.path.join(
    MODEL_FOLDER,
    "genre_classes.pkl"
)


# Save vectorizer + model
with open(model_path, "wb") as file:

    pickle.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model
        },
        file
    )


# Save genre classes
with open(encoder_path, "wb") as file:

    pickle.dump(
        list(best_model.classes_),
        file
    )


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "\nModel:",
    model_path
)

print(
    "Classes:",
    encoder_path
)

print("\nTraining completed successfully! 🎉")
