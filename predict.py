import pickle
import re


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/genre_model.pkl"

print("Loading trained model...")

with open(MODEL_PATH, "rb") as file:
    saved_model = pickle.load(file)


vectorizer = saved_model["vectorizer"]
model = saved_model["model"]

print("Model loaded successfully!")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep only letters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICT GENRE
# ============================================================

def predict_genre(plot):

    # Clean movie description
    cleaned_plot = clean_text(plot)

    # Convert text into TF-IDF
    plot_vector = vectorizer.transform(
        [cleaned_plot]
    )

    # Predict genre
    prediction = model.predict(
        plot_vector
    )

    return prediction[0]


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\n" + "=" * 60)
print("🎬 MOVIE GENRE PREDICTOR")
print("=" * 60)

print("\nEnter a movie plot below.")

print("Type 'exit' to close the program.")


while True:

    plot = input(
        "\nMovie Plot: "
    )

    if plot.lower() == "exit":

        print("\nProgram closed.")
        break

    if len(plot.strip()) < 10:

        print(
            "Please enter a longer movie description."
        )

        continue

    genre = predict_genre(
        plot
    )

    print(
        "\n🎬 Predicted Genre:",
        genre.upper()
    )
    