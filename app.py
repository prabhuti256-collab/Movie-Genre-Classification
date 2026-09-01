import streamlit as st
import pickle
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/genre_model.pkl"

with open(MODEL_PATH, "rb") as file:
    saved_model = pickle.load(file)

vectorizer = saved_model["vectorizer"]
model = saved_model["model"]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Keep letters and spaces
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
# PREDICTION FUNCTION
# ============================================================

def predict_genre(plot):

    cleaned_plot = clean_text(plot)

    plot_vector = vectorizer.transform(
        [cleaned_plot]
    )

    prediction = model.predict(
        plot_vector
    )

    return prediction[0]


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Movie Genre Classifier")

st.write(
    "Enter a movie plot summary and the "
    "machine learning model will predict its genre."
)


# ============================================================
# INPUT
# ============================================================

plot = st.text_area(
    "📝 Enter Movie Plot",
    height=180,
    placeholder=(
        "Example: A detective investigates "
        "a mysterious murder in a small town..."
    )
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🎯 Predict Genre",
    use_container_width=True
):

    if plot.strip() == "":

        st.warning(
            "Please enter a movie plot."
        )

    elif len(plot.strip()) < 10:

        st.warning(
            "Please enter a longer movie description."
        )

    else:

        genre = predict_genre(plot)

        st.success(
            f"🎬 Predicted Genre: {genre.upper()}"
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("🔍 How it works")

st.write(
    """
    1. Movie plot is entered by the user.
    
    2. Text is cleaned using NLP techniques.
    
    3. TF-IDF converts the text into numerical features.
    
    4. The trained machine learning model analyzes the features.
    
    5. The predicted movie genre is displayed.
    """
)

st.caption(
    "Movie Genre Classification | BTech AIML Project"
)
