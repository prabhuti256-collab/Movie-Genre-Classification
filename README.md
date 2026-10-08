# 🎬 Movie Genre Classification

A machine learning and **Natural Language Processing (NLP)** project that predicts the genre of a movie based on its **plot/description**.

The project uses text preprocessing, **TF-IDF vectorization**, and machine learning classification to automatically classify movie descriptions into their corresponding genres.

## 🚀 Project Overview

Movie Genre Classification is a text classification problem where the model learns relationships between movie descriptions and their genres.

The system takes a movie description as input and predicts the most likely genre.

### Example

```text
Input:
A young detective investigates a mysterious disappearance
in a small town and uncovers a dangerous secret.

Prediction:
Mystery
```

The project demonstrates how NLP can be applied to real-world text classification problems.

## 🧠 Machine Learning Approach

The project follows this workflow:

```text
Movie Dataset
      ↓
Text Cleaning
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Genre Prediction
```

## 🔤 Natural Language Processing

The movie descriptions are converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF gives higher importance to words that are useful for distinguishing between different movie genres.

For example:

```text
"murder", "detective", "investigation"
        ↓
Likely associated with
        ↓
Mystery / Crime
```

```text
"love", "relationship", "romance"
        ↓
Likely associated with
        ↓
Romance
```

## 🤖 Machine Learning Models

The project can use multiple classification algorithms for genre prediction, including:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

The models are trained using TF-IDF features extracted from movie descriptions.

The best-performing model can then be saved and used for prediction.

## 📊 Dataset

The project uses a movie dataset containing information such as:

* Movie title
* Movie description/plot
* Genre

The description text is used as the primary input feature, while the genre is used as the target variable.

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP
* TF-IDF
* Joblib
* Streamlit

## 📁 Project Structure

```text
Movie-Genre-Classification/
│
├── dataset/
│   └── movies.csv
│
├── models/
│   ├── genre_model.pkl
│   └── label_encoder.pkl
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Project Workflow

```text
Movie Description
       ↓
Text Cleaning
       ↓
Tokenization / Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Machine Learning Classifier
       ↓
Genre Prediction
```

## 🛠️ Text Preprocessing

The text data is processed before model training.

Typical preprocessing steps include:

* Handling missing descriptions
* Removing unnecessary characters
* Converting text to lowercase
* Removing unnecessary whitespace
* TF-IDF feature extraction

The processed text is then converted into numerical vectors that machine learning models can understand.

## 📈 Model Evaluation

The classification models can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help determine how accurately the model identifies different movie genres.

## 💾 Saved Models

The trained models are saved using **Joblib**.

```text
models/
├── genre_model.pkl
└── label_encoder.pkl
```

`genre_model.pkl` contains the trained classification model.

`label_encoder.pkl` contains the encoding information required to convert predicted class labels back into genre names.

## 🔍 Prediction

The trained model can predict the genre of a new movie description.

Example:

```text
Input:
A group of astronauts travels through space
to explore an unknown planet.

Output:
Science Fiction
```

## 🌐 Streamlit Application

The project can be integrated with a Streamlit web application.

The application allows users to:

1. Enter a movie description.
2. Submit the description.
3. Process the text using the trained NLP pipeline.
4. Predict the movie genre.
5. Display the predicted genre.

Run the application using:

```bash
streamlit run app.py
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/prabhuti256-collab/Movie-Genre-Classification.git
```

Navigate to the project:

```bash
cd Movie-Genre-Classification
```

Create a virtual environment:

```bash
py -3.12 -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

Run:

```bash
python train_model.py
```

This will preprocess the movie descriptions, train the classification model, evaluate its performance, and save the trained model.

## 🔍 Make Predictions

Run:

```bash
python predict.py
```

The script will load the trained model and predict the genre of the provided movie description.

## 🌐 Run the Streamlit App

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌟 Key Features

* NLP-based movie genre classification
* TF-IDF text feature extraction
* Multiple machine learning classifiers
* Automated genre prediction
* Saved trained model
* Label encoding
* Streamlit interface
* Easy-to-use prediction system

## 🔮 Future Improvements

* Add more movie genres
* Use word embeddings such as Word2Vec
* Implement transformer-based models
* Improve text preprocessing
* Add genre probability/confidence scores
* Build a more advanced Streamlit dashboard
* Deploy the application online
* Experiment with BERT or other transformer models
* Support multi-label genre prediction

## 👩‍💻 Author

**Prabhuti**

B.Tech AI & ML Student

GitHub: https://github.com/prabhuti256-collab

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
