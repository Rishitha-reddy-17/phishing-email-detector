import pandas as pd
from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("emails.csv")

# Keep required columns
df = df[['Email Text', 'Email Type']]

# Rename columns
df.columns = ['text', 'label']

# Remove empty rows
df = df.dropna()

# Convert labels into numbers
df['label'] = df['label'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})

# =========================================
# INPUTS AND OUTPUTS
# =========================================

X = df['text']
y = df['label']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# TF-IDF FEATURE EXTRACTION
# =========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================================
# MODEL TRAINING
# =========================================

print("Training Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Model Training Completed!")

# =========================================
# MODEL ACCURACY
# =========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, y_pred)

# =========================================
# DETECT EMAIL FUNCTION
# =========================================

def detect_email():

    email = email_box.get("1.0", END)

    # Empty check
    if email.strip() == "":

        messagebox.showwarning(
            "Warning",
            "Please Enter Email Content"
        )

        return

    # Convert email into TF-IDF
    email_tfidf = vectorizer.transform([email])

    # Predict
    prediction = model.predict(email_tfidf)

    # =====================================
    # SUSPICIOUS KEYWORDS
    # =====================================

    suspicious_keywords = [
        "urgent",
        "verify",
        "password",
        "bank",
        "click",
        "login",
        "account",
        "winner",
        "free",
        "offer",
        "limited",
        "credit",
        "otp",
        "suspended",
        "security"
    ]

    detected_keywords = []

    lower_email = email.lower()

    for word in suspicious_keywords:

        if word in lower_email:

            detected_keywords.append(word)

    # =====================================
    # URL DETECTION
    # =====================================

    url_count = (
        lower_email.count("http")
        + lower_email.count("www")
    )

    # =====================================
    # RESULT DISPLAY
    # =====================================

    if prediction[0] == 1:

        result_label.config(
            text="⚠️ PHISHING EMAIL DETECTED",
            fg="red"
        )

    else:

        result_label.config(
            text="✅ SAFE EMAIL",
            fg="lime"
        )

    # =====================================
    # EMAIL ANALYSIS
    # =====================================

    analysis_text.delete("1.0", END)

    analysis_text.insert(
        END,
        f"Model Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    analysis_text.insert(
        END,
        f"URLs Detected: {url_count}\n\n"
    )

    analysis_text.insert(
        END,
        "Suspicious Keywords Found:\n"
    )

    if detected_keywords:

        for word in detected_keywords:

            analysis_text.insert(
                END,
                f"• {word}\n"
            )

    else:

        analysis_text.insert(
            END,
            "No suspicious keywords detected.\n"
        )

# =========================================
# SHOW CONFUSION MATRIX
# =========================================

def show_matrix():

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Safe', 'Phishing'],
        yticklabels=['Safe', 'Phishing']
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Phishing Email Detection Confusion Matrix")

    plt.show()

# =========================================
# GUI WINDOW
# =========================================

root = Tk()

root.title("AI Phishing Email Detector")

# Fullscreen window
root.state("zoomed")

root.configure(bg="#0f172a")

# =========================================
# TITLE
# =========================================

title = Label(
    root,
    text="AI PHISHING EMAIL DETECTOR",
    font=("Arial", 30, "bold"),
    bg="#0f172a",
    fg="cyan"
)

title.pack(pady=15)

# =========================================
# SUBTITLE
# =========================================

subtitle = Label(
    root,
    text="Machine Learning Based Cybersecurity Tool",
    font=("Arial", 16),
    bg="#0f172a",
    fg="white"
)

subtitle.pack()

# =========================================
# STATUS LABEL
# =========================================

status_label = Label(
    root,
    text=f"✅ Model Trained Successfully\nAccuracy: {accuracy * 100:.2f}%",
    font=("Arial", 15, "bold"),
    bg="#0f172a",
    fg="lime",
    justify="center"
)

status_label.pack(pady=15)

# =========================================
# INPUT LABEL
# =========================================

input_label = Label(
    root,
    text="Enter Email Content:",
    font=("Arial", 16, "bold"),
    bg="#0f172a",
    fg="white"
)

input_label.pack(pady=10)

# =========================================
# EMAIL TEXT BOX
# =========================================

email_box = ScrolledText(
    root,
    height=10,
    width=110,
    font=("Arial", 12),
    bg="#1e293b",
    fg="white",
    insertbackground="white"
)

email_box.pack(pady=10)

# =========================================
# BUTTON FRAME
# =========================================

button_frame = Frame(
    root,
    bg="#0f172a"
)

button_frame.pack(pady=15)

# =========================================
# DETECT BUTTON
# =========================================

detect_button = Button(
    button_frame,
    text="Detect Email",
    font=("Arial", 14, "bold"),
    bg="cyan",
    fg="black",
    padx=25,
    pady=10,
    command=detect_email
)

detect_button.grid(
    row=0,
    column=0,
    padx=20
)

# =========================================
# CONFUSION MATRIX BUTTON
# =========================================

matrix_button = Button(
    button_frame,
    text="Show Confusion Matrix",
    font=("Arial", 14, "bold"),
    bg="orange",
    fg="black",
    padx=25,
    pady=10,
    command=show_matrix
)

matrix_button.grid(
    row=0,
    column=1,
    padx=20
)

# =========================================
# RESULT LABEL
# =========================================

result_label = Label(
    root,
    text="",
    font=("Arial", 24, "bold"),
    bg="#0f172a"
)

result_label.pack(pady=15)

# =========================================
# ANALYSIS TITLE
# =========================================

analysis_label = Label(
    root,
    text="Email Analysis",
    font=("Arial", 20, "bold"),
    bg="#0f172a",
    fg="cyan"
)

analysis_label.pack()

# =========================================
# ANALYSIS BOX
# =========================================

analysis_text = ScrolledText(
    root,
    height=10,
    width=90,
    font=("Arial", 11),
    bg="#1e293b",
    fg="white"
)

analysis_text.pack(pady=10)

# =========================================
# RUN APPLICATION
# =========================================

root.mainloop()
