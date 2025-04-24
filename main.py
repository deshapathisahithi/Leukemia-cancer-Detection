from tkinter import messagebox, filedialog, simpledialog, Scrollbar, Text, Label, Button, Tk, END 
import tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
import joblib
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import datetime

# Initialize main window
main = Tk()
main.title("CLASSIFICATION OF LEUKEMIA WHITE BLOOD CELL CANCER USING IMAGE PROCESSING AND MACHINE LEARNING")
main.geometry("1300x1200")

# Global variables
global filename, X, Y, model, model_et, accuracy
base_model = VGG16(weights='imagenet', include_top=False)
class_names = ['ALL', 'normal']

# History logging
def log_history(event):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("history.txt", "a") as f:
        f.write(f"[{timestamp}] {event}\n")

def view_history():
    text.delete('1.0', END)
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            history = f.read()
            text.insert(END, history)
    else:
        text.insert(END, "No history available.\n")

def clear_history():
    if os.path.exists("history.txt"):
        os.remove("history.txt")
        text.insert(END, "History cleared.\n")
        log_history("History cleared")
    else:
        text.insert(END, "No history to clear.\n")

# Feature extraction using VGG16
def extract_EfficientNetB3_features(img_path):
    if isinstance(img_path, bytes):
        img_path = img_path.decode('utf-8')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Upload dataset folder
def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, 'Dataset loaded\n')
    log_history("Dataset uploaded")

# Image processing and feature extraction
def imageProcessing():
    global X, Y, X_train, X_test, y_train, y_test
    model_data_path = "model/myimg_data.txt.npy"
    model_label_path = "modelmyimg_label.txt.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path):
        X = np.load(model_data_path)
        Y = np.load(model_label_path)
        text.insert(END, 'Model files loaded\n')
    else:
        X_features, y_labels = [], []
        for class_label, class_name in enumerate(class_names):
            class_folder = os.path.join("dataset", class_name)
            for img_file in os.listdir(class_folder):
                if img_file != 'Thumbs.db':
                    img_path = os.path.join(class_folder, img_file)
                    features = extract_EfficientNetB3_features(img_path)
                    X_features.append(features)
                    y_labels.append(class_label)
        X = np.array(X_features)
        Y = np.array(y_labels)
        np.save(model_data_path, X)
        np.save(model_label_path, Y)
        text.insert(END, 'Image processing completed\n')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    text.insert(END, f'Shape of X_train: {X_train.shape}\n')
    text.insert(END, f'Shape of X_test: {X_test.shape}\n')
    log_history("Image preprocessing completed")

# Train Naive Bayes classifier
def Model():
    global model
    model_filename = 'model/naive_bayes_model.joblib'
    text.delete('1.0', END)
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        text.insert(END, "Loaded Naive Bayes model from file.\n")
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)
    y_pred = model.predict(X_test)
    show_metrics(y_pred, y_test, "Naive Bayes")
    log_history("Naive Bayes model trained")

# Train ExtraTrees classifier
def Model1():
    global model_et
    model_et_filename = 'EfficientNetB3_extra_trees_model.joblib'
    text.delete('1.0', END)
    if os.path.exists(model_et_filename):
        model_et = joblib.load(model_et_filename)
        text.insert(END, "Loaded ExtraTreesClassifier model from file.\n")
    else:
        model_et = ExtraTreesClassifier(random_state=42)
        model_et.fit(X_train, y_train)
        joblib.dump(model_et, model_et_filename)
        text.insert(END, f"Trained and saved Extra Trees model to {model_et_filename}.\n")
    y_pred_et = model_et.predict(X_test)
    show_metrics(y_pred_et, y_test, "Extra Trees")
    log_history("Extra Trees model trained")

# Show classification metrics and plot confusion matrix
def show_metrics(y_pred, y_true, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100

    text.insert(END, f"{model_name} Confusion Matrix:\n{conf_matrix}\n\n")
    text.insert(END, f"{model_name} Classification Report:\n{class_report}\n")
    text.insert(END, f"{model_name} Accuracy: {accuracy:.2f}%\n")
    text.insert(END, f"{model_name} Precision: {precision:.2f}%\n")
    text.insert(END, f"{model_name} Recall: {recall:.2f}%\n")
    text.insert(END, f"{model_name} F1 Score: {f1:.2f}%\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Compare model performance using bar chart
def compare_models():
    metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # Evaluate Naive Bayes if trained
    if 'model' in globals():
        y_pred_nb = model.predict(X_test)
        metrics['Model'].append('Naive Bayes')
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred_nb) * 100)
        metrics['Precision'].append(precision_score(y_test, y_pred_nb, average='weighted') * 100)
        metrics['Recall'].append(recall_score(y_test, y_pred_nb, average='weighted') * 100)
        metrics['F1 Score'].append(f1_score(y_test, y_pred_nb, average='weighted') * 100)

    # Evaluate Extra Trees if trained
    if 'model_et' in globals():
        y_pred_et = model_et.predict(X_test)
        metrics['Model'].append('Extra Trees')
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred_et) * 100)
        metrics['Precision'].append(precision_score(y_test, y_pred_et, average='weighted') * 100)
        metrics['Recall'].append(recall_score(y_test, y_pred_et, average='weighted') * 100)
        metrics['F1 Score'].append(f1_score(y_test, y_pred_et, average='weighted') * 100)

    # Plotting
    if metrics['Model']:
        x = np.arange(len(metrics['Model']))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5*width, metrics['Accuracy'], width, label='Accuracy')
        ax.bar(x - 0.5*width, metrics['Precision'], width, label='Precision')
        ax.bar(x + 0.5*width, metrics['Recall'], width, label='Recall')
        ax.bar(x + 1.5*width, metrics['F1 Score'], width, label='F1 Score')

        ax.set_ylabel('Scores (%)')
        ax.set_title('Comparison of Model Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics['Model'])
        ax.legend()
        plt.tight_layout()
        plt.show()

        log_history("Compared model performance")
    else:
        text.insert(END, "Train at least one model to compare performance.\n")

# Upload and predict a single image
def predict():
    global model_et
    file_path = filedialog.askopenfilename(initialdir="testImages")

    if not file_path:
        return

    if 'model_et' not in globals():
        text.insert(END, "Model is not initialized. Train or load the model first.\n")
        return

    try:
        features = extract_EfficientNetB3_features(file_path)
        preds = model_et.predict(features.reshape(1, -1))
        class_label = class_names[int(preds[0])]

        img = cv2.imread(file_path)
        img = cv2.resize(img, (800, 400))
        cv2.putText(img, f"Recognized as: {class_label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f'Recognized as: {class_label}', img)
        cv2.waitKey(0)

        log_history(f"Test image predicted as: {class_label}")
    except Exception as e:
        text.insert(END, f"Error during prediction: {e}\n")

# Exit the application
def close():
    main.destroy()

# UI Components
font = ('times', 15, 'bold')
title = Label(main, text='CLASSIFICATION OF LEUKEMIA WHITE BLOOD CELL CANCER USING IMAGE PROCESSING AND MACHINE LEARNING', bg='LightBlue1', fg='black', font=font, height=3, width=120)
title.place(x=0, y=5)

ff = ('times', 12, 'bold')
Button(main, text="Upload Dataset", command=uploadDataset, font=ff).place(x=20, y=100)
Button(main, text="Image preprocessing", command=imageProcessing, font=ff).place(x=20, y=150)
Button(main, text="Build & Train NBC", command=Model, font=ff).place(x=20, y=200)
Button(main, text="Build & Train kNN", command=Model1, font=ff).place(x=20, y=250)
Button(main, text="Upload test image", command=predict, font=ff).place(x=20, y=300)
Button(main, text="View History", command=view_history, font=ff).place(x=20, y=350)
Button(main, text="Clear History", command=clear_history, font=ff).place(x=20, y=400)
Button(main, text="Compare Performance", command=compare_models, font=ff).place(x=20, y=450)
Button(main, text="Exit", command=close, font=ff).place(x=20, y=500)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=85, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450, y=100)

main.config(bg='SkyBlue')
main.mainloop()
