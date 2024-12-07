import re
import os
import shutil
import speech_recognition as sr
import pyttsx3
import tensorflow as tf
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime
from tkinter import filedialog
import tkinter as tk

# Initial balance
balance = 0  # Assuming an initial balance of Rs 0

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Function to convert text to speech
def SpeakText(command):
    engine.say(command)
    engine.runAndWait()

# Function to convert date string to YYYY-MM-DD format
def convert_to_date(date_str):
    try:
        # Try parsing the date in the expected format
        date_obj = datetime.strptime(date_str, "%d %B %Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        # Handle errors gracefully
        SpeakText("The provided date is invalid. Please try again with a valid format, like '21 November 2024'.")
        raise ValueError(f"Invalid date format: {date_str}")

# Function to check wallet balance
def check_balance():
    text = "Your current balance is Rs {}".format(balance)
    print(text)
    SpeakText(text)

# Function to record expenditure
def record_expenditure(amount, recipient):
    with open('spent_records.txt', 'a') as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"Rs{amount} was paid to {recipient} at {timestamp}\n"
        file.write(entry)

# Function to record received money
def record_received(amount, sender):
    with open('received_records.txt', 'a') as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"Rs{amount} was received from {sender} at {timestamp}\n"
        file.write(entry)

# Function to append and reset spent records
def reset_spent_records():
    if os.path.exists('spent_records.txt'):
        with open('spent_records.txt', 'r') as source, open('all_spent_records.txt', 'a') as target:
            shutil.copyfileobj(source, target)
        open('spent_records.txt', 'w').close()

# Function to append and reset received records
def reset_received_records():
    if os.path.exists('received_records.txt'):
        with open('received_records.txt', 'r') as source, open('all_received_records.txt', 'a') as target:
            shutil.copyfileobj(source, target)
        open('received_records.txt', 'w').close()

# Function to review transactions on a specific date
def review_transactions_on_date(date):
    transactions = []
    with open('all_spent_records.txt', 'r') as file:
        transactions += [record.strip() for record in file if date in record]
    with open('all_received_records.txt', 'r') as file:
        transactions += [record.strip() for record in file if date in record]

    if transactions:
        for transaction in transactions:
            print(transaction)
            SpeakText(transaction)
    else:
        text = f"No transactions found on {date}"
        print(text)
        SpeakText(text)

# Function to load and preprocess an image for currency recognition
def preprocess_test_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load the currency recognition model
currency_model = tf.keras.models.load_model('D:/Programing/RnD/my_model_new')
currency_classes = ['10', '20', '50', '100', '200', '500', '2000']

# Load the card recognition model
card_model = torch.load('card_detect_model.pth')
card_classes = ['Aadhaar', 'PAN', 'Driving_Licence', 'Voter_ID']

# Define transformations for card recognition
card_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess the input image for card recognition
def preprocess_card_image(image_path, transform):
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to predict the card type
def predict_card(image_path, model, transform, classes):
    img = preprocess_card_image(image_path, transform)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = classes[predicted_idx]
        return predicted_class

# Main loop to listen for commands
while True:
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source)
            MyText = recognizer.recognize_google(audio).lower()

        # Currency Recognition
        if "check currency" in MyText:
            SpeakText("Capturing image of the currency.")
            image_path = capture_image_from_camera()

            if image_path:
                test_image = preprocess_test_image(image_path)
                predictions = currency_model.predict(test_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_label = currency_classes[predicted_class]

                SpeakText(f"The predicted currency is {predicted_label} rupees. Was this amount received or spent?")
                transaction_type = None
                try_count = 0

                while transaction_type not in ["received", "spent"] and try_count < 3:
                    try:
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            print("Waiting for transaction type (received/spent)...")
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                            transaction_type = recognizer.recognize_google(audio).lower()
                    except sr.UnknownValueError:
                        SpeakText("I didn't understand. Please say 'received' or 'spent'.")
                        try_count += 1

                if transaction_type in ["received", "spent"]:
                    amount = int(predicted_label)
                    if "received" in transaction_type:
                        balance += amount
                        record_received(amount, "Currency Check")
                        SpeakText(f"{amount} rupees credited. New balance: {balance}")
                    elif "spent" in transaction_type:
                        if balance >= amount:
                            balance -= amount
                            record_expenditure(amount, "Currency Check")
                            SpeakText(f"{amount} rupees debited. New balance: {balance}")
                        else:
                            SpeakText("Insufficient balance for this transaction.")

        # Card Recognition
        elif "check card" in MyText:
            SpeakText("Capturing image of the card.")
            image_path = capture_image_from_camera()
            if image_path:
                predicted_card = predict_card(image_path, card_model, card_transform, card_classes)
                SpeakText(f"The recognized card is {predicted_card}.")

        # Other commands (e.g., balance checks, transactions) follow here...
        elif match := re.search(r'(\d+)\s*.paid\s*to\s(.+)', MyText):
            amount_spent, recipient = int(match.group(1)), match.group(2)
            if balance >= amount_spent:
                balance -= amount_spent
                record_expenditure(amount_spent, recipient)
                SpeakText(f"Paid {amount_spent} to {recipient}. Balance: {balance}")
            else:
                SpeakText("Insufficient balance. Please add funds.")

        elif match := re.search(r'(\d+)\s*.received\s*from\s(.+)', MyText):
            amount_received, sender = int(match.group(1)), match.group(2)
            balance += amount_received
            record_received(amount_received, sender)
            SpeakText(f"Received {amount_received} from {sender}. Balance: {balance}")

        elif "check balance" in MyText:
            check_balance()

        elif match := re.search(r'set initial amount (\d+)', MyText):
            balance = int(match.group(1))
            SpeakText(f"Initial balance set to {balance}")
            reset_spent_records()
            reset_received_records()

        elif "review paid records" in MyText:
            with open('spent_records.txt', 'r') as file:
                records = file.readlines()
                if records:
                    for record in records:
                        SpeakText(record.strip())
                else:
                    SpeakText("No recent expenditures.")

        elif "review received records" in MyText:
            with open('received_records.txt', 'r') as file:
                records = file.readlines()
                if records:
                    for record in records:
                        SpeakText(record.strip())
                else:
                    SpeakText("No recent receipts.")

        elif match := re.search(r'did i make any transactions on (\d{1,2}\s*\w+\s*\d{4})', MyText):
            date = convert_to_date(match.group(1))
            review_transactions_on_date(date)

        elif "exit" in MyText:
            SpeakText("Exiting. Goodbye!")
            break

        else:
            SpeakText("Sorry, I didn't get that. Please try again.")

    except sr.UnknownValueError:
        SpeakText("I didn't understand. Please try again.")
    except sr.RequestError as e:
        print("Request failed; {0}".format(e))
