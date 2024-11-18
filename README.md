# SMS Spam Classifier

This project is an SMS Spam Classifier built using a Naive Bayes classifier. The application is developed with Streamlit, allowing users to easily interact with the model and classify SMS messages as spam or not spam.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How It Works](#how-it-works)

## Features

- User-friendly interface to input SMS messages.
- Real-time classification of messages as spam or ham (not spam).
- Visualizations of model performance metrics.
- Built with a Naive Bayes classifier for efficient spam detection.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NLTK (Natural Language Toolkit)
- NumPy

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/chauhan-pratham/SMS_Spam_Classifier

2. Open Your Command Line Interface (CLI):
   - On **Windows**, you can use Command Prompt or PowerShell.
   - On **macOS** or **Linux**, you can use the Terminal.
     
3. Change Directory to the Repository:
   After cloning, you need to navigate into the repository's directory. Use the `cd` (change directory) command:
   ```bash
   cd SMS_Spam_Classifier
   ```

   ### Additional Tips

- **List Files**: Once you are in the repository, you can list the files and directories using:
  ```bash
  ls  # On macOS/Linux
  dir # On Windows
  ```

- **Check Current Directory**: If you want to confirm that you are in the correct directory, you can use:
  ```bash
  pwd  # On macOS/Linux
  cd   # On Windows
  ```

- **Return to Previous Directory**: If you need to go back to the previous directory, you can use:
  ```bash
  cd ..
  ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
Note: When you type this web-browser will open and will load you app.

6. Enter an SMS message in the input box and click the "Predict" button to see if the message is spam or not.

## How It Works

1. **Data Collection**: The model is trained on a dataset of SMS messages labeled as spam or ham.
2. **Preprocessing**: The text data is cleaned and preprocessed using techniques such as tokenization, stopword removal, and vectorization.
3. **Model Training**: A Naive Bayes classifier is trained on the preprocessed data.
4. **Prediction**: The trained model predicts whether a new SMS message is spam or not based on user input.
