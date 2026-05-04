**ASL Interpreter System**
This project is an American Sign Language (ASL) Interpreter that recognizes hand gestures and converts them into readable text using Machine Learning models (CNN/LSTM).



**Steps to Run the Project Locally**

Step - 1: Install Dependencies
Install all required libraries listed in "dependencies.txt"

Run the following script to install the dependencies:
pip install -r dependencies.txt


Step - 2: Prepare the Dataset
Run the following scripts to collect data:
python collect_static.py
python collect_sequence.py

Note - A pre-collected dataset is already provided, so this step can be skipped.


Step - 3: Train the model
Run the following scripts to train the models:
python train_cnn.py
python train_lstm.py

This will generate:
cnn_model.h5
cnn_labels.npy
lstm_model.h5
lstm_labels.npy


Step - 4: Run Prediction
Run the following script to predict ASL:
python predict.py



**Structure of the Repository**
ASL-Interpreter/
│── data/                 # Dataset
│   └── sequence/         # Contains Sequence used to train LSTM model
|   └── static/alphabets  # Contains images used to train CNN model
│
│── models/               # Trained models
│   └── cnn_labels.npy    # NumPy file storing the label mapping for CNN model
│   └── cnn_model.h5      # CNN trained model
│   └── lstm_labels.npy   # NumPy file storing the label mapping for LSTM model
│   └── lstm_model.h5     # LSTM trained model
│
│── collect_sequence.py  # Script for collecting sequences
│── collect_static.py    # Script for collecting static images(alphabets)
│── predict.py           # Prediction script
│── train_cnn.py         # Model training using CNN
│── train_lstm.py        # Model training using LSTM
│── README.md



**Notes:**
1. CNN is used for static gesture recognition
2. LSTM is used for sequence-based (dynamic) gesture recognition
3. MediaPipe is used for hand landmark detection
