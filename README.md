# Project Description for CPE 393
This project is a machine learning-based API built using Flask that predicts housing prices based on input features. The model is trained on housing data, and the API provides predictions along with confidence scores.

The main objective of this project is to allow users to send POST requests with a set of input features, and receive predictions (housing prices) along with the confidence of those predictions.

Model Details:
The model used in this project is a regression model ( Random Forest) that predicts housing prices based on 13 input features.

The model is saved as model2.pkl and is loaded when the Flask app starts.

## Setup Steps

### Step 1: Train the Model

Run train.py. (model2.pkl will be saved in app folder)


### Step 2: 

cd "project folder directory" and Run app.py


### Step 3:  Build Docker image

docker build -t ml-model .

### Step 4:  Run Docker container

docker run -p 9000:9000 ml-model

## Sample API Request and Response

### Predict Endpoint: `/predict`  
**Method:** `POST`  
**URL:** `http://127.0.0.1:9000/predict`

#### Request Body

```json
{
  "features": [
    [7420, 4, 2, 3, 1, 0, 1, 0, 1, 2, 1, 1, 0]
  ]
}
```

Each array contains 13 input features in this order:  
`[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus_semi-furnished, furnishingstatus_unfurnished]`

---

#### Response Example

```json
[
  {
    "prediction": 9010628.2,
    "confidence": 0.0
  }
]
```

- `prediction` = predicted house price
- `confidence` = simulated confidence score (0 = low, 1 = high)

---

## Health Check

To make sure the API is running:

Go to:  
[http://127.0.0.1:9000/health](http://127.0.0.1:9000/health)

Expected response:

```json
{
  "status": "ok"
}
```

---

## Tech Stack

- Python
- scikit-learn
- Flask
- Docker

or can pip install -r requirements.txt


Suprawee Chaisombat 3455