from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and column labels
model = pickle.load(open("model/predictor.pickle", "rb"))
columns = pickle.load(open("model/model_columns.pkl", "rb"))

# Categories for one-hot encoding
company_list = ['acer','apple','asus','dell','hp','lenovo','msi','other','toshiba']
typename_list = ['2in1convertible','gaming','netbook','notebook','ultrabook','workstation']
opsys_list = ['android','chromeos','linux','macosx','noos','windows10','windows10s','windows7','macos']
cpu_list = ['amd','intelcorei3','intelcorei5','intelcorei7','other']
gpu_list = ['amd','intel','nvidia']


def build_feature_vector(data):
    feature_list = []

    # Numeric features
    feature_list.append(int(data['ram']))
    feature_list.append(float(data['weight']))
    feature_list.append(int(data['touchscreen']))
    feature_list.append(int(data['ips']))

    # Helper for one-hot encoding
    def encode(lst, value):
        for item in lst:
            feature_list.append(1 if item == value else 0)

    # One-hot encode each category
    encode(company_list, data['company'])
    encode(typename_list, data['typename'])
    encode(opsys_list, data['opsys'])
    encode(cpu_list, data['cpu'])
    encode(gpu_list, data['gpu'])

    return feature_list


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create feature vector
        vector = build_feature_vector(data)

        # Convert to DataFrame aligned with training columns
        df = pd.DataFrame([vector], columns=columns)

        # Predict
        prediction = model.predict(df)[0]

        return jsonify({
            "success": True,
            "prediction": float(prediction)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/", methods=["GET"])
def home():
    return {"service": "Laptop Price Predictor", "status": "running"}


if __name__ == "__main__":
    app.run(port=5001, debug=True)
