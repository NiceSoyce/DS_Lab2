from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def do_prediction():
    json = request.get_json()
    model = joblib.load("digits_pca_svm.pkl")
    df = pd.DataFrame(json, index=[0])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df)

    df_x_scaled = scaler.transform(df)

    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
    y_predict = model.predict(df_x_scaled)

    result = {"Predicted Number": y_predict[0]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0') 