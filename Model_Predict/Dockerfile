FROM python:3.12-slim

WORKDIR /app

RUN pip install pandas scikit-learn joblib matplotlib

ADD Model_Predict/server.py /app/server.py/
ADD digits_pca_svm.pkl /app/digits_pca_svm.pkl

ENTRYPOINT ["python", "-u", "server.py"]
CMD []
