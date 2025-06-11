FROM python:3.12-slim

WORKDIR /app

RUN pip install pandas scikit-learn gunicorn joblib flask

ADD server.py /app/server.py
ADD digits_pca_svm.pkl /app/digits_pca_svm.pkl

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0:5000", "server:app"]