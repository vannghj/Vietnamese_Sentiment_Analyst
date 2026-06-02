from src.predict import predict_sentiment

text = input("Nhập review: ")

result = predict_sentiment(text)

print("Prediction:", result)