from textblob import TextBlob

while True:
    y = input("Type your Sentence: ")
    edu = TextBlob(y)
    x = edu.sentiment.polarity
    # print(x)
    if x < 0:
        print("Negative")
    elif x == 0:
        print("Neutral")
    else:
        print("Positive")
