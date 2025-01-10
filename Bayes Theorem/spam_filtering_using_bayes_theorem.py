import pandas as pd
import re
from collections import Counter

#  Step 1: Loading the Dataset

# loading the dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# print(data.head())
# output:
"""
     v1                                                 v2 Unnamed: 2 Unnamed: 3 Unnamed: 4
0   ham  Go until jurong point, crazy.. Available only ...        NaN        NaN        NaN
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN        NaN        NaN
3   ham  U dun say so early hor... U c already then say...        NaN        NaN        NaN
4   ham  Nah I don't think he goes to usf, he lives aro...        NaN        NaN        NaN

"""
# print(data.columns)
# Output:
"""
Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='object')
"""
# dropping unnecessary columns
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

# renaming the columns for clarity
data.columns = ['label', 'message']
# print(data.columns)
# Output:
"""
Index(['label', 'message'], dtype='object')
"""
# looking out the changes
# print(data.head())
"""
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

"""
# Mapping 'ham' to 0 and 'spam' to 1
data['label'] = data['label'].map({'ham':0, 'spam':1})

# print(data.head())
# Output:
"""
   label                                            message
0      0  Go until jurong point, crazy.. Available only ...
1      0                      Ok lar... Joking wif u oni...
2      1  Free entry in 2 a wkly comp to win FA Cup fina...
3      0  U dun say so early hor... U c already then say...
4      0  Nah I don't think he goes to usf, he lives aro...

"""

# Step 2: Data Preprocessing (Cleaning the Text)

# A function to clean text (removing special characters, numbers, and converting to lowercase)
def cleaned_text(messages):
    return re.sub(r'[^a-zA-Z\s]', ' ', messages.lower())

# applying cleaning and storing cleaned data on 'message_cleaned' columns
data['message_cleaned'] = data['message'].apply(cleaned_text)

# print(data.head())
"""
   label                                            message                                    message_cleaned
0      0  Go until jurong point, crazy.. Available only ...                       ,      ..                ...
1      0                      Ok lar... Joking wif u oni...                            ...                 ...
2      1  Free entry in 2 a wkly comp to win FA Cup fina...                                                ...
3      0  U dun say so early hor... U c already then say...                        ...                     ...
4      0  Nah I don't think he goes to usf, he lives aro...           '                      ,             ...

"""

# Spliting dataset into ham and spam for separate analysis
spam_message = data[data['label'] == 1]
ham_message = data[data['label'] == 0]


# Step 3: Calculate Probabilities

"""
Prior Probability:
    The probability of a message being spam (P(Spam)) or ham (P(Ham)) before analyzing its content.
    Formula:
        P(Spam)= Number of Spam Messages / Total Messages
        P(Ham) = Number of Ham Messages / Total Messages

"""
# prior probabilities
p_spam = len(spam_message) / len(data)
p_ham = len(ham_message) / len(data)

# print(f"P(Spam): {p_spam}, P(Ham): {p_ham}")
# Output:
"""
P(Spam): 0.13406317300789664, P(Ham): 0.8659368269921034
"""

"""
Likelihood:
    The probability of a word appearing in spam messages (P(Word | Spam)) or ham messages (P(Word | Ham)).

"""

# Tokenizing messages
spam_words = Counter(" ".join(spam_message['message_cleaned']).split())
ham_words = Counter(" ".join(ham_message['message_cleaned']).split())

# Total number of words in Spam and Ham
total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())

# Vocabulary size
vocabulary_size = len(set(" ".join(data['message_cleaned']).split()))

"""
Laplace Smoothing:
    Adds +1 to avoid zero probability for unseen words.
    Formula:   
        P(Word | Spam) = (Frequency of Word in Spam + 1) / (Total spam word + vocabulary size)
"""
# a function to calculate likelihood
def likelihood(word, word_count, total_words):
    return (word_count.get(word, 0)+1) / (total_words + vocabulary_size)


# Step 4: Classify New Messages

"""
Posterior Probability:
    Using Bayes' Theorem:
        P(Spam | Message)∝P(Spam) ⋅ ∏ P(Word | Spam)
    Multiply prior probability (P(Spam)) with likelihoods of words (P(Word | Spam)).

Comparison:
    Compare P(Spam | Message) and P(Ham | Message).
    Predict "spam" if P(Spam | Message) > P(Ham | Message), otherwise "ham."
"""
# function to calculate posterior probability:
def classify_message(messages):
    words = cleaned_text(messages).split()

    # calculating p(Spam | Message)
    spam_prob = p_spam
    ham_prob = p_ham
    for word in words:
        spam_prob *=likelihood(word, spam_words, total_spam_words)
        ham_prob *=likelihood(word, ham_words, total_ham_words)

    # return the class with higher probability
    return 1 if spam_prob > ham_prob else 0

# Step 5: Evaluate the Model

# applying classifier to the dataset
data['message_cleaned'] = data['message'].apply(classify_message)

# applying the classifier and creating the 'Predicted_label' column
data['Predicted_label'] = data['message'].apply(classify_message)

# evaluating accuracy
accuracy = (data['label'] == data['Predicted_label']).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")


# Step 6: Test on New Messages


new_messages = ["Get 90 percent discount. Buy 5 get 1 free"]
for message in new_messages:
    prediction = classify_message(message)
    label = 'Spam' if prediction == 1 else 'Ham'
    print(f"Message: {message} => {label}")
