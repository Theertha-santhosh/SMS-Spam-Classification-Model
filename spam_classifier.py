import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# 🔹 Download stopwords (if not available)
nltk.download('stopwords')

# 🔹 Load Dataset
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['labels', 'messages'])

# 🔹 Data Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Keep spaces
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['processed_messages'] = df['messages'].apply(preprocess_text)

# 🔹 Convert Text to TF-IDF Features
tfidf = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))  # Bi-grams for better context
X = tfidf.fit_transform(df['processed_messages']).toarray()

# 🔹 Encode Labels (Spam = 1, Ham = 0)
y = df['labels'].map({'ham': 0, 'spam': 1}).values

# 🔹 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 Balance Dataset with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 🔹 Train an Ensemble Model (Combining Three Models)
log_reg = LogisticRegression(C=10, max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
xgboost = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)

# 🔹 Train all models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# 🔹 Get Predictions (Averaging Probabilities)
log_preds = log_reg.predict_proba(X_test)[:, 1]  # Probability of spam
rf_preds = random_forest.predict_proba(X_test)[:, 1]
xgb_preds = xgboost.predict_proba(X_test)[:, 1]

# 🔹 Final Prediction using Weighted Average
final_probs = (0.3 * log_preds) + (0.3 * rf_preds) + (0.4 * xgb_preds)
final_predictions = np.where(final_probs > 0.4, 1, 0)  # **Lower threshold to 0.4**

# 🔹 Evaluate Model
conf_matrix = confusion_matrix(y_test, final_predictions)
class_report = classification_report(y_test, final_predictions)

# 🔹 Print Results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
