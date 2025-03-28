import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df.rename(columns={'v1': 'SPAM/HAM', 'v2': 'Text'}, inplace=True)
df = df[['SPAM/HAM', 'Text']]  # Keep only relevant columns

# Define X and y
X = df['Text']
y = df['SPAM/HAM'].map({'ham': 0, 'spam': 1})

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Scaling
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf.toarray())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Save the model, vectorizer, and scaler
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, vectorizer, and scaler saved successfully!")
