# Comprehensive Fake News Detection and Correction System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
from collections import Counter
import pickle
import joblib
from datetime import datetime

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Advanced NLP
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy

# API for fact-checking (you'll need to get API keys)
import requests
import json

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.tokenizer = None
        self.deep_model = None
        self.text_processor = TextProcessor()
        self.fact_checker = FactChecker()
        self.news_corrector = NewsCorrector()
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            print(f"Dataset loaded successfully with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Handle different column names based on your dataset
            if 'text' in df.columns:
                text_col = 'text'
            elif 'title' in df.columns:
                text_col = 'title'
            else:
                text_col = df.columns[1]  # Assume second column is text
            
            if 'label' in df.columns:
                label_col = 'label'
            else:
                label_col = df.columns[-1]  # Assume last column is label
            
            # Clean and prepare data
            df = df.dropna(subset=[text_col, label_col])
            df[text_col] = df[text_col].astype(str)
            
            # Convert labels to binary (0 for real, 1 for fake)
            if df[label_col].dtype == 'object':
                unique_labels = df[label_col].unique()
                print(f"Unique labels: {unique_labels}")
                # Map labels to binary
                label_mapping = {}
                for label in unique_labels:
                    if str(label).lower() in ['fake', 'false', '1', 'unreliable']:
                        label_mapping[label] = 1
                    else:
                        label_mapping[label] = 0
                df[label_col] = df[label_col].map(label_mapping)
            
            self.df = df[[text_col, label_col]].copy()
            self.df.columns = ['text', 'label']
            
            print(f"Final dataset shape: {self.df.shape}")
            print(f"Label distribution:\n{self.df['label'].value_counts()}")
            
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Label distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Real vs Fake News')
        plt.xlabel('Label (0=Real, 1=Fake)')
        plt.ylabel('Count')
        
        # Text length analysis
        self.df['text_length'] = self.df['text'].str.len()
        
        plt.subplot(1, 3, 2)
        self.df.boxplot(column='text_length', by='label', ax=plt.gca())
        plt.title('Text Length by Label')
        plt.xlabel('Label (0=Real, 1=Fake)')
        plt.ylabel('Text Length')
        
        # Word count analysis
        self.df['word_count'] = self.df['text'].str.split().str.len()
        
        plt.subplot(1, 3, 3)
        self.df.boxplot(column='word_count', by='label', ax=plt.gca())
        plt.title('Word Count by Label')
        plt.xlabel('Label (0=Real, 1=Fake)')
        plt.ylabel('Word Count')
        
        plt.tight_layout()
        plt.show()
        
        # Advanced text analysis
        self.advanced_text_analysis()
    
    def advanced_text_analysis(self):
        """Perform advanced text analysis"""
        # Readability scores
        self.df['readability'] = self.df['text'].apply(lambda x: flesch_reading_ease(x) if len(x) > 0 else 0)
        
        # Sentiment analysis (basic)
        from textblob import TextBlob
        self.df['sentiment'] = self.df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Plot advanced features
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(data=self.df, x='label', y='readability')
        plt.title('Readability Score by Label')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(data=self.df, x='label', y='sentiment')
        plt.title('Sentiment by Label')
        
        # Most common words in fake vs real news
        plt.subplot(1, 3, 3)
        fake_words = ' '.join(self.df[self.df['label'] == 1]['text'].str.lower())
        real_words = ' '.join(self.df[self.df['label'] == 0]['text'].str.lower())
        
        fake_word_freq = Counter(fake_words.split()).most_common(20)
        real_word_freq = Counter(real_words.split()).most_common(20)
        
        fake_df = pd.DataFrame(fake_word_freq, columns=['word', 'freq'])
        plt.barh(fake_df['word'], fake_df['freq'], color='salmon', alpha=0.7)
        plt.title('Top Words in Fake News')
        
        plt.tight_layout()
        plt.show()
    
    def train_traditional_models(self):
        """Train multiple traditional ML models"""
        print("=== Training Traditional ML Models ===")
        
        # Prepare text data
        X = self.df['text']
        y = self.df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Text preprocessing and vectorization
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=2
        )
        
        count_vectorizer = CountVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42, probability=True),
            'Passive Aggressive': PassiveAggressiveClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        # Train with TF-IDF
        print("Training with TF-IDF vectorizer...")
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[f"{name}_TFIDF"] = {
                'model': model,
                'vectorizer': tfidf_vectorizer,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f if auc_score else 'N/A'}")
        
        # Create ensemble model
        ensemble_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('nb', MultinomialNB()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
        
        ensemble = VotingClassifier(ensemble_models, voting='soft')
        ensemble.fit(X_train_tfidf, y_train)
        y_pred_ensemble = ensemble.predict(X_test_tfidf)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        results['Ensemble_TFIDF'] = {
            'model': ensemble,
            'vectorizer': tfidf_vectorizer,
            'accuracy': ensemble_accuracy,
            'predictions': y_pred_ensemble
        }
        
        print(f"Ensemble Model - Accuracy: {ensemble_accuracy:.4f}")
        
        self.models.update(results)
        self.X_test, self.y_test = X_test, y_test
        
        # Plot results
        self.plot_model_comparison()
        
        return results
    
    def train_deep_learning_model(self):
        """Train deep learning model using LSTM"""
        print("=== Training Deep Learning Model ===")
        
        X = self.df['text'].values
        y = self.df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Tokenization
        max_features = 20000
        maxlen = 200
        
        tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
        
        # Build LSTM model
        model = Sequential([
            Embedding(max_features, 128, input_length=maxlen),
            Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("LSTM Model Architecture:")
        model.summary()
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train_pad, y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_test_pad, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        
        print(f"LSTM Model - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Store model
        self.deep_model = model
        self.tokenizer = tokenizer
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_model_comparison(self):
        """Plot comparison of different models"""
        accuracies = []
        model_names = []
        
        for name, result in self.models.items():
            if 'accuracy' in result:
                accuracies.append(result['accuracy'])
                model_names.append(name.replace('_TFIDF', ''))
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'lightblue', 'pink', 'lightgray'])
        plt.title('Model Comparison - Accuracy Scores')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history for deep learning model"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_news(self, text, model_name='Ensemble_TFIDF'):
        """Predict if news is fake or real"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Using Ensemble_TFIDF")
            model_name = 'Ensemble_TFIDF'
        
        model_info = self.models[model_name]
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        
        # Preprocess text
        processed_text = self.text_processor.clean_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'text': text,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': probability[1] if probability is not None else None,
            'model_used': model_name
        }
        
        return result
    
    def predict_with_deep_model(self, text):
        """Predict using deep learning model"""
        if self.deep_model is None or self.tokenizer is None:
            return None
        
        # Preprocess
        processed_text = self.text_processor.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=200)
        
        # Predict
        probability = self.deep_model.predict(padded)[0][0]
        prediction = 'FAKE' if probability > 0.5 else 'REAL'
        
        return {
            'prediction': prediction,
            'confidence': probability,
            'model_used': 'LSTM'
        }
    
    def comprehensive_analysis(self, text):
        """Perform comprehensive analysis of news text"""
        print("=== Comprehensive News Analysis ===")
        
        # Basic predictions
        traditional_pred = self.predict_news(text)
        deep_pred = self.predict_with_deep_model(text)
        
        # Text analysis
        text_analysis = self.text_processor.analyze_text_features(text)
        
        # Fact checking (if available)
        fact_check_result = self.fact_checker.check_facts(text)
        
        # Generate corrected version
        corrected_news = self.news_corrector.correct_news(text, traditional_pred)
        
        results = {
            'original_text': text,
            'traditional_model_prediction': traditional_pred,
            'deep_model_prediction': deep_pred,
            'text_analysis': text_analysis,
            'fact_check': fact_check_result,
            'corrected_news': corrected_news,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
    
    def save_models(self, filepath_prefix='fake_news_models'):
        """Save trained models"""
        # Save traditional models
        for name, model_info in self.models.items():
            model_filename = f"{filepath_prefix}_{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model_info, model_filename)
        
        # Save deep learning model
        if self.deep_model:
            self.deep_model.save(f"{filepath_prefix}_lstm_model.h5")
            
        # Save tokenizer
        if self.tokenizer:
            with open(f"{filepath_prefix}_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        print("Models saved successfully!")
    
    def load_models(self, filepath_prefix='fake_news_models'):
        """Load saved models"""
        import os
        
        # Load traditional models
        for filename in os.listdir('.'):
            if filename.startswith(filepath_prefix) and filename.endswith('.pkl') and 'tokenizer' not in filename:
                model_name = filename.replace(filepath_prefix + '_', '').replace('.pkl', '')
                self.models[model_name] = joblib.load(filename)
        
        # Load deep learning model
        lstm_file = f"{filepath_prefix}_lstm_model.h5"
        if os.path.exists(lstm_file):
            from tensorflow.keras.models import load_model
            self.deep_model = load_model(lstm_file)
        
        # Load tokenizer
        tokenizer_file = f"{filepath_prefix}_tokenizer.pkl"
        if os.path.exists(tokenizer_file):
            with open(tokenizer_file, 'rb') as f:
                self.tokenizer = pickle.load(f)
        
        print("Models loaded successfully!")


class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_text_features(self, text):
        """Extract various text features"""
        features = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'readability_score': flesch_reading_ease(text) if len(text) > 0 else 0
        }
        
        # Sentiment analysis
        from textblob import TextBlob
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features


class FactChecker:
    def __init__(self):
        # You would need to implement API connections to fact-checking services
        self.api_endpoints = {
            'google_fact_check': 'https://factchecktools.googleapis.com/v1alpha1/claims:search',
            # Add more fact-checking APIs
        }
    
    def check_facts(self, text):
        """Check facts using various APIs (placeholder implementation)"""
        # This is a placeholder - you would implement actual API calls
        # to fact-checking services like Google Fact Check API, PolitiFact, etc.
        
        results = {
            'fact_check_available': False,
            'sources_checked': 0,
            'credibility_score': None,
            'conflicting_information': [],
            'supporting_sources': [],
            'message': 'Fact-checking service not implemented'
        }
        
        # Placeholder logic - in reality, you'd call external APIs
        suspicious_keywords = ['breaking', 'exclusive', 'shocking', 'you won\'t believe', 'doctors hate this']
        suspicious_count = sum(1 for keyword in suspicious_keywords if keyword in text.lower())
        
        if suspicious_count > 0:
            results['credibility_score'] = max(0.1, 0.8 - (suspicious_count * 0.2))
            results['message'] = f'Text contains {suspicious_count} potentially suspicious phrases'
        else:
            results['credibility_score'] = 0.7
            results['message'] = 'No obvious suspicious phrases detected'
        
        return results


class NewsCorrector:
    def __init__(self):
        # Initialize correction models/services
        pass
    
    def correct_news(self, text, prediction_result):
        """Generate corrected version of news if it's detected as fake"""
        if prediction_result['prediction'] == 'REAL':
            return {
                'correction_needed': False,
                'original_text': text,
                'corrected_text': text,
                'corrections_made': [],
                'message': 'News appears to be real, no corrections needed'
            }
        
        # Simple correction logic (you can enhance this with NLP models)
        corrections_made = []
        corrected_text = text
        
        # Remove sensational language
        sensational_words = {
            'shocking': 'notable',
            'unbelievable': 'significant',
            'breaking': 'recent',
            'exclusive': 'reported',
            'amazing': 'interesting',
            'incredible': 'remarkable'
        }
        
        for sensational, replacement in sensational_words.items():
            if sensational in corrected_text.lower():
                corrected_text = re.sub(sensational, replacement, corrected_text, flags=re.IGNORECASE)
                corrections_made.append(f"Replaced '{sensational}' with '{replacement}'")
        
        # Add hedging language
        if not any(hedge in corrected_text.lower() for hedge in ['allegedly', 'reportedly', 'according to']):
            corrected_text = "According to reports, " + corrected_text
            corrections_made.append("Added hedging language 'According to reports'")
        
        # Add disclaimer
        disclaimer = "\n\n[Note: This information should be verified with multiple reliable sources before sharing.]"
        corrected_text += disclaimer
        corrections_made.append("Added verification disclaimer")
        
        return {
            'correction_needed': True,
            'original_text': text,
            'corrected_text': corrected_text,
            'corrections_made': corrections_made,
            'confidence_score': prediction_result.get('confidence', 0),
            'message': f"Made {len(corrections_made)} corrections to improve reliability"
        }


# API Wrapper Class for External Integration
class FakeNewsAPI:
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_endpoint(self, text):
        """API endpoint for analyzing news"""
        try:
            result = self.detector.comprehensive_analysis(text)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_endpoint(self, text, model='ensemble'):
        """API endpoint for quick prediction"""
        try:
            result = self.detector.predict_news(text, model)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


# Main execution function
def main():
    """Main function to run the fake news detection system"""
    print("=== Fake News Detection and Correction System ===")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load your dataset
    dataset_path = "WELFake_Dataset.csv"  # Update with your actual dataset path
    df = detector.load_data(dataset_path)
    
    if df is not None:
        # Perform EDA
        detector.exploratory_data_analysis()
        
        # Train models
        traditional_results = detector.train_traditional_models()
        
        # Train deep learning model
        deep_model, history = detector.train_deep_learning_model()
        
        # Save models
        detector.save_models()
        
        print("\n=== Testing with Sample News ===")
        
        # Test with sample news
        sample_news = [
            "BREAKING: Scientists discover shocking truth about vaccines that doctors don't want you to know!",
            "The Federal Reserve announced a 0.25% interest rate increase following the latest economic indicators.",
            "You won't believe what this celebrity did! Click here to find out more shocking details!",
            "According to recent studies published in Nature, climate change effects are accelerating faster than previously predicted."
        ]
        
        for news in sample_news:
            print(f"\n--- Analyzing: {news[:60]}... ---")
            analysis = detector.comprehensive_analysis(news)
            
            print(f"Traditional Model: {analysis['traditional_model_prediction']['prediction']} "
                  f"({analysis['traditional_model_prediction']['confidence']:.3f})")
            
            if analysis['deep_model_prediction']:
                print(f"Deep Model: {analysis['deep_model_prediction']['prediction']} "
                      f"({analysis['deep_model_prediction']['confidence']:.3f})")
            
            print(f"Fact Check Score: {analysis['fact_check']['credibility_score']:.3f}")
            
            if analysis['corrected_news']['correction_needed']:
                print(f"Corrections Made: {len(analysis['corrected_news']['corrections_made'])}")
        
        # Initialize API wrapper
        api = FakeNewsAPI(detector)
        
        print("\n=== System Ready for Deployment ===")
        print("Models trained and saved successfully!")
        print("API wrapper initialized for React integration!")
        
        return detector, api
    
    else:
        print("Failed to load dataset. Please check the file path and format.")
        return None, None


# Advanced Feature Engineering Class
class AdvancedFeatureExtractor:
    def __init__(self):
        self.pos_tags = None
        self.named_entities = None
        
    def extract_linguistic_features(self, text):
        """Extract advanced linguistic features"""
        features = {}
        
        # POS tag distribution
        tokens = word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Count different types of POS tags
        pos_counts = Counter([tag for word, tag in pos_tags])
        total_words = len(tokens)
        
        features.update({
            'noun_ratio': pos_counts.get('NN', 0) / total_words if total_words > 0 else 0,
            'verb_ratio': pos_counts.get('VB', 0) / total_words if total_words > 0 else 0,
            'adj_ratio': pos_counts.get('JJ', 0) / total_words if total_words > 0 else 0,
            'adv_ratio': pos_counts.get('RB', 0) / total_words if total_words > 0 else 0,
        })
        
        # Complexity features
        features['type_token_ratio'] = len(set(tokens)) / len(tokens) if tokens else 0
        features['avg_sentence_length'] = len(tokens) / max(1, len(re.findall(r'[.!?]+', text)))
        
        # Emotional indicators
        emotional_words = ['amazing', 'shocking', 'incredible', 'unbelievable', 'stunning', 'outrageous']
        features['emotional_word_count'] = sum(1 for word in tokens if word in emotional_words)
        
        # Caps and punctuation patterns
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_ratio'] = text.count('!') / len(text) if text else 0
        
        return features
    
    def extract_semantic_features(self, text):
        """Extract semantic features using word embeddings"""
        # This would require word embeddings like Word2Vec, GloVe, or BERT
        # For now, we'll use simple keyword-based features
        
        credible_indicators = ['study', 'research', 'according to', 'data shows', 'expert', 'professor']
        suspicious_indicators = ['they don\'t want you to know', 'secret', 'hidden truth', 'conspiracy']
        
        features = {
            'credible_indicators': sum(1 for indicator in credible_indicators if indicator in text.lower()),
            'suspicious_indicators': sum(1 for indicator in suspicious_indicators if indicator in text.lower())
        }
        
        return features


# Enhanced Model Training with Feature Engineering
class EnhancedModelTrainer:
    def __init__(self, detector):
        self.detector = detector
        self.feature_extractor = AdvancedFeatureExtractor()
        
    def create_enhanced_features(self, texts):
        """Create enhanced feature matrix"""
        all_features = []
        
        for text in texts:
            # Basic text features
            basic_features = self.detector.text_processor.analyze_text_features(text)
            
            # Linguistic features
            linguistic_features = self.feature_extractor.extract_linguistic_features(text)
            
            # Semantic features
            semantic_features = self.feature_extractor.extract_semantic_features(text)
            
            # Combine all features
            combined_features = {**basic_features, **linguistic_features, **semantic_features}
            all_features.append(combined_features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        return feature_df.fillna(0)
    
    def train_enhanced_models(self):
        """Train models with enhanced features"""
        print("=== Training Enhanced Models with Feature Engineering ===")
        
        X = self.detector.df['text']
        y = self.detector.df['label']
        
        # Create enhanced features
        print("Extracting enhanced features...")
        feature_matrix = self.create_enhanced_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced models
        enhanced_models = {
            'Enhanced_RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'Enhanced_GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
            'Enhanced_XGBoost': self.get_xgboost_model(),
            'Enhanced_SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'Enhanced_MLP': MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in enhanced_models.items():
            if model is not None:
                print(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_columns': feature_matrix.columns.tolist(),
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f if auc_score else 'N/A'}")
        
        return results
    
    def get_xgboost_model(self):
        """Get XGBoost model if available"""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        except ImportError:
            print("XGBoost not available, skipping...")
            return None


# Comprehensive Evaluation and Visualization
class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_all_models(self, detector, X_test, y_test):
        """Comprehensive evaluation of all models"""
        print("=== Comprehensive Model Evaluation ===")
        
        evaluation_results = {}
        
        for model_name, model_info in detector.models.items():
            if 'model' in model_info and 'vectorizer' in model_info:
                model = model_info['model']
                vectorizer = model_info['vectorizer']
                
                # Prepare test data
                X_test_transformed = vectorizer.transform(X_test)
                
                # Predictions
                y_pred = model.predict(X_test_transformed)
                y_pred_proba = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                evaluation_results[model_name] = metrics
        
        # Create comparison visualizations
        self.plot_comprehensive_comparison(evaluation_results)
        
        return evaluation_results
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def plot_comprehensive_comparison(self, evaluation_results):
        """Plot comprehensive model comparison"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model].get(metric, 0) for model in model_names]
            bars = axes[i].bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # ROC curves
        self.plot_roc_curves(evaluation_results)
    
    def plot_roc_curves(self, evaluation_results):
        """Plot ROC curves for models with probability predictions"""
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in evaluation_results.items():
            if 'auc_score' in metrics:
                # This would require storing y_true and y_pred_proba
                # For demonstration, we'll skip the actual ROC curve plotting
                pass
        
        plt.title('ROC Curves Comparison')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()


# Configuration and Deployment Helper
class DeploymentHelper:
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration for deployment"""
        return {
            'model_version': '1.0',
            'api_version': 'v1',
            'supported_models': ['ensemble', 'lstm', 'enhanced_rf'],
            'max_text_length': 10000,
            'confidence_threshold': 0.5,
            'rate_limit': 100,  # requests per hour
        }
    
    def create_flask_app(self, detector):
        """Create Flask app for API deployment"""
        try:
            from flask import Flask, request, jsonify
            from flask_cors import CORS
            
            app = Flask(__name__)
            CORS(app)  # Enable CORS for React integration
            
            api_wrapper = FakeNewsAPI(detector)
            
            @app.route('/api/v1/analyze', methods=['POST'])
            def analyze_news():
                try:
                    data = request.json
                    text = data.get('text', '')
                    
                    if len(text) > self.config['max_text_length']:
                        return jsonify({
                            'status': 'error',
                            'message': 'Text too long'
                        }), 400
                    
                    result = api_wrapper.analyze_endpoint(text)
                    return jsonify(result)
                
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    }), 500
            
            @app.route('/api/v1/predict', methods=['POST'])
            def predict_news():
                try:
                    data = request.json
                    text = data.get('text', '')
                    model = data.get('model', 'ensemble')
                    
                    if len(text) > self.config['max_text_length']:
                        return jsonify({
                            'status': 'error',
                            'message': 'Text too long'
                        }), 400
                    
                    result = api_wrapper.predict_endpoint(text, model)
                    return jsonify(result)
                
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    }), 500
            
            @app.route('/api/v1/health', methods=['GET'])
            def health_check():
                return jsonify({
                    'status': 'healthy',
                    'version': self.config['model_version'],
                    'supported_models': self.config['supported_models']
                })
            
            return app
        
        except ImportError:
            print("Flask not available. Install with: pip install flask flask-cors")
            return None
    
    def create_fastapi_app(self, detector):
        """Create FastAPI app for high-performance deployment"""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
            
            app = FastAPI(title="Fake News Detection API", version=self.config['api_version'])
            
            # Enable CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            api_wrapper = FakeNewsAPI(detector)
            
            class TextInput(BaseModel):
                text: str
                model: str = "ensemble"
            
            @app.post("/api/v1/analyze")
            async def analyze_news(input_data: TextInput):
                if len(input_data.text) > self.config['max_text_length']:
                    raise HTTPException(status_code=400, detail="Text too long")
                
                result = api_wrapper.analyze_endpoint(input_data.text)
                return result
            
            @app.post("/api/v1/predict")
            async def predict_news(input_data: TextInput):
                if len(input_data.text) > self.config['max_text_length']:
                    raise HTTPException(status_code=400, detail="Text too long")
                
                result = api_wrapper.predict_endpoint(input_data.text, input_data.model)
                return result
            
            @app.get("/api/v1/health")
            async def health_check():
                return {
                    'status': 'healthy',
                    'version': self.config['model_version'],
                    'supported_models': self.config['supported_models']
                }
            
            return app
        
        except ImportError:
            print("FastAPI not available. Install with: pip install fastapi uvicorn")
            return None


# Testing and Validation Suite
class TestingSuite:
    def __init__(self, detector):
        self.detector = detector
        
    def run_comprehensive_tests(self):
        """Run comprehensive testing suite"""
        print("=== Running Comprehensive Tests ===")
        
        test_cases = [
            {
                'text': "BREAKING: Scientists discover that water is wet! This shocking revelation will change everything you know about H2O!",
                'expected': 'FAKE',
                'description': 'Obviously fake sensational news'
            },
            {
                'text': "The Federal Reserve announced today a 0.25% increase in interest rates, citing inflation concerns and economic indicators.",
                'expected': 'REAL',
                'description': 'Factual financial news'
            },
            {
                'text': "You won't believe what happens next! Doctors hate this one weird trick that eliminates all diseases instantly!",
                'expected': 'FAKE',
                'description': 'Clickbait medical misinformation'
            },
            {
                'text': "According to a study published in Nature journal, researchers have identified new biomarkers for early cancer detection.",
                'expected': 'REAL',
                'description': 'Scientific research news'
            },
            {
                'text': "Secret government documents reveal aliens have been living among us for decades! The truth they don't want you to know!",
                'expected': 'FAKE',
                'description': 'Conspiracy theory content'
            }
        ]
        
        results = []
        correct_predictions = 0
        
        for i, test_case in enumerate(test_cases):
            prediction = self.detector.predict_news(test_case['text'])
            is_correct = prediction['prediction'] == test_case['expected']
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'test_id': i + 1,
                'description': test_case['description'],
                'expected': test_case['expected'],
                'predicted': prediction['prediction'],
                'confidence': prediction['confidence'],
                'correct': is_correct
            })
            
            print(f"Test {i+1}: {test_case['description']}")
            print(f"Expected: {test_case['expected']}, Predicted: {prediction['prediction']}, Confidence: {prediction['confidence']:.3f}")
            print(f"Result: {'✓ PASS' if is_correct else '✗ FAIL'}\n")
        
        accuracy = correct_predictions / len(test_cases)
        print(f"Test Suite Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")
        
        return results


# Usage Instructions and Documentation
def print_usage_instructions():
    """Print comprehensive usage instructions"""
    print("""
=== FAKE NEWS DETECTION SYSTEM - USAGE INSTRUCTIONS ===

1. INSTALLATION REQUIREMENTS:
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk textstat textblob transformers flask fastapi

2. BASIC USAGE:
   - Load your dataset: detector.load_data('your_dataset.csv')
   - Train models: detector.train_traditional_models()
   - Make predictions: detector.predict_news('your news text')

3. ADVANCED FEATURES:
   - Comprehensive analysis: detector.comprehensive_analysis('news text')
   - Deep learning model: detector.train_deep_learning_model()
   - Enhanced features: EnhancedModelTrainer(detector).train_enhanced_models()

4. API DEPLOYMENT:
   - Flask: deployment_helper.create_flask_app(detector)
   - FastAPI: deployment_helper.create_fastapi_app(detector)

5. FOR REACT INTEGRATION:
   - Use the API endpoints: /api/v1/analyze and /api/v1/predict
   - Send POST requests with JSON: {'text': 'news content'}

6. TESTING:
   - Run test suite: TestingSuite(detector).run_comprehensive_tests()

7. SAVING/LOADING MODELS:
   - Save: detector.save_models()
   - Load: detector.load_models()

=== SYSTEM FEATURES ===
✓ Multiple ML algorithms (Logistic Regression, Random Forest, SVM, etc.)
✓ Deep Learning with LSTM
✓ Advanced feature engineering
✓ Text preprocessing and cleaning
✓ Fact-checking integration (placeholder)
✓ News correction and suggestions
✓ Comprehensive evaluation metrics
✓ API ready for web deployment
✓ Visualization and analysis tools
✓ Testing and validation suite

=== FOR YOUR HACKATHON ===
1. Replace 'WELFake_Dataset.csv' with your actual dataset path
2. Run main() function to train all models
3. Use the API endpoints for your React frontend
4. Customize the correction logic in NewsCorrector class
5. Add your own fact-checking API integrations

Good luck with your hackathon! 🚀
    """)


if __name__ == "__main__":
    # Print usage instructions
    print_usage_instructions()
    
    # Run the main system
    detector, api = main()
    
    if detector is not None:
        # Initialize additional components
        enhanced_trainer = EnhancedModelTrainer(detector)
        evaluator = ModelEvaluator()
        deployment_helper = DeploymentHelper()
        testing_suite = TestingSuite(detector)
        
        print("\n=== Additional Training and Testing ===")
        
        # Train enhanced models
        enhanced_results = enhanced_trainer.train_enhanced_models()
        detector.models.update(enhanced_results)
        
        # Run comprehensive evaluation
        evaluation_results = evaluator.evaluate_all_models(detector, detector.X_test, detector.y_test)
        
        # Run testing suite
        test_results = testing_suite.run_comprehensive_tests()
        
        print("\n=== Deployment Options ===")
        print("1. Flask App: deployment_helper.create_flask_app(detector)")
        print("2. FastAPI App: deployment_helper.create_fastapi_app(detector)")
        print("3. Direct Integration: Use FakeNewsAPI(detector) for custom integration")
        
        # Example of creating deployment app
        flask_app = deployment_helper.create_flask_app(detector)
        if flask_app:
            print("\nFlask app created successfully!")
            print("Run with: flask_app.run(debug=True, host='0.0.0.0', port=5000)")
        
        print("\n=== System Ready for Production! ===")
        print("All models trained, tested, and ready for deployment.")
        print("Connect your React frontend to the API endpoints for full functionality.")