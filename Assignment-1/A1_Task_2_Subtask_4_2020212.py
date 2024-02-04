
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


# Finding Best parameters


# Load sentences and labels from text files
corpus_path = os.path.join('..', 'Dataset', 'corpus.txt')
with open(corpus_path, 'r', encoding='utf-8') as f:
    sentences = f.readlines()

labels_path = os.path.join('..', 'Dataset', 'labels.txt')
with open(labels_path, 'r', encoding='utf-8') as f:
    labels = f.readlines()


# Shuffle the dataset
seed = 42
np.random.seed(seed)
indices = np.arange(len(sentences))
np.random.shuffle(indices)

sentences = np.array(sentences)[indices]
labels = np.array(labels)[indices]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=seed)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


svm_model = SVC()

# Grid Search to find the best parameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)


y_pred = best_model.predict(X_test_tfidf)

# Evaluate the accuracy of the best model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy of the Best Model: {accuracy * 100:.2f}%")


# Using best parameters for Testing Generated Outputs

# Trained on NO smoothing BIGRAMS


#Changing the variable names
X_train = sentences
y_train = [value.strip() for value in labels]


test_folder_path = os.path.join('Test Samples', 'coeff_1_no')
test_files = os.listdir(test_folder_path)

X_test = []
y_test = []

for test_file in test_files:
    file_path = os.path.join(test_folder_path, test_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines =f.readlines()
        X_test.extend(lines)
    
    # Extract label from the file name
    label = test_file.split('_')[1]
    for _ in range(50):
        y_test.append(label)

y_test = [value.replace('.txt', '') for value in y_test]


# Shuffle the dataset
seed = 42
np.random.seed(seed)
indices = np.arange(len(X_test))
np.random.shuffle(indices)

X_test = np.array(X_test)[indices]
y_test = np.array(y_test)[indices]


# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create and train SVC using best parameters as per Grid Search
svc = SVC(C=1, gamma=100, kernel='linear')
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)


# Evaluate the model
accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Trained on LAPLACE smoothing BIGRAMS


#Changing the variable names
X_train = sentences
y_train = [value.strip() for value in labels]


test_folder_path = os.path.join('Test Samples', 'coeff_1_laplace')
test_files = os.listdir(test_folder_path)

X_test = []
y_test = []

for test_file in test_files:
    file_path = os.path.join(test_folder_path, test_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines =f.readlines()
        X_test.extend(lines)
    
    # Extract label from the file name
    label = test_file.split('_')[1]
    for _ in range(50):
        y_test.append(label)

y_test = [value.replace('.txt', '') for value in y_test]


# Shuffle the dataset
seed = 42
np.random.seed(seed)
indices = np.arange(len(X_test))
np.random.shuffle(indices)

X_test = np.array(X_test)[indices]
y_test = np.array(y_test)[indices]


# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create and train SVC using best parameters as per Grid Search
svc = SVC(C=1, gamma=100, kernel='linear')
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)


# Evaluate the model
accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Trained on NO smoothing SAMPLE LEVEL data values


#Changing the variable names
X_train = sentences
y_train = [value.strip() for value in labels]


test_folder_path = os.path.join('Test Samples', 'sample_level')
test_files = os.listdir(test_folder_path)

X_test = []
y_test = []

for test_file in test_files:
    file_path = os.path.join(test_folder_path, test_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines =f.readlines()
        X_test.extend(lines)
    
    # Extract label from the file name
    label = test_file.split('_')[1]
    for _ in range(50):
        y_test.append(label)

y_test = [value.replace('.txt', '') for value in y_test]


# Shuffle the dataset
seed = 42
np.random.seed(seed)
indices = np.arange(len(X_test))
np.random.shuffle(indices)

X_test = np.array(X_test)[indices]
y_test = np.array(y_test)[indices]


# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create and train SVC using best parameters as per Grid Search
svc = SVC(C=1, gamma=100, kernel='linear')
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)


# Evaluate the model
accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Trained on KNESER-NEY smoothing BIGRAMS


#Changing the variable names
X_train = sentences
y_train = [value.strip() for value in labels]


test_folder_path = os.path.join('Test Samples', 'coeff_1_kneser')
test_files = os.listdir(test_folder_path)

X_test = []
y_test = []

for test_file in test_files:
    file_path = os.path.join(test_folder_path, test_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines =f.readlines()
        X_test.extend(lines)
    
    # Extract label from the file name
    label = test_file.split('_')[1]
    for _ in range(50):
        y_test.append(label)

y_test = [value.replace('.txt', '') for value in y_test]


# Shuffle the dataset
seed = 42
np.random.seed(seed)
indices = np.arange(len(X_test))
np.random.shuffle(indices)

X_test = np.array(X_test)[indices]
y_test = np.array(y_test)[indices]


# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create and train SVC using best parameters as per Grid Search
svc = SVC(C=1, gamma=100, kernel='linear')
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)


# Evaluate the model
accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


