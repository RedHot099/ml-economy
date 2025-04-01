# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_resampled, y_train_resampled)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


gnb.fit(X_train_resampled, y_train_resampled)


y_pred = gnb.predict(X_test)


print("\nGaussian Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
import numpy as np


class_weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}


lr = LogisticRegression(solver="lbfgs", random_state=42, class_weight=class_weight_dict)
lr.fit(X_train_resampled, y_train_resampled)
y_pred = lr.predict(X_test)


print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm.keras import TqdmCallback
from imblearn.over_sampling import SMOTE


tf.config.set_visible_devices([], "GPU")  # Hide GPUs from TensorFlow


df_copy = df.copy()


scaler = StandardScaler()
X = scaler.fit_transform(df_copy.drop("Class", axis=1))
y = df_copy["Class"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


input_dim = X_train.shape[1]
encoding_dim = 14


input_layer = Input(shape=(input_dim,))
encoder = Dense(28, activation="relu")(input_layer)
encoder = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(28, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)


autoencoder = Model(inputs=input_layer, outputs=decoder)


autoencoder.compile(optimizer="adam", loss="mse")


autoencoder.fit(
    X_train_resampled,
    X_train_resampled,
    epochs=30,
    batch_size=32,
    shuffle=True,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)],
)


reconstructed_X_test = autoencoder.predict(X_test, verbose=0)
mse = np.mean(np.power(X_test - reconstructed_X_test, 2), axis=1)
threshold = np.quantile(mse, 0.95)
y_pred = (mse > threshold).astype(int)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
