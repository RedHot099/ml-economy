# %%
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)


# %%
symbol = "DAX"
ticker = yf.Ticker(symbol)

hist_data = ticker.history(period="2y")
# Prepare the data
data = hist_data[["Close"]].copy()
data.reset_index(inplace=True)

# Create train-test split (last 2 months as test data)
test_size = 45
train_data = data.iloc[:-test_size]
test_data = data.iloc[-test_size:]

print(f"Training data: {train_data.shape[0]} days")
print(f"Test data: {test_data.shape[0]} days")
print(f"Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[["Close"]])
test_scaled = scaler.transform(test_data[["Close"]])


# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 20  # Number of previous days to use for prediction
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Build LSTM model
model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model with early stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
train_mape = np.mean(np.abs((y_train_inv - train_predict) / y_train_inv)) * 100
test_mape = np.mean(np.abs((y_test_inv - test_predict) / y_test_inv)) * 100

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Test MAPE: {test_mape:.2f}%")

# Calculate baseline metrics
baseline_predictions = y_test_inv[:-1]  # Previous day's actual value
baseline_targets = y_test_inv[1:]  # Current day's actual value
baseline_rmse = np.sqrt(mean_squared_error(baseline_targets, baseline_predictions))
baseline_mape = (
    np.mean(np.abs((baseline_targets - baseline_predictions) / baseline_targets)) * 100
)

print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Baseline MAPE: {baseline_mape:.2f}%")

# %%

# Plot the results
plt.figure(figsize=(16, 8))

# Plot the entire dataset
plt.plot(data["Date"], data["Close"], label="Actual Close Price", color="blue")

# Plot test predictions (accounting for sequence length)
test_dates = test_data["Date"][seq_length:].reset_index(drop=True)
plt.plot(test_dates, test_predict, label="LSTM Prediction", color="red", linewidth=2)

# Add a vertical line to separate train and test data
train_end_date = train_data["Date"].iloc[-1]
plt.axvline(x=train_end_date, color="gray", linestyle="--", alpha=0.7)
plt.text(
    train_end_date,
    plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
    " Train | Test ",
    rotation=90,
    verticalalignment="bottom",
)

plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Add metrics as text on the plot
plt.figtext(
    0.15,
    0.15,
    f"Test RMSE: {test_rmse:.2f}\nTest MAPE: {test_mape:.2f}%\nBaseline RMSE: {baseline_rmse:.2f}",
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.savefig("figs/results.png")
plt.show()


# Create date range for test predictions (accounting for sequence length)
test_dates = test_data["Date"][seq_length:].reset_index(drop=True)

# Plot actual vs predicted for test period
plt.plot(test_dates, y_test_inv, label="Actual Close Price")
plt.plot(test_dates, test_predict, label="LSTM Prediction")

plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("figs/test.png")
plt.show()


# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figs/loss.png")
plt.show()
