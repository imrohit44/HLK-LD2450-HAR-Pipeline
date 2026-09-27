import pandas as pd
import numpy as np
import joblib
import serial
import time
import collections # For fixed-size buffer

# --- Configuration Constants ---
MODEL_FILE = 'har_model.joblib'
SCALER_FILE = 'scaler.joblib'
LABEL_ENCODER_FILE = 'label_encoder.joblib'
SEQUENCE_LENGTH_FRAMES = 40 # Must match training data
SERIAL_PORT = '/dev/ttyUSB0' # Adjust this for your specific serial port
BAUD_RATE = 115200 # Adjust this for your sensor's baud rate

# --- Load Model Components ---
print("Loading model, scaler, and label encoder...")
loaded_model = joblib.load(MODEL_FILE)
loaded_scaler = joblib.load(SCALER_FILE)
loaded_label_encoder = joblib.load(LABEL_ENCODER_FILE)
print("Model components loaded successfully.")

# --- Feature Extraction Function (MUST match training script) ---
def extract_features(group):
    features = {}
    for col in ['x_mm', 'y_mm', 'speed_cm_s']:
        features[f'{col}_start'] = group[col].iloc[0]
        features[f'{col}_mean'] = group[col].mean()
        features[f'{col}_max'] = group[col].max()
        features[f'{col}_min'] = group[col].min()
        features[f'{col}_std'] = group[col].std()
        features[f'{col}_median'] = group[col].median()
        features[f'{col}_range'] = group[col].max() - group[col].min()
        diffs = group[col].diff().dropna()
        if not diffs.empty:
            features[f'{col}_diff_mean'] = diffs.mean()
            features[f'{col}_diff_std'] = diffs.std()
            features[f'{col}_diff_max'] = diffs.max()
            features[f'{col}_diff_min'] = diffs.min()
            features[f'{col}_diff_abs_mean'] = np.mean(np.abs(diffs))
            features[f'{col}_diff_abs_max'] = np.max(np.abs(diffs))
        else:
            features[f'{col}_diff_mean'] = 0.0 ; features[f'{col}_diff_std'] = 0.0
            features[f'{col}_diff_max'] = 0.0 ; features[f'{col}_diff_min'] = 0.0
            features[f'{col}_diff_abs_mean'] = 0.0 ; features[f'{col}_diff_abs_max'] = 0.0
    speed_threshold = 5
    features['speed_zero_count'] = (group['speed_cm_s'] < speed_threshold).sum()
    features['speed_zero_ratio'] = features['speed_zero_count'] / len(group)
    active_speeds = group['speed_cm_s'][group['speed_cm_s'] >= speed_threshold]
    features['speed_active_mean'] = active_speeds.mean() if not active_speeds.empty else 0.0
    features['speed_active_std'] = active_speeds.std() if len(active_speeds) > 1 else 0.0
    return pd.Series(features)

# --- Live Inference Function ---
def live_predict(sensor_data_buffer: collections.deque):
    """
    Takes a deque of raw sensor frames, extracts features, scales, and predicts activity.
    """
    # Convert deque to DataFrame for feature extraction
    df_sequence = pd.DataFrame(list(sensor_data_buffer), columns=['x_mm', 'y_mm', 'speed_cm_s'])
    
    # Add a dummy sequence_id and label for compatibility with extract_features
    df_sequence['sequence_id'] = 0
    df_sequence['label'] = 'dummy'

    # Extract features
    features_raw = df_sequence.groupby('sequence_id').apply(extract_features).drop('label', axis=1)

    # Ensure feature order matches training (important for scaler and model)
    # The feature_names list should be consistent between training and inference
    global feature_names # Access global feature names if defined, or hardcode them
    if 'feature_names' in globals():
        features_raw = features_raw[feature_names]
    else:
        print("Warning: 'feature_names' not defined globally. Assuming feature order is consistent.")

    # Scale features
    features_scaled = loaded_scaler.transform(features_raw)

    # Predict activity
    prediction_encoded = loaded_model.predict(features_scaled)[0]
    prediction_label = loaded_label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = loaded_model.predict_proba(features_scaled)[0]
    prob_dict = {label: prob for label, prob in zip(loaded_label_encoder.classes_, probabilities)}

    return prediction_label, prob_dict

# --- Main Serial Reading Loop ---
if __name__ == '__main__':
    # Initialize serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1);
        print(f"Connected to serial port {SERIAL_PORT} at {BAUD_RATE} bps")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Please ensure the sensor is connected and the port is correct.")
        print("Exiting live inference script.")
        exit()

    # Circular buffer to store sensor frames
    sensor_data_buffer = collections.deque(maxlen=SEQUENCE_LENGTH_FRAMES)

    print(f"Buffering {SEQUENCE_LENGTH_FRAMES} frames before first prediction...")
    
    try:
        while True:
            # Read a line from the serial port
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        # Assuming data format: X_position, Y_position, Speed (comma-separated)
                        # Example: -123.45, 1234.56, 80.12
                        x_str, y_str, speed_str = line.split(',')
                        x = float(x_str.strip())
                        y = float(y_str.strip())
                        speed = float(speed_str.strip())
                        
                        # Add data to buffer
                        sensor_data_buffer.append({'x_mm': x, 'y_mm': y, 'speed_cm_s': speed})
                        
                        # Once buffer is full, perform prediction
                        if len(sensor_data_buffer) == SEQUENCE_LENGTH_FRAMES:
                            predicted_activity, probabilities = live_predict(sensor_data_buffer)
                            print(f"
Predicted Activity: {predicted_activity}")
                            print("Probabilities:")
                            for act, prob in probabilities.items():
                                print(f"  {act}: {prob:.2f}")
                            # Optionally clear buffer or just let it continue as a sliding window
                            # sensor_data_buffer.clear() # Uncomment to predict only once per full buffer

                    except ValueError as ve:
                        print(f"Error parsing data: {line} - {ve}")
                    except Exception as e:
                        print(f"An unexpected error occurred during prediction: {e}")
            time.sleep(0.01) # Small delay to prevent busy-waiting

    except KeyboardInterrupt:
        print("
Live inference stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
