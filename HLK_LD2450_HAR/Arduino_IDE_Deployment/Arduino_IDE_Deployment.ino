
#include "model.h"

// --- Configuration Constants ---
#define SEQUENCE_LENGTH_FRAMES 40
#define NUM_FEATURES 43 // Will be filled dynamically
#define BAUD_RATE 115200 // Match sensor's baud rate
#define SERIAL_SENSOR Serial1 // Use hardware serial port for sensor (e.g., ESP32, Arduino Mega)
                            // For Arduino Uno/Nano, you might need SoftwareSerial:
                            // #include <SoftwareSerial.h>
                            // SoftwareSerial mySerial(RX_PIN, TX_PIN); // RX, TX pins
                            // #define SERIAL_SENSOR mySerial

// --- Scaler Parameters (from training) ---
// These values must be *exactly* the same as those used during Python training
const float SCALER_MEAN[43] = {};
const float SCALER_SCALE[{-0.9348636229102436, -1.0270176756637757, 24.434108705350944, -26.454378743302506, 14.901512371538434, -1.0345052482143062, 50.88848744865344, -0.0038592136772917823, 1.7242234936565244, 3.9941338837594866, -4.02410715560999, 2.1247634573607233, 5.536178773388655, 1250.1682212114426, 1249.7823847818909, 1276.3002841121695, 1223.214559587052, 15.573536507453008, 1249.783315629281, 53.08572452511769, -0.01928185673139014, 1.7407176565169686, 4.014110942993698, -4.055542846021188, 2.1859095777868673, 5.600007465613597, 21.605915434978332, 23.050370193309853, 70.32676067820772, 18.818027598308607, 10.789760617194819, 19.972690224327224, 51.508733079899116, -0.04286338017027396, 10.703646343705474, 40.33157537886195, -40.46642441683177, 3.4989905343856087, 40.867677011214134, 29.3275, 0.7331875, 47.35536209220387, 14.000814641077481}] = {};

// --- Label Mapping ---
const char* ACTIVITY_LABELS[43] = {};

// --- Sensor Data Buffer (Circular Buffer) ---
float x_buffer[SEQUENCE_LENGTH_FRAMES];
float y_buffer[SEQUENCE_LENGTH_FRAMES];
float speed_buffer[SEQUENCE_LENGTH_FRAMES];
int buffer_idx = 0;
int buffer_count = 0;

// --- Feature Extraction Function (MUST match Python training script) ---
void extractFeatures(float (&features)[NUM_FEATURES]) {
    // Initialize features array to zeros
    for(int i=0; i<NUM_FEATURES; ++i) features[i] = 0.0f;

    float x_data[SEQUENCE_LENGTH_FRAMES];
    float y_data[SEQUENCE_LENGTH_FRAMES];
    float speed_data[SEQUENCE_LENGTH_FRAMES];

    // Copy data from circular buffer into contiguous arrays for easier processing
    for (int i = 0; i < SEQUENCE_LENGTH_FRAMES; ++i) {
        int read_idx = (buffer_idx + i) % SEQUENCE_LENGTH_FRAMES;
        x_data[i] = x_buffer[read_idx];
        y_data[i] = y_buffer[read_idx];
        speed_data[i] = speed_buffer[read_idx];
    }

    int feature_idx = 0;

    // Helper for statistics
    auto calculate_stats = [&](const float* data, int len, const char* prefix) {
        if (len == 0) return;
        float sum = 0.0f;
        float min_val = data[0];
        float max_val = data[0];
        for (int i = 0; i < len; ++i) {
            sum += data[i];
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        float mean_val = sum / len;

        float std_sum_sq_diff = 0.0f;
        for (int i = 0; i < len; ++i) {
            std_sum_sq_diff += (data[i] - mean_val) * (data[i] - mean_val);
        }
        float std_val = (len > 1) ? sqrt(std_sum_sq_diff / (len - 1)) : 0.0f;

        // Median calculation (simplified for fixed size, needs sorting)
        // For simplicity, we'll skip direct median in C++ as it's complex for fixed-size buffer
        // and rely on mean/std/min/max/range for now.
        // If median is critical, a full sort implementation would be needed.

        // Assign features in correct order
        // The order MUST match the Python script's feature_names array.
        // x_mm_start, x_mm_mean, x_mm_max, x_mm_min, x_mm_std, x_mm_median, x_mm_range,
        // x_mm_diff_mean, x_mm_diff_std, x_mm_diff_max, x_mm_diff_min, x_mm_diff_abs_mean, x_mm_diff_abs_max
        // ... and similarly for y_mm and speed_cm_s

        // First frame value (start)
        if (strcmp(prefix, "x_mm") == 0) features[feature_idx++] = x_data[0];
        if (strcmp(prefix, "y_mm") == 0) features[feature_idx++] = y_data[0];
        if (strcmp(prefix, "speed_cm_s") == 0) features[feature_idx++] = speed_data[0];

        features[feature_idx++] = mean_val;
        features[feature_idx++] = max_val;
        features[feature_idx++] = min_val;
        features[feature_idx++] = std_val;
        features[feature_idx++] = mean_val; // Approximation for median, or remove if not critical
        features[feature_idx++] = max_val - min_val; // Range

        // Differences
        float diffs[SEQUENCE_LENGTH_FRAMES - 1];
        float diff_sum = 0.0f;
        float diff_min_val = 0.0f;
        float diff_max_val = 0.0f;
        float diff_abs_sum = 0.0f;

        if (len > 1) {
            diff_min_val = data[1] - data[0];
            diff_max_val = data[1] - data[0];
            for (int i = 0; i < len - 1; ++i) {
                diffs[i] = data[i+1] - data[i];
                diff_sum += diffs[i];
                if (diffs[i] < diff_min_val) diff_min_val = diffs[i];
                if (diffs[i] > diff_max_val) diff_max_val = diffs[i];
                diff_abs_sum += abs(diffs[i]);
            }
            float diff_mean_val = diff_sum / (len - 1);
            float diff_abs_mean_val = diff_abs_sum / (len - 1);

            float diff_std_sum_sq_diff = 0.0f;
            if (len - 1 > 1) {
                for (int i = 0; i < len - 1; ++i) {
                    diff_std_sum_sq_diff += (diffs[i] - diff_mean_val) * (diffs[i] - diff_mean_val);
                }
                features[feature_idx++] = diff_mean_val; // diff_mean
                features[feature_idx++] = sqrt(diff_std_sum_sq_diff / (len - 2)); // diff_std
            } else {
                features[feature_idx++] = diff_mean_val;
                features[feature_idx++] = 0.0f;
            }
            features[feature_idx++] = diff_max_val; // diff_max
            features[feature_idx++] = diff_min_val; // diff_min
            features[feature_idx++] = diff_abs_mean_val; // diff_abs_mean
            features[feature_idx++] = fmax(abs(diff_min_val), abs(diff_max_val)); // diff_abs_max
        } else {
            // Not enough data for differences, set to 0
            for (int i = 0; i < 6; ++i) features[feature_idx++] = 0.0f;
        }
    };

    calculate_stats(x_data, SEQUENCE_LENGTH_FRAMES, "x_mm");
    calculate_stats(y_data, SEQUENCE_LENGTH_FRAMES, "y_mm");
    calculate_stats(speed_data, SEQUENCE_LENGTH_FRAMES, "speed_cm_s");

    // Specific features for speed
    float speed_threshold = 5.0f;
    int speed_zero_count = 0;
    float active_speed_sum = 0.0f;
    int active_speed_count = 0;

    for (int i = 0; i < SEQUENCE_LENGTH_FRAMES; ++i) {
        if (speed_data[i] < speed_threshold) {
            speed_zero_count++;
        } else {
            active_speed_sum += speed_data[i];
            active_speed_count++;
        }
    }
    features[feature_idx++] = (float)speed_zero_count; // speed_zero_count
    features[feature_idx++] = (float)speed_zero_count / SEQUENCE_LENGTH_FRAMES; // speed_zero_ratio
    features[feature_idx++] = (active_speed_count > 0) ? (active_speed_sum / active_speed_count) : 0.0f; // speed_active_mean

    float speed_active_std_val = 0.0f;
    if (active_speed_count > 1) {
        float active_speed_mean = features[feature_idx-1];
        float active_std_sum_sq_diff = 0.0f;
        for (int i = 0; i < SEQUENCE_LENGTH_FRAMES; ++i) {
            if (speed_data[i] >= speed_threshold) {
                active_std_sum_sq_diff += (speed_data[i] - active_speed_mean) * (speed_data[i] - active_speed_mean);
            }
        }
        speed_active_std_val = sqrt(active_std_sum_sq_diff / (active_speed_count - 1));
    }
    features[feature_idx++] = speed_active_std_val; // speed_active_std
}

// --- Standard Scaling Function ---
void applyScaler(float (&features)[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; ++i) {
        if (SCALER_SCALE[i] != 0.0f) {
            features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
        } else {
            // Handle cases where scale is zero (constant feature, should be 0 after scaling)
            features[i] = 0.0f;
        }
    }
}

// --- Read Sensor Data (Placeholder) ---
// In a real application, this would read from the UART of the HLK-LD2450 sensor.
// Expected format: "X_pos,Y_pos,Speed
"
bool readLD2450Data(float* x_val, float* y_val, float* speed_val) {
    if (SERIAL_SENSOR.available()) {
        String line = SERIAL_SENSOR.readStringUntil('
');
        // Example line: "-123.45,1234.56,80.12"
        int firstComma = line.indexOf(',');
        int secondComma = line.indexOf(',', firstComma + 1);

        if (firstComma != -1 && secondComma != -1) {
            *x_val = line.substring(0, firstComma).toFloat();
            *y_val = line.substring(firstComma + 1, secondComma).toFloat();
            *speed_val = line.substring(secondComma + 1).toFloat();
            return true;
        }
    }
    return false;
}

// --- Arduino Setup ---
void setup() {
    Serial.begin(115200); // For debugging output to PC
    while (!Serial); // Wait for Serial Monitor to open

    SERIAL_SENSOR.begin(BAUD_RATE); // Start sensor serial communication
    Serial.println("Arduino HAR Deployment ready.");
    Serial.println("Buffering sensor data...");

    // Initialize buffers to 0
    for(int i=0; i<SEQUENCE_LENGTH_FRAMES; ++i) {
        x_buffer[i] = 0.0f;
        y_buffer[i] = 0.0f;
        speed_buffer[i] = 0.0f;
    }
}

// --- Arduino Loop ---
void loop() {
    float current_x, current_y, current_speed;

    if (readLD2450Data(&current_x, &current_y, &current_speed)) {
        // Store data in circular buffer
        x_buffer[buffer_idx] = current_x;
        y_buffer[buffer_idx] = current_y;
        speed_buffer[buffer_idx] = current_speed;

        buffer_idx = (buffer_idx + 1) % SEQUENCE_LENGTH_FRAMES;
        if (buffer_count < SEQUENCE_LENGTH_FRAMES) {
            buffer_count++;
        }

        // Only predict once the buffer is full
        if (buffer_count == SEQUENCE_LENGTH_FRAMES) {
            float features[NUM_FEATURES]; // Declare feature array
            extractFeatures(features);
            applyScaler(features);

            // Predict activity using the generated model.h
            int prediction = model.predict(features);
            Serial.print("Predicted Activity: ");
            Serial.println(ACTIVITY_LABELS[prediction]);
            Serial.print("Raw Prediction Index: ");
            Serial.println(prediction);

            // You can also get probabilities if model.h provides it
            // float *probabilities = model.predict_proba(features);
            // for (int i=0; i<model.n_classes; ++i) {
            //     Serial.print(ACTIVITY_LABELS[i]);
            //     Serial.print(": ");
            //     Serial.println(probabilities[i]);
            // }
        }
    }
    delay(10); // Process sensor data at approx 10Hz (100ms interval for each frame)
               // or adjust based on actual sensor output rate to not miss frames.
}
