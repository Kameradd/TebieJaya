#include "DHT.h"

// --- Pin Definitions ---
#define DHTPIN 4
#define DHTTYPE DHT22  

#define PIR_PIN 5        // PIR motion sensor digital pin
#define MIC_PIN 34       // MAX4466 analog output to ADC pin (use a 3.3V ADC pin)

// Threshold (rough approximation, not calibrated to real dB SPL)
#define DB_THRESHOLD 50  

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17);

  dht.begin();
  pinMode(PIR_PIN, INPUT);
}

void loop() {
  delay(2000);

  // --- DHT22 ---
  float humidity = dht.readHumidity();
  float temperature_c = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature_c)) {
    Serial2.println("Failed to read from DHT sensor!");
    Serial.println("Failed Bos");
    return;
  }

  // --- PIR ---
  int motion = digitalRead(PIR_PIN);  // 1 = motion detected, 0 = none

  // --- MAX4466 Approximate dB ---
  // Take multiple samples for RMS
  const int samples = 100;
  long sumsq = 0;
  for (int i = 0; i < samples; i++) {
    int16_t micValue = analogRead(MIC_PIN) - 2048; // center at 0 (ESP32 ADC ~0–4095)
    sumsq += micValue * micValue;
  }
  float rms = sqrt((float)sumsq / samples);

  // Convert RMS to "dB-like" scale (uncalibrated)
  float dB = 20.0 * log10(rms);

  // Apply threshold
  bool sound_detected = (dB > DB_THRESHOLD);

  // --- Payload ---
  String dataString = "TEMP:" + String(temperature_c, 1) +
                      ",HUM:" + String(humidity, 1) +
                      ",MOTION:" + String(motion) +
                      ",SOUND:" + String(sound_detected);

  Serial2.println(dataString);
  Serial.println(dataString);
}
