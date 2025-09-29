#include <WiFi.h>
#include <PubSubClient.h>
#include "DHT.h"
#include <time.h>
#include <WiFiClientSecure.h>
// --- WiFi ---
const char* ssid = "Cantika";
const char* password = "cantikaacantik";

// MQTT Broker settings 
const int mqtt_port = 8883;// MQTT port (TLS) 
const char *mqtt_broker = "REDACTED"; // EMQX broker endpoint 
const char *mqtt_topic = "REDACTED"; // MQTT topic 
const char *mqtt_username = "REDACTED"; // MQTT username for authentication 
const char *mqtt_password = "REDACTED"; // MQTT password for authentication 

// NTP Server settings
const char *ntp_server = "pool.ntp.org";     // Default NTP server
// const char* ntp_server = "cn.pool.ntp.org"; // Recommended NTP server for users in China
const long gmt_offset_sec = 7 * 3600;       // adjust to your timezone
const int daylight_offset_sec = 0;


WiFiClientSecure espClient;
PubSubClient mqtt_client(espClient);

static const char ca_cert[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDrzCCApegAwIBAgIQCDvgVpBCRrGhdWrJWZHHSjANBgkqhkiG9w0BAQUFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBD
QTAeFw0wNjExMTAwMDAwMDBaFw0zMTExMTAwMDAwMDBaMGExCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j
b20xIDAeBgNVBAMTF0RpZ2lDZXJ0IEdsb2JhbCBSb290IENBMIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4jvhEXLeqKTTo1eqUKKPC3eQyaKl7hLOllsB
CSDMAZOnTjC3U/dDxGkAV53ijSLdhwZAAIEJzs4bg7/fzTtxRuLWZscFs3YnFo97
nh6Vfe63SKMI2tavegw5BmV/Sl0fvBf4q77uKNd0f3p4mVmFaG5cIzJLv07A6Fpt
43C/dxC//AH2hdmoRBBYMql1GNXRor5H4idq9Joz+EkIYIvUX7Q6hL+hqkpMfT7P
T19sdl6gSzeRntwi5m3OFBqOasv+zbMUZBfHWymeMr/y7vrTC0LUq7dBMtoM1O/4
gdW7jVg/tRvoSSiicNoxBN33shbyTApOB6jtSj1etX+jkMOvJwIDAQABo2MwYTAO
BgNVHQ8BAf8EBAMCAYYwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUA95QNVbR
TLtm8KPiGxvDl7I90VUwHwYDVR0jBBgwFoAUA95QNVbRTLtm8KPiGxvDl7I90VUw
DQYJKoZIhvcNAQEFBQADggEBAMucN6pIExIK+t1EnE9SsPTfrgT1eXkIoyQY/Esr
hMAtudXH/vTBH1jLuG2cenTnmCmrEbXjcKChzUyImZOMkXDiqw8cvpOp/2PV5Adg
06O/nVsJ8dWO41P0jmP6P6fbtGbfYmbW0W5BjfIttep3Sp+dWOIrWcBAI+0tKIJF
PnlUkiaY4IBIqDfv8NZ5YBberOgOzW6sRBc4L0na4UU+Krk2U886UAb3LujEV0ls
YSEY1QSteDwsOoBrp+uvFRTp2InBuThs4pFsiv9kuXclVzDAGySj4dzp30d8tbQk
CAUw7C29C79Fv1C5qfPrmAESrciIxpg0X40KPMbp1ZWVbd4=
-----END CERTIFICATE-----
)EOF";

// --- Sensors ---
#define DHTPIN 4
#define DHTTYPE DHT22
#define PIR_PIN 5
#define MIC_PIN 34
#define DB_THRESHOLD 50

DHT dht(DHTPIN, DHTTYPE);

// ---- Functions ----
void connectToWiFi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void syncTime() {
  configTime(gmt_offset_sec, daylight_offset_sec, ntp_server);
  Serial.print("Waiting for NTP time sync");
  while (time(nullptr) < 8 * 3600 * 2) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println(" done");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

void connectToMQTT() {
  espClient.setCACert(ca_cert);
  while (!mqtt_client.connected()) {
    String client_id = "esp32-client-" + String(WiFi.macAddress());
    Serial.printf("Connecting to MQTT Broker as %s...\n", client_id.c_str());

    if (mqtt_client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("Connected to MQTT broker (TLS)");
      mqtt_client.subscribe(mqtt_topic);
      mqtt_client.publish(mqtt_topic, "Hello EMQX from ESP32 over TLS!");
    } else {
      Serial.print("Failed, rc=");
      Serial.println(mqtt_client.state());
      delay(5000);
    }
  }
}


void setup() {
  Serial.begin(115200);
  connectToWiFi();
  syncTime();
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqttCallback);
  connectToMQTT();
  dht.begin();
  pinMode(PIR_PIN, INPUT);
}

void loop() {
  if (!mqtt_client.connected()) {
    connectToMQTT();
  }
  mqtt_client.loop();

  delay(2000);

  // --- Read DHT22 ---
  float humidity = dht.readHumidity();
  float temperature_c = dht.readTemperature();
  if (isnan(humidity) || isnan(temperature_c)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // --- Read PIR ---
  int motion = digitalRead(PIR_PIN);

  // --- Read MIC (RMS approx) ---
  const int samples = 100;
  long sumsq = 0;
  for (int i = 0; i < samples; i++) {
    int16_t micValue = analogRead(MIC_PIN) - 2048;
    sumsq += micValue * micValue;
  }
  float rms = sqrt((float)sumsq / samples);
  float dB = 20.0 * log10(rms);
  bool sound_detected = (dB > DB_THRESHOLD);

  // --- JSON payload ---
  String payload = "{";
  payload += "\"temp\":" + String(temperature_c, 1) + ",";
  payload += "\"hum\":" + String(humidity, 1) + ",";
  payload += "\"motion\":" + String(motion) + ",";
  payload += "\"sound\":" + String(sound_detected ? 1 : 0);
  payload += "}";

  // --- Publish ---
  Serial.print("Publishing payload: ");
  Serial.println(payload);

  mqtt_client.publish(mqtt_topic, payload.c_str());
}
