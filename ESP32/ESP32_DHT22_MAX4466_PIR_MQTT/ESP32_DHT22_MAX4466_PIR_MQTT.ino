#include <WiFi.h>
#include <PubSubClient.h>
#include "DHT.h"
#include <time.h>

// --- WiFi ---
const char* ssid = "Cantika";
const char* password = "cantikaacantik";

// --- MQTT / EMQX ---
const char* mqtt_server = "broker.emqx.io";   // or your own EMQX host
const int mqtt_port = 8883;                   // 8883 if using TLS
const char* mqtt_client_id = "";
const char* mqtt_user = "";       // EMQX username (API key)
const char* mqtt_pass = "";    // EMQX secret/password
const char* mqtt_topic = "esp32";     // topic to publish

// NTP Server settings
const char *ntp_server = "pool.ntp.org";     // Default NTP server
// const char* ntp_server = "cn.pool.ntp.org"; // Recommended NTP server for users in China
const long gmt_offset_sec = 7 * 3600;       // adjust to your timezone
const int daylight_offset_sec = 0;


BearSSL::WiFiClientSecure espClient;
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

void setup_wifi() {
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected, IP address: ");
  Serial.println(WiFi.localIP());
  syncTime();  // X.509 validation requires synchronization time
}

void connectToMQTT() {
    BearSSL::X509List serverTrustedCA(ca_cert);
    espClient.setTrustAnchors(&serverTrustedCA);
    while (!mqtt_client.connected()) {
        String client_id = "esp8266-client-" + String(WiFi.macAddress());
        Serial.printf("Connecting to MQTT Broker as %s.....\n", client_id.c_str());
        if (mqtt_client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
            Serial.println("Connected to MQTT broker");
            mqtt_client.subscribe(mqtt_topic);
            // Publish message upon successful connection
            mqtt_client.publish(mqtt_topic, "Hi EMQX I'm ESP8266 ^^");
        } else {
            char err_buf[128];
            espClient.getLastSSLError(err_buf, sizeof(err_buf));
            Serial.print("Failed to connect to MQTT broker, rc=");
            Serial.println(mqtt_client.state());
            Serial.print("SSL error: ");
            Serial.println(err_buf);
            delay(5000);
        }
    }
}

void mqttCallback(char *topic, byte *payload, unsigned int length) {
    Serial.print("Message received on topic: ");
    Serial.print(topic);
    Serial.print("]: ");
    for (int i = 0; i < length; i++) {
        Serial.print((char) payload[i]);
    }
    Serial.println();
}

void syncTime() {
    configTime(gmt_offset_sec, daylight_offset_sec, ntp_server);
    Serial.print("Waiting for NTP time sync: ");
    while (time(nullptr) < 8 * 3600 * 2) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("Time synchronized");
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
        Serial.print("Current time: ");
        Serial.println(asctime(&timeinfo));
    } else {
        Serial.println("Failed to obtain local time");
    }
}


void setup() {
  Serial.begin(115200);

  dht.begin();
  pinMode(PIR_PIN, INPUT);

  setup_wifi();
  mqtt_client.setServer(mqtt_server, mqtt_port);
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
