/*
  SAMS — MPU6050 + simple buzzer on D8

  Board: Arduino Uno / Nano AVR
    arduino-cli compile/upload --fqbn arduino:avr:uno …

  Buzzer: + → D8, − → GND (active), or 3-pin VCC/GND/I/O→D8
  B1/B2 → ON (solid HIGH)   B0 → OFF
*/

#include <Wire.h>

const int MPU_ADDR = 0x68;
const int BUZZER_PIN = 8;
const int STATUS_LED_G = 5;
const int STATUS_LED_Y = 6;
const int STATUS_LED_R = 7;
const unsigned long SAMPLE_US = 20000;

enum BuzzMode { BUZZ_OFF = 0, BUZZ_ON = 1 };
volatile BuzzMode buzzMode = BUZZ_OFF;

String cmdBuf;

int16_t read16(int reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)2);
  return (Wire.read() << 8) | Wire.read();
}

void setStatusLeds(bool g, bool y, bool r) {
  digitalWrite(STATUS_LED_G, g ? HIGH : LOW);
  digitalWrite(STATUS_LED_Y, y ? HIGH : LOW);
  digitalWrite(STATUS_LED_R, r ? HIGH : LOW);
}

void buzzerOff() {
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  buzzMode = BUZZ_OFF;
}

void buzzerOn() {
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  buzzMode = BUZZ_ON;
}

void applyCommand(const String& line) {
  if (line.length() == 0) return;
  if (line == "B0") {
    buzzerOff();
    Serial.println("# ACK B0");
  } else if (line == "B1" || line == "B2") {
    buzzerOn();
    Serial.print("# ACK ");
    Serial.println(line);
  } else if (line == "L0") setStatusLeds(false, false, false);
  else if (line == "LG") setStatusLeds(true, false, false);
  else if (line == "LY") setStatusLeds(false, true, false);
  else if (line == "LR") setStatusLeds(false, false, true);
}

void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      cmdBuf.trim();
      if (cmdBuf.length()) applyCommand(cmdBuf);
      cmdBuf = "";
    } else if (cmdBuf.length() < 16) {
      cmdBuf += c;
    } else {
      cmdBuf = "";
    }
  }
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(STATUS_LED_G, OUTPUT);
  pinMode(STATUS_LED_Y, OUTPUT);
  pinMode(STATUS_LED_R, OUTPUT);
  buzzerOff();
  setStatusLeds(false, false, false);

  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}

  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(50);

  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("# SAMS MPU6050 ready (buzzer=D8)");
}

void loop() {
  handleSerialCommands();
  if (buzzMode == BUZZ_ON) {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  }

  static unsigned long last = 0;
  unsigned long now = micros();
  if (now - last < SAMPLE_US) return;
  last = now;

  handleSerialCommands();

  float ax = read16(0x3B) / 16384.0f;
  float ay = read16(0x3D) / 16384.0f;
  float az = read16(0x3F) / 16384.0f;

  Serial.print(ax, 4);
  Serial.print(',');
  Serial.print(ay, 4);
  Serial.print(',');
  Serial.println(az, 4);
}
