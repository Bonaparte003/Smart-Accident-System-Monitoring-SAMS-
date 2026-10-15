/*
  SAMS — find the MPU accelerometer
*/

#include "SoftI2C.h"

const uint8_t PINS[] = {A0, A1, A2, A3, A4, A5, 4, 6, 7, 9, 10, 11, 12};
const uint8_t N_PINS = sizeof(PINS);

void printPin(uint8_t pin) {
  if (pin >= A0) {
    Serial.print('A');
    Serial.print(pin - A0);
  } else {
    Serial.print('D');
    Serial.print(pin);
  }
}

bool pulledUp(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
  delay(1);
  pinMode(pin, INPUT);
  delay(1);
  bool up = digitalRead(pin) == HIGH;
  pinMode(pin, INPUT);
  return up;
}

bool answers(SoftI2C& bus, uint8_t addr) {
  bus.beginTransmission(addr);
  return bus.endTransmission() == 0;
}

void release(uint8_t sda, uint8_t scl) {
  pinMode(sda, INPUT);
  pinMode(scl, INPUT);
}

void scan() {
  Serial.println(F("# ---- scan ----"));
  bool up[N_PINS];
  uint8_t nUp = 0;
  Serial.print(F("# pulled up:"));
  for (uint8_t i = 0; i < N_PINS; i++) {
    up[i] = pulledUp(PINS[i]);
    if (up[i]) {
      Serial.print(' ');
      printPin(PINS[i]);
      nUp++;
    }
  }
  Serial.println(nUp ? F("") : F(" none"));

  for (uint8_t i = 0; i < N_PINS; i++) {
    if (!up[i]) continue;
    for (uint8_t j = 0; j < N_PINS; j++) {
      if (i == j || !up[j]) continue;
      SoftI2C bus(PINS[i], PINS[j]);
      bus.begin();
      uint8_t found = 0;
      for (uint8_t addr = 1; addr < 127; addr++) {
        if (!answers(bus, addr)) continue;
        if (!found) {
          Serial.print(F("# SDA="));
          printPin(PINS[i]);
          Serial.print(F(" SCL="));
          printPin(PINS[j]);
          Serial.print(F(" devices:"));
        }
        Serial.print(F(" 0x"));
        Serial.print(addr, HEX);
        found++;
      }
      if (found) Serial.println();
      release(PINS[i], PINS[j]);
    }
  }

  for (uint8_t i = 0; i < N_PINS; i++) {
    for (uint8_t j = 0; j < N_PINS; j++) {
      if (i == j || (up[i] && up[j])) continue;
      SoftI2C bus(PINS[i], PINS[j]);
      bus.begin();
      for (uint8_t addr = 0x68; addr <= 0x69; addr++) {
        if (!answers(bus, addr)) continue;
        Serial.print(F("# MPU6? SDA="));
        printPin(PINS[i]);
        Serial.print(F(" SCL="));
        printPin(PINS[j]);
        Serial.print(F(" at 0x"));
        Serial.println(addr, HEX);
      }
      release(PINS[i], PINS[j]);
    }
  }
  Serial.println(F("#==== done"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}
  Serial.println(F("# SAMS I2C pin scan"));
}

void loop() {
  scan();
  delay(5000);
}
