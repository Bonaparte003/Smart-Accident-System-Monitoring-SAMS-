/*
  SAMS — electrical check of the MPU Accelerometer on A2 (SDA) / A3 (SCL)
*/

const uint8_t SDA_PIN = A2;
const uint8_t SCL_PIN = A3;

unsigned int halfUs = 100;

void dly() { delayMicroseconds(halfUs); }
void lineLow(uint8_t p) { digitalWrite(p, LOW); pinMode(p, OUTPUT); }
void lineHigh(uint8_t p) { pinMode(p, INPUT); }

float volts(uint8_t p) {
  pinMode(p, INPUT);
  delay(2);
  analogRead(p);
  long sum = 0;
  for (int i = 0; i < 16; i++) sum += analogRead(p);
  return sum / 16.0f * 5.0f / 1023.0f;
}

void startCond() { lineHigh(SDA_PIN); lineHigh(SCL_PIN); dly(); lineLow(SDA_PIN); dly(); lineLow(SCL_PIN); dly(); }
void stopCond() { lineLow(SDA_PIN); dly(); lineHigh(SCL_PIN); dly(); lineHigh(SDA_PIN); dly(); }

bool writeByte(uint8_t v) {
  for (uint8_t b = 0x80; b; b >>= 1) {
    if (v & b) lineHigh(SDA_PIN); else lineLow(SDA_PIN);
    dly(); lineHigh(SCL_PIN); dly(); lineLow(SCL_PIN);
  }
  lineHigh(SDA_PIN); dly(); lineHigh(SCL_PIN); dly();
  bool ack = digitalRead(SDA_PIN) == LOW;
  lineLow(SCL_PIN); dly();
  return ack;
}

uint8_t readByte() {
  uint8_t v = 0;
  lineHigh(SDA_PIN);
  for (uint8_t b = 0x80; b; b >>= 1) {
    dly(); lineHigh(SCL_PIN); dly();
    if (digitalRead(SDA_PIN)) v |= b;
    lineLow(SCL_PIN);
  }
  lineHigh(SDA_PIN); dly(); lineHigh(SCL_PIN); dly(); lineLow(SCL_PIN);
  return v;
}

void recoverBus() {
  lineHigh(SDA_PIN);
  for (int i = 0; i < 9; i++) { lineLow(SCL_PIN); dly(); lineHigh(SCL_PIN); dly(); }
  stopCond();
}

int whoAmI(uint8_t addr) {
  startCond();
  if (!writeByte(addr << 1)) { stopCond(); return -1; }
  if (!writeByte(0x75)) { stopCond(); return -2; }
  startCond();
  if (!writeByte((addr << 1) | 1)) { stopCond(); return -1; }
  int v = readByte();
  stopCond();
  return v;
}

void report() {
  Serial.println(F("# ---- MPU6050 diag ----"));
  float vSda = volts(SDA_PIN);
  float vScl = volts(SCL_PIN);
  Serial.print(F("# idle volts SDA(A2)="));
  Serial.print(vSda, 2);
  Serial.print(F(" SCL(A3)="));
  Serial.println(vScl, 2);

  lineLow(SDA_PIN);
  delay(2);
  float sclWhileSdaLow = volts(SCL_PIN);
  lineHigh(SDA_PIN);
  lineLow(SCL_PIN);
  delay(2);
  float sdaWhileSclLow = volts(SDA_PIN);
  lineHigh(SCL_PIN);
  Serial.print(F("# short test: SCL with SDA held low="));
  Serial.print(sclWhileSdaLow, 2);
  Serial.print(F("  SDA with SCL held low="));
  Serial.println(sdaWhileSclLow, 2);

  lineLow(SDA_PIN);
  delay(1);
  pinMode(SDA_PIN, INPUT);
  unsigned long t0 = micros();
  while (digitalRead(SDA_PIN) == LOW && micros() - t0 < 2000) {}
  unsigned long riseSda = micros() - t0;
  lineLow(SCL_PIN);
  delay(1);
  pinMode(SCL_PIN, INPUT);
  t0 = micros();
  while (digitalRead(SCL_PIN) == LOW && micros() - t0 < 2000) {}
  unsigned long riseScl = micros() - t0;
  Serial.print(F("# rise us SDA="));
  Serial.print(riseSda);
  Serial.print(F(" SCL="));
  Serial.println(riseScl);

  const unsigned int speeds[2] = {100, 10};
  for (uint8_t s = 0; s < 2; s++) {
    halfUs = speeds[s];
    recoverBus();
    for (uint8_t addr = 0x68; addr <= 0x69; addr++) {
      int r = whoAmI(addr);
      Serial.print(F("# half-period "));
      Serial.print(halfUs);
      Serial.print(F("us addr 0x"));
      Serial.print(addr, HEX);
      Serial.print(F(": "));
      if (r == -1) Serial.println(F("no ACK"));
      else if (r == -2) Serial.println(F("address ACK, register NACK"));
      else { Serial.print(F("WHO_AM_I=0x")); Serial.println(r, HEX); }
    }
  }
  Serial.println(F("# ---- done ----"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}
  lineHigh(SDA_PIN);
  lineHigh(SCL_PIN);
  Serial.println(F("# SAMS MPU6050 diag (SDA=A2 SCL=A3)"));
}

void loop() {
  report();
  delay(5000);
}
