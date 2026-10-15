/*
  SAMS — MPU Accelerometer+ VL53L0X + buzzer on D8 + motor on D9

*/

#include "VL53L0X.h"

uint8_t mpuAddr = 0x68;
const int MPU_SDA = A2;
const int MPU_SCL = A3;
const int TOF_SDA = A0;
const int TOF_SCL = A1;
const int BUZZER_PIN = 8;
const int STATUS_LED_G = 2;
const int STATUS_LED_Y = 3;
const int STATUS_LED_R = 5;
const int MOTOR_PIN = 9;
const unsigned long BOOT_SEQ_GUARD_MS = 5000;
const unsigned long SAMPLE_US = 20000;
const unsigned long TOF_PERIOD_MS = 200;

SoftI2C mpuBus(MPU_SDA, MPU_SCL);
SoftI2C tofBus(TOF_SDA, TOF_SCL);
VL53L0X tof;
bool tofOk = false;

enum BuzzMode { BUZZ_OFF = 0, BUZZ_ON = 1 };
volatile BuzzMode buzzMode = BUZZ_OFF;

enum MotorMode { MOTOR_OFF = 0, MOTOR_PULSE = 1, MOTOR_ON = 2 };
MotorMode motorMode = MOTOR_OFF;
int motorDuty = 255;
const unsigned long MOTOR_PULSE_ON_MS = 2000;
const unsigned long MOTOR_PULSE_PERIOD_MS = 3000;

String cmdBuf;

bool wakeMpuAt(uint8_t addr) {
  mpuBus.beginTransmission(addr);
  mpuBus.write(0x6B);
  mpuBus.write(0);
  return mpuBus.endTransmission() == 0;
}

int readMpuReg(uint8_t reg) {
  mpuBus.beginTransmission(mpuAddr);
  mpuBus.write(reg);
  if (mpuBus.endTransmission() != 0) return -1;
  if (mpuBus.requestFrom(mpuAddr, 1) != 1) return -1;
  return mpuBus.read();
}

// The MPU6050, MPU6500 and MPU9250 share the wake (0x6B) and accel (0x3B) registers
// and the ±2 g default, so one driver reads them all; WHO_AM_I says which is wired.
int mpuWhoAmI = -1;

const __FlashStringHelper* mpuChip() {
  switch (mpuWhoAmI) {
    case 0x68: return F("MPU6050");
    case 0x70: return F("MPU6500");
    case 0x71: return F("MPU9250");
    case 0x73: return F("MPU9255");
    default: return F("unknown");
  }
}

bool wakeMpu() {
  static bool announced = false;
  const uint8_t addrs[2] = {0x68, 0x69};
  for (uint8_t i = 0; i < 2; i++) {
    if (!wakeMpuAt(addrs[i])) continue;
    bool changed = mpuAddr != addrs[i];
    mpuAddr = addrs[i];
    if (changed || !announced) {
      announced = true;
      mpuWhoAmI = readMpuReg(0x75);
      Serial.print(F("# "));
      Serial.print(mpuChip());
      Serial.print(F(" found at 0x"));
      Serial.print(mpuAddr, HEX);
      Serial.print(F(" WHO_AM_I=0x"));
      Serial.println(mpuWhoAmI, HEX);
    }
    return true;
  }
  return false;
}

// A line is pulled up by the module when it rises back after being driven low;
// a pin with nothing on it keeps reading low.
bool pulledUp(int pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
  delay(1);
  pinMode(pin, INPUT);
  delay(1);
  return digitalRead(pin) == HIGH;
}

uint8_t scanBus(SoftI2C& bus) {
  uint8_t found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    bus.beginTransmission(addr);
    if (bus.endTransmission() == 0) {
      Serial.print(F(" 0x"));
      Serial.print(addr, HEX);
      found++;
    }
  }
  Serial.println(found ? F("") : F(" none"));
  return found;
}

void reportI2cBus() {
  bool sda = pulledUp(MPU_SDA);
  bool scl = pulledUp(MPU_SCL);
  Serial.print(F("# I2C A2/A3 pull-ups from the module: SDA(A2)="));
  Serial.print(sda ? F("yes") : F("no"));
  Serial.print(F(" SCL(A3)="));
  Serial.println(scl ? F("yes") : F("no"));
  Serial.print(F("# I2C A2/A3 devices:"));
  mpuBus.begin();
  if (scanBus(mpuBus)) return;
  SoftI2C swapped(MPU_SCL, MPU_SDA);
  swapped.begin();
  Serial.print(F("# I2C with SDA/SCL swapped (A3/A2) devices:"));
  if (scanBus(swapped)) Serial.println(F("# The MPU6050 SDA and SCL wires are swapped: SDA goes to A2, SCL to A3"));
  mpuBus.begin();
}

bool readAccel(float& ax, float& ay, float& az) {
  mpuBus.beginTransmission(mpuAddr);
  mpuBus.write(0x3B);
  if (mpuBus.endTransmission() != 0) return false;
  if (mpuBus.requestFrom(mpuAddr, 6) != 6) return false;
  int16_t raw[3];
  for (int i = 0; i < 3; i++) {
    uint8_t hi = mpuBus.read();
    uint8_t lo = mpuBus.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  ax = raw[0] / 16384.0f;
  ay = raw[1] / 16384.0f;
  az = raw[2] / 16384.0f;
  return true;
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

void setMotorMode(MotorMode mode, int duty = 255) {
  motorMode = mode;
  motorDuty = constrain(duty, 0, 255);
  analogWrite(MOTOR_PIN, mode == MOTOR_ON ? motorDuty : 0);
}

void updateMotor() {
  if (motorMode != MOTOR_PULSE) return;
  bool on = (millis() % MOTOR_PULSE_PERIOD_MS) < MOTOR_PULSE_ON_MS;
  analogWrite(MOTOR_PIN, on ? motorDuty : 0);
}
void startupSequence() {
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);

  setStatusLeds(true, false, false);
  delay(300);
  setStatusLeds(false, true, false);
  delay(300);
  setStatusLeds(false, false, true);
  delay(300);
  setStatusLeds(false, false, false);
  delay(100);

  digitalWrite(MOTOR_PIN, HIGH);
  delay(500);
  setMotorMode(MOTOR_OFF);
  buzzMode = BUZZ_OFF;
}

void applyCommand(const String& line) {
  if (line.length() == 0) return;
  if (line == "ST") {
    if (millis() >= BOOT_SEQ_GUARD_MS) startupSequence();
    Serial.println("# ACK ST");
    return;
  }
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
  else if (line == "V0" || line == "V1" || line == "V2") {
    setMotorMode(line == "V0" ? MOTOR_OFF : line == "V1" ? MOTOR_PULSE : MOTOR_ON);
    Serial.print("# ACK ");
    Serial.println(line);
  } else if (line.startsWith("M")) {
    int duty = line.substring(1).toInt();
    setMotorMode(duty > 0 ? MOTOR_ON : MOTOR_OFF, duty);
    Serial.print("# ACK M");
    Serial.println(duty);
  }
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

bool startTof() {
  tof.setTimeout(1000);
  if (!tof.init()) {
    tofOk = false;
    return false;
  }
  tof.setSignalRateLimit(0.1);
  tof.setMeasurementTimingBudget(100000);
  tof.startContinuous(TOF_PERIOD_MS);
  tofOk = true;
  return true;
}


void pollTof() {
  unsigned long nowMs = millis();
  static unsigned long lastIdCheck = 0;
  if (!tofOk) {
    if (nowMs - lastIdCheck < 3000) return;
    lastIdCheck = nowMs;
    uint8_t model = tof.readReg(VL53L0X::IDENTIFICATION_MODEL_ID);
    if (startTof()) {
      Serial.println("# TOF ready");
      return;
    }
    Serial.print("# TOF not found model=0x");
    Serial.println(model, HEX);
    return;
  }

  if (nowMs - lastIdCheck >= 3000) {
    lastIdCheck = nowMs;
    uint8_t model = tof.readReg(VL53L0X::IDENTIFICATION_MODEL_ID);
    if (model != 0xEE) {
      tofOk = false;
      Serial.print("# TOF not found model=0x");
      Serial.println(model, HEX);
      return;
    }
  }

  if ((tof.readReg(VL53L0X::RESULT_INTERRUPT_STATUS) & 0x07) == 0) return;
  uint16_t mm = tof.readReg16Bit(VL53L0X::RESULT_RANGE_STATUS + 10);
  uint8_t rangeStatus = (tof.readReg(VL53L0X::RESULT_RANGE_STATUS) & 0x78) >> 3;
  tof.writeReg(VL53L0X::SYSTEM_INTERRUPT_CLEAR, 0x01);
  static unsigned long lastGood = 0;
  if (mm >= 20 && mm <= 4000) {
    lastGood = nowMs;
    Serial.print("TOF:");
    Serial.println(mm / 1000.0f, 2);
    return;
  }

  static unsigned long lastRaw = 0;
  if (nowMs - lastGood < 1000 || nowMs - lastRaw < 1000) return;
  lastRaw = nowMs;
  Serial.print("# TOF raw=");
  Serial.print(mm);
  Serial.print(" timeout=0 status=");
  Serial.println(rangeStatus);
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(STATUS_LED_G, OUTPUT);
  pinMode(STATUS_LED_Y, OUTPUT);
  pinMode(STATUS_LED_R, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  TCCR1B = (TCCR1B & 0b11111000) | 0x01;
  buzzerOff();
  setStatusLeds(false, false, false);

  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}

  mpuBus.begin();
  bool mpuOk = wakeMpu();
  delay(50);

  tofBus.begin();
  tof.setBus(&tofBus);
  startTof();

  startupSequence();

  Serial.print(F("# SAMS ready chip="));
  Serial.print(mpuOk ? mpuChip() : F("none"));
  Serial.println(F(" (mpu=A2/A3 buzzer=D8 motor=D9 tof=A0/A1)"));
  if (!mpuOk) {
    Serial.println(F("# MPU6050 not found (check SDA A2, SCL A3, VCC, GND)"));
    reportI2cBus();
  }
  if (!tofOk) Serial.println("# TOF not found");
}

void loop() {
  handleSerialCommands();
  updateMotor();
  if (buzzMode == BUZZ_ON) {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  }

  static unsigned long last = 0;
  unsigned long now = micros();
  if (now - last < SAMPLE_US) return;
  last = now;

  handleSerialCommands();

  float ax, ay, az;
  if (readAccel(ax, ay, az)) {
    Serial.print(ax, 4);
    Serial.print(',');
    Serial.print(ay, 4);
    Serial.print(',');
    Serial.println(az, 4);
  } else {
    static unsigned long lastWarn = 0;
    static unsigned long lastScan = 0;
    if (millis() - lastWarn >= 1000) {
      lastWarn = millis();
      Serial.println(F("# MPU6050 not responding (check SDA A2, SCL A3, VCC, GND)"));
      if (!wakeMpu() && millis() - lastScan >= 5000) {
        lastScan = millis();
        reportI2cBus();
      }
    }
  }

  pollTof();
}
