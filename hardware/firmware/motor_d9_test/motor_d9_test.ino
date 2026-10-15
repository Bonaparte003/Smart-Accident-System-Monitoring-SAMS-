/*
  SAMS motor test — motor(s) on D9 (PWM) and GND
*/

const int MOTOR_PIN = 9;

void motorSet(int duty) {
  duty = constrain(duty, 0, 255);
  analogWrite(MOTOR_PIN, duty);
  digitalWrite(LED_BUILTIN, duty > 0 ? HIGH : LOW);
  Serial.print("# MOTOR ");
  Serial.println(duty);
}

void pulses() {
  for (int i = 0; i < 3; i++) {
    motorSet(255);
    delay(400);
    motorSet(0);
    delay(300);
  }
}

void ramp() {
  for (int d = 0; d <= 255; d += 15) {
    motorSet(d);
    delay(150);
  }
  for (int d = 255; d >= 0; d -= 15) {
    motorSet(d);
    delay(150);
  }
  motorSet(0);
}

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  Serial.begin(115200);
  delay(400);
  Serial.println("# motor test ready (D9)");
  pulses();
  Serial.println("# 1=on  0=off  M<0-255>=speed  S=ramp  P=pulses");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  line.toUpperCase();
  if (line == "1" || line == "M1") {
    motorSet(255);
  } else if (line == "0" || line == "M0") {
    motorSet(0);
  } else if (line.startsWith("M")) {
    motorSet(line.substring(1).toInt());
  } else if (line == "S") {
    ramp();
  } else if (line == "P") {
    pulses();
  } else if (line.length()) {
    Serial.print("# ? ");
    Serial.println(line);
  }
}
