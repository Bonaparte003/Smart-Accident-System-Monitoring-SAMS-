/*
  SAMS buzzer debug — D8 solid beep
  ./hardware/tools/hw.sh flash-buzzer (The debuging command)
*/

const int BUZZER_PIN = 8;

void buzzerOn() {
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
}

void buzzerOff() {
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  buzzerOff();
  Serial.begin(115200);
  delay(400);
  Serial.println("- buzzer debug ready (D8)");
  for (int i = 0; i < 3; i++) {
    buzzerOn();
    delay(300);
    buzzerOff();
    delay(150);
  }
  Serial.println("- B1=on  B0=off");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line == "B1" || line == "B2" || line == "1") {
      buzzerOn();
      Serial.println("- ACK B1");
    } else if (line == "B0" || line == "0") {
      buzzerOff();
      Serial.println("- ACK B0");
    }
  }
}
