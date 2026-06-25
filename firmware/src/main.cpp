#include <Arduino.h>

const int micPin  = A0;
const int modePin = 2;

int sampleCount = 0;

void sendMode() {
  if (digitalRead(modePin) == LOW) {
    Serial.println("MODE:HEART");
  } else {
    Serial.println("MODE:LUNG");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(modePin, INPUT_PULLUP);
  sendMode();
}

void loop() {
  int value = analogRead(micPin);
  Serial.println(value);

  sampleCount++;
  if (sampleCount >= 2000) {   // re-broadcast mode every ~1 second
    sampleCount = 0;
    sendMode();
  }

  delayMicroseconds(500);      // ~2 kHz sampling
}
