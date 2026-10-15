#pragma once

#include <Arduino.h>

class SoftI2C {
 public:
  SoftI2C(uint8_t sda, uint8_t scl) : _sda(sda), _scl(scl) {}

  void begin() {
    sdaHigh();
    sclHigh();
  }

  void beginTransmission(uint8_t addr) {
    _addr = addr;
    _txLen = 0;
  }

  size_t write(uint8_t value) {
    if (_txLen >= sizeof(_tx)) return 0;
    _tx[_txLen++] = value;
    return 1;
  }

  uint8_t endTransmission() {
    startCond();
    if (!writeByte((_addr << 1) | 0)) {
      stopCond();
      return 2;
    }
    for (uint8_t i = 0; i < _txLen; i++) {
      if (!writeByte(_tx[i])) {
        stopCond();
          return 3;
      }
    }
    stopCond();
    return 0;
  }

  uint8_t requestFrom(uint8_t addr, uint8_t qty) {
    if (qty > sizeof(_rx)) qty = sizeof(_rx);
    _rxLen = 0;
    _rxPos = 0;
    startCond();
    if (!writeByte((addr << 1) | 1)) {
      stopCond();
      return 0;
    }
    for (uint8_t i = 0; i < qty; i++) {
      _rx[_rxLen++] = readByte(i + 1 < qty);
    }
    stopCond();
    return _rxLen;
  }

  int read() {
    if (_rxPos >= _rxLen) return -1;
    return _rx[_rxPos++];
  }

 private:
  void dly() { delayMicroseconds(8); }
  void sdaLow() {
    digitalWrite(_sda, LOW);
    pinMode(_sda, OUTPUT);
  }
  void sdaHigh() { pinMode(_sda, INPUT_PULLUP); }
  void sclLow() {
    digitalWrite(_scl, LOW);
    pinMode(_scl, OUTPUT);
  }
  void sclHigh() {
    pinMode(_scl, INPUT_PULLUP);
    for (uint16_t i = 0; digitalRead(_scl) == LOW && i < 2500; i++) {
      dly();
    }
  }

  void startCond() {
    sdaHigh();
    sclHigh();
    dly();
    sdaLow();
    dly();
    sclLow();
  }

  void stopCond() {
    sdaLow();
    dly();
    sclHigh();
    dly();
    sdaHigh();
    dly();
  }

  bool writeByte(uint8_t value) {
    for (uint8_t bit = 0x80; bit; bit >>= 1) {
      if (value & bit) sdaHigh();
      else sdaLow();
      dly();
      sclHigh();
      dly();
      sclLow();
    }
    sdaHigh();
    dly();
    sclHigh();
    bool ack = digitalRead(_sda) == LOW;
    dly();
    sclLow();
    return ack;
  }

  uint8_t readByte(bool ack) {
    uint8_t value = 0;
    sdaHigh();
    for (uint8_t bit = 0x80; bit; bit >>= 1) {
      dly();
      sclHigh();
      if (digitalRead(_sda)) value |= bit;
      dly();
      sclLow();
    }
    if (ack) sdaLow();
    else sdaHigh();
    dly();
    sclHigh();
    dly();
    sclLow();
    sdaHigh();
    return value;
  }

  uint8_t _sda;
  uint8_t _scl;
  uint8_t _addr = 0;
  uint8_t _tx[32];
  uint8_t _txLen = 0;
  uint8_t _rx[32];
  uint8_t _rxLen = 0;
  uint8_t _rxPos = 0;
};
