// Arduino.h — just enough of the Arduino and AVR environment to compile and
// RUN firmware/lift_controller/lift_controller.ino on a development machine.
//
// This is not an emulator. It is a set of stand-ins with one job: let the
// sketch's own control flow execute against simulated time, simulated pins and
// a simulated Timer1, so the velocity ramp, the reversal-through-zero rule,
// the command timeout and the limit handling can be asserted rather than
// reasoned about. `arduino-cli compile` remains the check that the same source
// is valid for the real ATmega328P.
//
// Used by tests/firmware/lift_harness.cpp, driven by tests/test_lift_firmware.py.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

using std::isinf;
using std::isnan;

// ── Pins and levels ─────────────────────────────────────────────────────────

#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2

// ── Registers ───────────────────────────────────────────────────────────────
// Real bit positions, so the sketch's register arithmetic is exercised as
// written rather than against invented constants.

extern uint8_t TCCR1A, TCCR1B, TIMSK1, PORTB, PIND;
extern uint16_t TCNT1, ICR1, OCR1A;

enum : uint8_t {
  CS10 = 0, CS11 = 1, CS12 = 2,
  WGM10 = 0, WGM11 = 1, WGM12 = 3, WGM13 = 4,
  COM1A0 = 6, COM1A1 = 7,
  TOIE1 = 0,
  PB1 = 1,
  PD4 = 4, PD5 = 5,
};

#define ISR(vector_name) void vector_name(void)

void noInterrupts();
void interrupts();

// ── Time (virtual) ──────────────────────────────────────────────────────────
//
// delay() does not sleep: it advances simulated time and runs the simulated
// pulse train for that long, so the sketch's blocking DRIVER_STARTUP_MS is
// modelled exactly as the pause in pulse generation that it is.

unsigned long millis();
unsigned long micros();
void delay(unsigned long ms);

// ── Digital I/O ─────────────────────────────────────────────────────────────

void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t value);
int digitalRead(uint8_t pin);

// ── Maths helpers Arduino provides as macros ────────────────────────────────

#define constrain(amt, low, high) \
  ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt)))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// The sketch calls sqrt() on floats.
inline float sqrt_shim(float x) { return std::sqrt(x); }
#define sqrt(x) sqrt_shim(x)

// F() puts a literal in flash on AVR. Here it is the literal itself.
#define F(string_literal) (string_literal)

// ── String ──────────────────────────────────────────────────────────────────

class String {
 public:
  String() = default;
  String(const char *text) : value_(text ? text : "") {}
  String(const std::string &text) : value_(text) {}

  unsigned int length() const { return static_cast<unsigned int>(value_.size()); }
  char charAt(unsigned int index) const {
    return index < value_.size() ? value_[index] : '\0';
  }
  char operator[](unsigned int index) const { return charAt(index); }

  int indexOf(char needle) const {
    const auto found = value_.find(needle);
    return found == std::string::npos ? -1 : static_cast<int>(found);
  }

  String substring(unsigned int from) const {
    return from >= value_.size() ? String() : String(value_.substr(from));
  }
  String substring(unsigned int from, unsigned int to) const {
    if (from >= value_.size() || to <= from) return String();
    return String(value_.substr(from, to - from));
  }

  void trim() {
    const auto first = value_.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      value_.clear();
      return;
    }
    const auto last = value_.find_last_not_of(" \t\r\n");
    value_ = value_.substr(first, last - first + 1);
  }

  void toLowerCase() {
    for (char &c : value_) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
  }

  float toFloat() const {
    try {
      return std::stof(value_);
    } catch (...) {
      return 0.0f;
    }
  }

  bool operator==(const char *other) const { return value_ == std::string(other); }
  bool operator!=(const char *other) const { return !(*this == other); }

  const char *c_str() const { return value_.c_str(); }
  const std::string &str() const { return value_; }

 private:
  std::string value_;
};

// ── Serial ──────────────────────────────────────────────────────────────────

class SerialShim {
 public:
  void begin(long) {}
  void setTimeout(unsigned long) {}

  int available() const { return input_.empty() ? 0 : 1; }
  String readStringUntil(char) {
    if (input_.empty()) return String();
    const std::string line = input_.front();
    input_.pop_front();
    return String(line);
  }

  void print(const char *text) { pending_ += text; }
  void print(const String &text) { pending_ += text.str(); }
  void print(char c) { pending_ += c; }
  void print(int value) { print_number("%d", value); }
  void print(long value) { print_number("%ld", value); }
  void print(unsigned long value) { print_number("%lu", value); }
  void print(double value, int digits = 2) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
    pending_ += buffer;
  }

  void println() { flush_line(); }
  template <typename T>
  void println(const T &value) {
    print(value);
    flush_line();
  }
  void println(double value, int digits) {
    print(value, digits);
    flush_line();
  }

  // -- harness side --
  void feed(const std::string &line) { input_.push_back(line); }
  const std::vector<std::string> &lines() const { return lines_; }
  void clear_lines() { lines_.clear(); }

 private:
  void print_number(const char *format, long value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), format, value);
    pending_ += buffer;
  }
  void flush_line() {
    lines_.push_back(pending_);
    pending_.clear();
  }

  std::string pending_;
  std::vector<std::string> lines_;
  std::deque<std::string> input_;
};

extern SerialShim Serial;
