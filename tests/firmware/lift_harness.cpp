// lift_harness.cpp — run the lift firmware on the host and assert what it does.
//
// The sketch is compiled in verbatim (see the #include below), against the
// stand-ins in include/Arduino.h. Time, the limit switches and the stepper are
// simulated: every simulated millisecond, the harness works out how many
// Timer1 overflows the sketch's own ICR1 and prescaler settings call for, and
// calls the sketch's ISR that many times. Pulses therefore come from the
// firmware's own timer configuration, not from the harness's idea of it.
//
// Driven by tests/test_lift_firmware.py.

#include "Arduino.h"

// ─────────────────────────────────────────────────────────────────────────────
// Shim state
// ─────────────────────────────────────────────────────────────────────────────

uint8_t TCCR1A = 0, TCCR1B = 0, TIMSK1 = 0, PORTB = 0, PIND = 0;
uint16_t TCNT1 = 0, ICR1 = 0, OCR1A = 0;

SerialShim Serial;

static unsigned long g_millis = 0;
static unsigned long g_micros = 0;
static uint8_t g_pin_level[32] = {0};
static double g_pulse_accum_us = 0.0;
static unsigned long g_total_pulses = 0;

void noInterrupts() {}
void interrupts() {}

unsigned long millis() { return g_millis; }
unsigned long micros() { return g_micros; }

void pinMode(uint8_t, uint8_t) {}

void digitalWrite(uint8_t pin, uint8_t value) {
  if (pin < 32) g_pin_level[pin] = value;
}

int digitalRead(uint8_t pin) { return pin < 32 ? g_pin_level[pin] : LOW; }

// The sketch reads D4/D5 straight out of PIND inside the ISR, so the two views
// of the limit switches have to be kept consistent.
static void set_limit(uint8_t pin, bool active) {
  g_pin_level[pin] = active ? HIGH : LOW;
  const uint8_t bit = (pin == 4) ? (1 << PD4) : (1 << PD5);
  if (active) {
    PIND |= bit;
  } else {
    PIND &= static_cast<uint8_t>(~bit);
  }
}

// The sketch's ISR, declared here because the sketch is included below.
extern "C" void TIMER1_OVF_vect(void);

static bool timer_clocked() {
  return (TCCR1B & ((1 << CS12) | (1 << CS11) | (1 << CS10))) != 0;
}

static double timer_period_us() {
  uint16_t prescaler = 1;
  if (TCCR1B & (1 << CS11)) prescaler = 8;
  if (TCCR1B & (1 << CS12)) prescaler = 256;
  return (static_cast<double>(ICR1) + 1.0) * prescaler / 16.0;
}

// One simulated millisecond of stepper.
static void advance_one_ms() {
  if (!timer_clocked()) {
    g_pulse_accum_us = 0.0;
  } else {
    g_pulse_accum_us += 1000.0;
    // The period is re-read every pulse: the overflow ISR may commit a pending
    // frequency change, or stop the clock entirely, part-way through.
    while (timer_clocked()) {
      const double period = timer_period_us();
      if (period <= 0.0 || g_pulse_accum_us < period) break;
      g_pulse_accum_us -= period;
      g_total_pulses++;
      TIMER1_OVF_vect();
    }
    if (!timer_clocked()) g_pulse_accum_us = 0.0;
  }
  g_millis += 1;
  g_micros += 1000;
}

void delay(unsigned long ms) {
  for (unsigned long i = 0; i < ms; i++) advance_one_ms();
}

// ─────────────────────────────────────────────────────────────────────────────
// The firmware under test
// ─────────────────────────────────────────────────────────────────────────────

#include "lift_controller.ino"

// ─────────────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────────────

static int g_checks = 0;
static int g_failures = 0;

static void check(const char *name, bool condition, const std::string &detail = "") {
  g_checks++;
  if (!condition) g_failures++;
  std::printf("  %s  %s", condition ? "PASS" : "FAIL", name);
  if (!detail.empty()) std::printf("  [%s]", detail.c_str());
  std::printf("\n");
}

static std::string num(double value, int digits = 3) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
  return buffer;
}

// Run the sketch's loop() until the given simulated time. loop() ends in
// delay(2), so time advances whether or not anything happens.
static void run_for(unsigned long ms) {
  const unsigned long until = g_millis + ms;
  while (g_millis < until) loop();
}

static void send(const char *command) { Serial.feed(command); }

// Stream a velocity for `ms`, refreshing it every 50 ms the way the host does.
static void stream(const char *command, unsigned long ms) {
  const unsigned long until = g_millis + ms;
  while (g_millis < until) {
    send(command);
    run_for(50);
  }
}

static bool saw(const char *needle) {
  for (const std::string &line : Serial.lines()) {
    if (line.find(needle) != std::string::npos) return true;
  }
  return false;
}

static int count_lines(const char *needle) {
  int total = 0;
  for (const std::string &line : Serial.lines()) {
    if (line.find(needle) != std::string::npos) total++;
  }
  return total;
}

static float height_mm() {
  return static_cast<float>(positionPulses) / PULSES_PER_MM;
}

static double step_frequency_hz() {
  if (!timer_clocked()) return 0.0;
  return 1e6 / timer_period_us();
}

static bool driver_powered() {
  return g_pin_level[DRIVER_RELAY_PIN] == (RELAY_ACTIVE_HIGH ? HIGH : LOW);
}

// Bring the board up sitting on the lower limit, which is how it establishes a
// zero without homing.
//
// On the real controller a reset re-initialises every global. Here the sketch
// is a linked translation unit whose globals outlive one "boot", so they have
// to be put back by hand — without this, state leaks between checks and the
// results are quietly meaningless.
static void boot() {
  g_millis = 0;
  g_micros = 0;
  g_pulse_accum_us = 0.0;
  g_total_pulses = 0;
  TCCR1A = TCCR1B = TIMSK1 = PORTB = PIND = 0;
  ICR1 = OCR1A = TCNT1 = 0;
  for (auto &level : g_pin_level) level = LOW;

  generatedPulses = 0;
  targetPulses = 0;
  positionPulses = 0;
  positionKnown = false;
  pulseTrainRunning = false;
  movingUp = false;
  finiteMove = false;
  homingMove = false;
  stopReason = STOP_NONE;
  timerUpdatePending = false;
  motionMode = MOTION_IDLE;
  lastTelemetryUs = 0;

  velCommandedMMs = 0.0f;
  velRampFromMMs = 0.0f;
  velRampToMMs = 0.0f;
  velRampDurationS = 0.0f;
  velRampStartUs = 0;
  velPendingTargetMMs = 0.0f;
  velReversePending = false;
  lastVelCommandMs = 0;
  lastVelUpdateUs = 0;
  velZeroSinceMs = 0;
  velTimedOut = false;
  velBlockedUp = false;
  velBlockedDown = false;

  profileRampTimeS = 0.0f;
  profileCruiseTimeS = 0.0f;
  profileStartUs = 0;

  set_limit(LOWER_LIMIT_PIN, true);
  Serial.clear_lines();
  setup();
  set_limit(LOWER_LIMIT_PIN, false);
}

// Park at a known height so a downward move has somewhere to go.
static void boot_at(float millimetres) {
  boot();
  noInterrupts();
  positionPulses = static_cast<int32_t>(millimetres * PULSES_PER_MM);
  positionKnown = true;
  interrupts();
}

// ── Checks ──────────────────────────────────────────────────────────────────

static void test_capability_banner() {
  std::printf("\ncapability advertisement\n");
  boot();
  check("startup advertises lift_velocity_v1", saw("Capabilities: lift_velocity_v1"));
  check("the ready banner is unchanged", saw("Lift controller ready."));
  check("height is established from the lower limit", saw("Height: 0.000 mm"));

  Serial.clear_lines();
  send("status");
  run_for(20);
  check("status advertises it too", saw("Capabilities: lift_velocity_v1"));
  check("status still reports the limits", saw("Upper limit:") && saw("Lower limit:"));
  check("status still reports motion", saw("Motion: IDLE"));
  check("status reports the commanded velocity", saw("Velocity: 0.00 mm/s"));
  check("velocity profile schedule is 324 Hz",
        std::fabs(1000000.0 / VEL_UPDATE_INTERVAL_US - 324.0) < 0.1,
        num(1000000.0 / VEL_UPDATE_INTERVAL_US, 3) + " Hz");
  check("height telemetry schedule is 36 Hz",
        std::fabs(1000000.0 / TELEMETRY_INTERVAL_US - 36.0) < 0.1,
        num(1000000.0 / TELEMETRY_INTERVAL_US, 3) + " Hz");
}

static void test_velocity_moves_up() {
  std::printf("\nvel: streamed upward motion\n");
  boot_at(100.0f);
  const float start = height_mm();

  stream("vel 10", 2000);

  check("entered velocity mode", saw("Velocity mode."));
  check("the driver relay closed", driver_powered());
  check("DIR is set for up", g_pin_level[DIR_PIN] == DIR_UP_LEVEL);
  check("the lift rose", height_mm() > start + 10.0f,
        num(start) + " -> " + num(height_mm()) + " mm");
  check("commanded velocity reached the request",
        std::fabs(velCommandedMMs - 10.0f) < 0.01f, num(velCommandedMMs));

  const double travelled = height_mm() - start;
  // 2 s of streaming, of which ~0.5 s is the driver relay delay and a little
  // more is the ramp, so a shade under 10 mm/s * 1.5 s.
  check("distance matches the commanded speed", travelled > 12.0 && travelled < 16.0,
        num(travelled) + " mm");
  check("height telemetry streamed", count_lines("Height:") > 20,
        std::to_string(count_lines("Height:")) + " lines");
}

static void test_velocity_moves_down() {
  std::printf("\nvel: streamed downward motion\n");
  boot_at(400.0f);
  const float start = height_mm();
  stream("vel -10", 1500);

  check("DIR is set for down", g_pin_level[DIR_PIN] == DIR_DOWN_LEVEL);
  check("the lift descended", height_mm() < start - 5.0f,
        num(start) + " -> " + num(height_mm()) + " mm");
  check("commanded velocity is negative", velCommandedMMs < -9.0f, num(velCommandedMMs));
}

static void test_clamp_and_minimum() {
  std::printf("\nvel: clamp and minimum active velocity\n");
  boot_at(100.0f);
  float peak = 0.0f;
  double peak_hz = 0.0;
  const unsigned long until = g_millis + 3000;
  while (g_millis < until) {
    send("vel 999");
    for (int i = 0; i < 25; i++) {
      loop();
      peak = max(peak, std::fabs(velCommandedMMs));
      peak_hz = max(peak_hz, step_frequency_hz());
    }
  }
  check("clamped to 50 mm/s", peak <= VEL_MAX_MM_S + 1e-3f, num(peak) + " mm/s");
  check("and actually reached it", peak > VEL_MAX_MM_S - 0.1f, num(peak) + " mm/s");
  check("step frequency matches 50 mm/s * 320 pulses/mm",
        peak_hz > 15500.0 && peak_hz < 16500.0, num(peak_hz, 0) + " Hz");

  boot_at(100.0f);
  const int32_t before = positionPulses;
  stream("vel 0.2", 800);
  check("a sub-0.5 mm/s request is a hold, not a creep", positionPulses == before,
        std::to_string(positionPulses - before) + " pulses");
  check("and it does not even energise the driver from idle", !driver_powered());
}

static void test_acceleration_and_jerk() {
  std::printf("\nvel: acceleration and jerk limits\n");
  boot_at(100.0f);

  // Sample the commanded velocity through a 0 -> 50 mm/s transition and
  // differentiate it twice. Sampling follows the ramp's own 324 Hz update rate:
  // the commanded velocity is a staircase between updates, and differencing
  // the flat parts would measure the sampling, not the motion.
  send("vel 50");
  run_for(600);                      // clear the driver relay delay
  std::vector<float> samples;
  std::vector<unsigned long> times;
  const unsigned long until = g_millis + 1200;
  while (g_millis < until) {
    send("vel 50");
    for (int i = 0; i < 10; i++) {
      const unsigned long schedule_before = lastVelUpdateUs;
      loop();
      if (lastVelUpdateUs != schedule_before) {
        samples.push_back(velCommandedMMs);
        // loop() applies the ramp before its final delay(2), so the command
        // became active two milliseconds before the harness regained control.
        times.push_back(g_micros - 2000UL);
      }
    }
  }

  double peak_accel = 0.0, peak_jerk = 0.0;
  for (size_t i = 2; i < samples.size(); i++) {
    const double dt1 = (times[i] - times[i - 1]) / 1000000.0;
    const double dt2 = (times[i - 1] - times[i - 2]) / 1000000.0;
    if (dt1 <= 0.0 || dt2 <= 0.0) continue;
    const double a1 = (samples[i] - samples[i - 1]) / dt1;
    const double a0 = (samples[i - 1] - samples[i - 2]) / dt2;
    peak_accel = max(peak_accel, std::fabs(a1));
    peak_jerk = max(peak_jerk, std::fabs(a1 - a0) / dt1);
  }
  // 10% of headroom: the ramp is sampled on a quantised 2 ms loop grid, so the
  // numerical peaks land slightly above the analytic ones.
  check("acceleration stays within 200 mm/s^2",
        peak_accel <= VEL_MAX_ACCEL_MM_S2 * 1.1, num(peak_accel, 1) + " mm/s^2");
  check("jerk stays within 2000 mm/s^3",
        peak_jerk <= VEL_MAX_JERK_MM_S3 * 1.1, num(peak_jerk, 1) + " mm/s^3");
  check("and the ramp completed", std::fabs(velCommandedMMs - 50.0f) < 0.1f,
        num(velCommandedMMs));
}

static void test_reversal_through_zero() {
  std::printf("\nvel: reversal passes through zero before DIR changes\n");
  boot_at(400.0f);
  stream("vel 20", 1500);
  check("moving up first", velCommandedMMs > 15.0f, num(velCommandedMMs));

  // Watch DIR and the pulse train together: DIR must never change while pulses
  // are being generated, and it must never change while velocity is non-zero.
  bool dir_changed_while_running = false;
  bool dir_changed_while_moving = false;
  bool reached_zero = false;
  uint8_t last_dir = g_pin_level[DIR_PIN];

  const unsigned long until = g_millis + 2000;
  while (g_millis < until) {
    send("vel -20");
    for (int i = 0; i < 25; i++) {
      loop();
      const uint8_t dir = g_pin_level[DIR_PIN];
      if (dir != last_dir) {
        if (pulseTrainRunning) dir_changed_while_running = true;
        if (std::fabs(velCommandedMMs) >= VEL_MIN_ACTIVE_MM_S) dir_changed_while_moving = true;
        last_dir = dir;
      }
      if (std::fabs(velCommandedMMs) < VEL_MIN_ACTIVE_MM_S) reached_zero = true;
    }
  }

  check("the ramp passed through zero", reached_zero);
  check("DIR never changed with the pulse train running", !dir_changed_while_running);
  check("DIR never changed at a non-zero velocity", !dir_changed_while_moving);
  check("DIR ended up pointing down", g_pin_level[DIR_PIN] == DIR_DOWN_LEVEL);
  check("and the lift is descending", velCommandedMMs < -15.0f, num(velCommandedMMs));
}

static void test_command_timeout() {
  std::printf("\nvel: 300 ms command timeout\n");
  boot_at(200.0f);
  stream("vel 20", 1500);
  check("moving before the silence", velCommandedMMs > 15.0f, num(velCommandedMMs));

  Serial.clear_lines();
  const unsigned long silent_from = g_millis;
  unsigned long stopped_at = 0;
  while (g_millis < silent_from + 2000) {   // no further commands
    loop();
    if (stopped_at == 0 && velCommandedMMs == 0.0f) stopped_at = g_millis;
  }

  check("it noticed the silence", saw("Velocity command timeout"));
  check("it stopped", std::fabs(velCommandedMMs) < 1e-6f, num(velCommandedMMs));
  check("the pulse train is stopped", !pulseTrainRunning);
  check("velocity mode exited", motionMode == MOTION_IDLE);
  check("the driver relay opened", !driver_powered());
  check("it said why", saw("Velocity stopped: no command for 300 ms."));

  // 300 ms to notice, then a jerk-limited ramp down from 20 mm/s.
  check("it stopped promptly after the silence began",
        stopped_at > 0 && (stopped_at - silent_from) < 900,
        std::to_string(stopped_at - silent_from) + " ms");
}

static void test_zero_hold_keeps_telemetry() {
  std::printf("\nvel: a zero hold keeps reporting height\n");
  boot_at(200.0f);
  stream("vel 20", 1200);

  Serial.clear_lines();
  const int32_t moving_at = positionPulses;
  stream("vel 0", 600);              // ramp down to the hold
  const int32_t held_at = positionPulses;
  check("the ramp to zero completed", velCommandedMMs == 0.0f, num(velCommandedMMs));
  check("it decelerated rather than stopping dead",
        (held_at - moving_at) > 0 && (held_at - moving_at) < 320 * 5,
        num((held_at - moving_at) / PULSES_PER_MM) + " mm of deceleration");

  Serial.clear_lines();
  stream("vel 0", 1500);             // hold, refreshed, inside the idle timeout
  check("height kept streaming while holding", count_lines("Height:") > 20,
        std::to_string(count_lines("Height:")) + " lines");
  check("the column did not move while holding", positionPulses == held_at,
        std::to_string(positionPulses - held_at) + " pulses");
  check("velocity mode is still active", motionMode == MOTION_VELOCITY);
  check("the driver stayed powered for the hold", driver_powered());

  Serial.clear_lines();
  stream("vel 0", 5500);             // past the idle exit
  check("a long hold ends the mode", motionMode == MOTION_IDLE);
  check("and opens the relay", !driver_powered());
  check("saying so", saw("Velocity idle; exiting velocity mode."));
}

static void test_limits() {
  std::printf("\nvel: limit switches and software travel\n");
  boot_at(200.0f);
  stream("vel 20", 1200);
  check("moving up", pulseTrainRunning);

  Serial.clear_lines();
  set_limit(UPPER_LIMIT_PIN, true);
  stream("vel 20", 900);             // keeps asking to go up, against the switch
  check("the upper limit stopped it", !pulseTrainRunning);
  check("it was reported", saw("LIMIT HIT: upper limit."));
  check("velocity mode exited", motionMode == MOTION_IDLE);
  check("the driver relay opened", !driver_powered());
  check("height was pinned to the top", std::fabs(height_mm() - MAX_HEIGHT_MM) < 0.01f,
        num(height_mm()) + " mm");
  check("further upward requests are refused",
        saw("UP blocked: upper limit is active."));

  // The refusal is announced once per episode — once as the switch closes
  // under a running command, and once more after the limit ends the mode and
  // the next request re-tries. What must not happen is one line per command.
  Serial.clear_lines();
  stream("vel 20", 1000);            // ~20 more requests into the same switch
  check("and is not repeated for every command afterwards",
        count_lines("UP blocked") == 0,
        std::to_string(count_lines("UP blocked")) + " further reports");
  check("the column stayed put", !pulseTrainRunning);

  set_limit(UPPER_LIMIT_PIN, false);
  Serial.clear_lines();
  stream("vel -20", 1500);
  check("driving away from a cleared limit works again", pulseTrainRunning);
  check("downward", g_pin_level[DIR_PIN] == DIR_DOWN_LEVEL);

  // Software travel limit, with both switches clear.
  boot_at(0.0f);
  noInterrupts();
  positionPulses = 4;                // four pulses above the software floor
  interrupts();
  Serial.clear_lines();
  stream("vel -20", 1500);
  check("the software floor stopped it", saw("LIMIT HIT: software travel limit."));
  check("at zero", positionPulses == 0, std::to_string(positionPulses) + " pulses");
}

static void test_invalid_commands() {
  std::printf("\nvel: invalid input\n");
  boot_at(200.0f);

  for (const char *bad : {"vel", "vel abc", "vel 1.2.3", "vel --5", "vel 5x"}) {
    Serial.clear_lines();
    const int32_t before = positionPulses;
    send(bad);
    run_for(100);
    const std::string label = std::string("rejects \"") + bad + "\"";
    check(label.c_str(), saw("Invalid velocity.") && positionPulses == before);
  }

  Serial.clear_lines();
  send("vel +12.5");
  run_for(700);
  check("an explicit + sign is accepted",
        !saw("Invalid velocity.") && velRampToMMs > 12.0f, num(velRampToMMs));
}

static void test_discrete_moves_are_unchanged() {
  std::printf("\nregression: the discrete moves still behave\n");
  boot_at(200.0f);
  const float start = height_mm();

  send("up 10");
  run_for(3000);
  check("a finite move completes", saw("Move complete."));
  check("it travelled the requested distance",
        std::fabs((height_mm() - start) - 10.0f) < 0.2f,
        num(height_mm() - start) + " mm");
  check("and stopped", motionMode == MOTION_IDLE && !pulseTrainRunning);

  // The direction-specific smooth profile has to survive: an upward move must
  // start at UP_START_SPEED_MM_S, not at the velocity mode's fixed rate.
  boot_at(200.0f);
  send("up 100");
  run_for(600);
  const double frequency = step_frequency_hz();
  check("an upward move starts on its own profile",
        frequency > UP_START_SPEED_MM_S * PULSES_PER_MM * 0.9 &&
            frequency <= UP_MAX_SPEED_MM_S * PULSES_PER_MM * 1.05,
        num(frequency / PULSES_PER_MM, 1) + " mm/s");

  send("stop");
  run_for(100);
  check("stop still stops", motionMode == MOTION_IDLE && !pulseTrainRunning);
  check("and says so", saw("Motion stopped by user."));

  boot_at(500.0f);
  send("down");
  run_for(400);
  check("a continuous down move runs", pulseTrainRunning);
  check("on the downward profile",
        step_frequency_hz() > DOWN_START_SPEED_MM_S * PULSES_PER_MM * 0.9 &&
            step_frequency_hz() <= DOWN_MAX_SPEED_MM_S * PULSES_PER_MM * 1.05,
        num(step_frequency_hz() / PULSES_PER_MM, 1) + " mm/s");
  send("stop");
  run_for(100);
  check("and stops", !pulseTrainRunning);
}

static void test_homing_is_unchanged() {
  std::printf("\nregression: homing\n");
  boot_at(880.0f);
  Serial.clear_lines();
  send("home");
  run_for(200);
  check("homing runs upward at its own fixed speed",
        std::fabs(step_frequency_hz() - HOMING_SPEED_MM_S * PULSES_PER_MM) < 50.0,
        num(step_frequency_hz() / PULSES_PER_MM, 1) + " mm/s");

  set_limit(UPPER_LIMIT_PIN, true);
  run_for(200);
  check("it completes on the upper switch", saw("Home complete."));
  check("and establishes the top of travel",
        std::fabs(height_mm() - MAX_HEIGHT_MM) < 0.01f, num(height_mm()) + " mm");
  set_limit(UPPER_LIMIT_PIN, false);
}

static void test_velocity_supersedes_a_discrete_move() {
  std::printf("\nvel: a velocity command interrupts a discrete move\n");
  boot_at(200.0f);
  send("up 200");
  run_for(700);
  check("the discrete move is running", motionMode == MOTION_DISTANCE);

  stream("vel -10", 1500);
  check("velocity mode took over", motionMode == MOTION_VELOCITY);
  check("and reversed the direction", g_pin_level[DIR_PIN] == DIR_DOWN_LEVEL);

  // ...and the other way round.
  send("up 20");
  run_for(2000);
  check("a discrete move takes back over", saw("Move complete."));
  check("leaving velocity mode behind", motionMode == MOTION_IDLE);
}

static void test_low_speed_prescaler() {
  std::printf("\nvel: low-speed timer prescaler\n");
  boot_at(200.0f);
  stream("vel 0.5", 2000);
  const double frequency = step_frequency_hz();
  check("0.5 mm/s is generated", frequency > 0.0, num(frequency, 1) + " Hz");
  check("at 0.5 mm/s * 320 pulses/mm", std::fabs(frequency - 160.0) < 5.0,
        num(frequency, 1) + " Hz");
  check("which needs prescaler 8", (TCCR1B & (1 << CS11)) != 0,
        "TCCR1B=" + std::to_string(TCCR1B));

  stream("vel 20", 2000);
  check("and prescaler 1 comes back at speed", (TCCR1B & (1 << CS10)) != 0,
        "TCCR1B=" + std::to_string(TCCR1B));
  check("at 20 mm/s * 320 pulses/mm", std::fabs(step_frequency_hz() - 6400.0) < 100.0,
        num(step_frequency_hz(), 1) + " Hz");
}

int main() {
  test_capability_banner();
  test_velocity_moves_up();
  test_velocity_moves_down();
  test_clamp_and_minimum();
  test_acceleration_and_jerk();
  test_reversal_through_zero();
  test_command_timeout();
  test_zero_hold_keeps_telemetry();
  test_limits();
  test_invalid_commands();
  test_discrete_moves_are_unchanged();
  test_homing_is_unchanged();
  test_velocity_supersedes_a_discrete_move();
  test_low_speed_prescaler();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
