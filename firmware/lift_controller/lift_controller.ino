#include <Arduino.h>
#include <avr/interrupt.h>

// ---------------- Pin assignments ----------------

constexpr uint8_t PUL_PIN = 9;          // OC1A: Timer1 hardware STEP output
constexpr uint8_t DIR_PIN = 3;

constexpr uint8_t UPPER_LIMIT_PIN = 4;  // NC switch to GND
constexpr uint8_t LOWER_LIMIT_PIN = 5;  // NC switch to GND

constexpr uint8_t DRIVER_RELAY_PIN = 6;

// Change this to false if your relay turns ON with a LOW signal.
constexpr bool RELAY_ACTIVE_HIGH = true;

// Change these if UP and DOWN are reversed.
constexpr uint8_t DIR_UP_LEVEL = LOW;
constexpr uint8_t DIR_DOWN_LEVEL = HIGH;

// ---------------- Motion settings ----------------

constexpr float PULSES_PER_MM = 320.0f;
// 1600 pulses/rev / 5 mm/rev = 320 pulses/mm

// Used only by the optional distance commands: "up 200" / "down 200".
constexpr float DEFAULT_MOVE_MM = 200.0f;

// Physical travel. This number is duplicated in five places and they MUST all
// agree — they drifted apart once already (the model said 0.9176 m while the
// lift is really 0.900 m):
//   firmware/lift_controller/lift_controller.ino  MAX_HEIGHT_MM      (here)
//   robot/base_motor.py                           LIFT_MAX_HEIGHT_M
//   robot/yor.py                                  lift_* max_height_m defaults
//   robot/teleop/wholebody_teleop.py              LIFT_RANGE
//   description/robot_wholebody.xml               "Slider 7" joint range,
//                                                 and lift_joint_pos ctrlrange
// The whole-body solver clamps its lift commands to the model, so if the model
// is the LARGER number it commands heights this firmware's software limit will
// never reach, and lift_to_height() stall-fails just short of its target.
constexpr float MAX_HEIGHT_MM = 900.0f;

// Homing must request MORE than the physical travel, so that the upper
// hardware limit is what ends the move. At exactly MAX_HEIGHT_MM a home
// starting from the bottom can run out of pulses just as it reaches the
// switch; that ends as STOP_COMPLETE, which finishMotion() reports as a homing
// fault. Keep this comfortably above MAX_HEIGHT_MM.
constexpr float HOMING_TRAVEL_MM = 1000.0f;

// Upward and downward moves have independent jerk-limited profiles because
// gravity and mechanical loading affect the lift differently by direction.
constexpr float UP_MAX_SPEED_MM_S = 60.0f;
constexpr float UP_START_SPEED_MM_S = 35.0f;
constexpr float UP_MAX_ACCEL_MM_S2 = 80.0f;
constexpr float UP_MAX_JERK_MM_S3 = 200.0f;

constexpr float DOWN_MAX_SPEED_MM_S = 55.0f;
constexpr float DOWN_START_SPEED_MM_S = 30.0f;
constexpr float DOWN_MAX_ACCEL_MM_S2 = 60.0f;
constexpr float DOWN_MAX_JERK_MM_S3 = 200.0f;

// Homing is upward at a separate fixed speed and does not use the S-curve.
constexpr float HOMING_SPEED_MM_S = 35.0f;

// ---------------- Streamed velocity mode ----------------
//
// "vel <signed mm/s>" is the whole-body path: the host runs a position PD
// against measured height and streams the resulting velocity. This firmware
// SHAPES that request, it does not close a velocity loop — there is no
// measured column velocity to close one against. It plans a quintic
// minimum-jerk transition from the velocity it is currently commanding to the
// one that was asked for, and converts the result into a step frequency.
//
// The discrete up / down / distance / home moves above are untouched by any
// of this and keep their own direction-specific profiles.
constexpr float VEL_MAX_MM_S = 50.0f;
// Below this the stepper is asked for so few pulses per second that a
// commanded "creep" is indistinguishable from a hold, so it is treated as a
// hold: pulses stop, but the mode, the driver power and the telemetry stay.
constexpr float VEL_MIN_ACTIVE_MM_S = 0.5f;
constexpr float VEL_MAX_ACCEL_MM_S2 = 200.0f;
constexpr float VEL_MAX_JERK_MM_S3 = 2000.0f;
// A transition always takes at least VEL_RAMP_MIN_S, so a stream of tiny
// changes cannot turn into a stream of step discontinuities, and never more
// than VEL_RAMP_MAX_S, so a large reversal still completes in bounded time.
constexpr float VEL_RAMP_MIN_S = 0.040f;
constexpr float VEL_RAMP_MAX_S = 2.000f;
constexpr unsigned long VEL_UPDATE_INTERVAL_US = 1000000UL / 324UL;
// The host refreshes at least every 100 ms. Losing three refreshes in a row
// means the link or the host is gone, and the column must not keep moving.
constexpr unsigned long VEL_COMMAND_TIMEOUT_MS = 300;
// A hold is temporary. After this long at zero the mode exits and the driver
// relay opens, so a forgotten stream does not leave the driver energised.
constexpr unsigned long VEL_ZERO_IDLE_MS = 5000;

// Protocol capability advertisement. The host must see this before it may use
// streamed velocity: an older sketch simply does not print it, and the host
// then falls back to up / down / stop.
//
// F() keeps it, and the velocity mode's other messages, in flash. This sketch
// runs on a 2 KB AVR and string literals otherwise live in RAM, where the new
// text alone would cost a seventh of it.
#define PROTOCOL_CAPABILITIES F("Capabilities: lift_velocity_v1")

// Allow relay, driver and brake time to become ready.
constexpr unsigned long DRIVER_STARTUP_MS = 500;

// Python's reader parses lines in the exact form "Height: <number> mm".
constexpr unsigned long TELEMETRY_INTERVAL_US = 1000000UL / 36UL;

constexpr int32_t MAX_HEIGHT_PULSES =
    static_cast<int32_t>(MAX_HEIGHT_MM * PULSES_PER_MM + 0.5f);

// ---------------- Motion state ----------------

enum MotionMode : uint8_t {
  MOTION_IDLE,
  MOTION_CONTINUOUS,
  MOTION_DISTANCE,
  MOTION_HOMING,
  MOTION_VELOCITY,
};

enum StopReason : uint8_t {
  STOP_NONE,
  STOP_COMPLETE,
  STOP_USER,
  STOP_UPPER_LIMIT,
  STOP_LOWER_LIMIT,
  STOP_SOFTWARE_LIMIT,
  STOP_VELOCITY_TIMEOUT,
};

volatile uint32_t generatedPulses = 0;
volatile uint32_t targetPulses = 0;
volatile int32_t positionPulses = 0;

volatile bool positionKnown = false;
volatile bool pulseTrainRunning = false;
volatile bool movingUp = false;
volatile bool finiteMove = false;
volatile bool homingMove = false;
volatile StopReason stopReason = STOP_NONE;

MotionMode motionMode = MOTION_IDLE;
unsigned long lastTelemetryUs = 0;

// Streamed velocity state. Only loop() touches these.
float velCommandedMMs = 0.0f;     // what the ramp is putting out right now
float velRampFromMMs = 0.0f;
float velRampToMMs = 0.0f;
float velRampDurationS = 0.0f;
unsigned long velRampStartUs = 0;
// A reversal is not a single ramp: the pulse train has to reach zero before
// DIR may change, so the far side of the reversal waits here.
float velPendingTargetMMs = 0.0f;
bool velReversePending = false;
unsigned long lastVelCommandMs = 0;
unsigned long lastVelUpdateUs = 0;
unsigned long velZeroSinceMs = 0;
bool velTimedOut = false;
// A host that keeps asking to drive into a closed limit switch would otherwise
// be answered thirty times a second. These latch the refusal until that
// direction is clear again.
bool velBlockedUp = false;
bool velBlockedDown = false;

// Planned normal-motion profile. These are only accessed from loop().
float activeStartSpeedMMs = UP_START_SPEED_MM_S;
float activeMaxSpeedMMs = UP_MAX_SPEED_MM_S;
float activeMaxAccelMMs2 = UP_MAX_ACCEL_MM_S2;
float activeMaxJerkMMs3 = UP_MAX_JERK_MM_S3;
float profilePeakSpeedMMs = UP_START_SPEED_MM_S;
float profileRampTimeS = 0.0f;
float profileCruiseTimeS = 0.0f;
unsigned long profileStartUs = 0;

// ---------------- Relay and limit control ----------------

void setDriverPower(bool enabled) {
  const uint8_t level = (enabled == RELAY_ACTIVE_HIGH) ? HIGH : LOW;
  digitalWrite(DRIVER_RELAY_PIN, level);
}

bool upperLimitActive() {
  return digitalRead(UPPER_LIMIT_PIN) == HIGH;
}

bool lowerLimitActive() {
  return digitalRead(LOWER_LIMIT_PIN) == HIGH;
}

// ---------------- Timer1 control ----------------

// Set by setPulseFrequency() while the pulse train is running; applied by the
// overflow ISR at TOP. See the comment at the end of TIMER1_OVF_vect.
volatile uint16_t pendingTimerTop = 0;
volatile uint8_t pendingTimerClockBits = (1 << CS10);
volatile bool timerUpdatePending = false;

void stopPulseTrainFromISR(StopReason reason) {
  // Stop Timer1 clock.
  TCCR1B &= ~((1 << CS12) | (1 << CS11) | (1 << CS10));

  // Disconnect OC1A from D9.
  TCCR1A &= ~((1 << COM1A1) | (1 << COM1A0));

  // Force D9 LOW using direct port access.
  PORTB &= ~(1 << PB1);

  stopReason = reason;
  pulseTrainRunning = false;
  // Nothing will reach an overflow to apply it now.
  timerUpdatePending = false;
}

ISR(TIMER1_OVF_vect) {
  if (!pulseTrainRunning) {
    return;
  }

  generatedPulses++;

  if (positionKnown && !homingMove) {
    positionPulses += movingUp ? 1 : -1;
  }

  // Directly read D4 and D5 for a fast safety check.
  const bool upperActive = (PIND & (1 << PD4)) != 0;
  const bool lowerActive = (PIND & (1 << PD5)) != 0;

  if (movingUp && upperActive) {
    positionPulses = MAX_HEIGHT_PULSES;
    positionKnown = true;
    stopPulseTrainFromISR(STOP_UPPER_LIMIT);
    return;
  }

  if (!movingUp && lowerActive) {
    positionPulses = 0;
    positionKnown = true;
    stopPulseTrainFromISR(STOP_LOWER_LIMIT);
    return;
  }

  // Once homed, software travel limits back up the physical switches.
  // Homing itself must reach the physical upper switch; an open-loop position
  // must never make a failed switch look like a successful home.
  if (positionKnown && !homingMove) {
    if (movingUp && positionPulses >= MAX_HEIGHT_PULSES) {
      positionPulses = MAX_HEIGHT_PULSES;
      stopPulseTrainFromISR(STOP_SOFTWARE_LIMIT);
      return;
    }

    if (!movingUp && positionPulses <= 0) {
      positionPulses = 0;
      stopPulseTrainFromISR(STOP_SOFTWARE_LIMIT);
      return;
    }
  }

  if (finiteMove && generatedPulses >= targetPulses) {
    stopPulseTrainFromISR(STOP_COMPLETE);
    return;
  }

  // A frequency change is committed here, at TOP, rather than the moment it is
  // computed. ICR1 is not double-buffered in mode 14, so writing it mid-cycle
  // can drop a step or stretch one period audibly — and the velocity ramp
  // rewrites it at 324 Hz.
  if (timerUpdatePending) {
    ICR1 = pendingTimerTop;
    OCR1A = static_cast<uint16_t>((static_cast<uint32_t>(pendingTimerTop) + 1UL) / 2UL);
    TCCR1B = (1 << WGM13) | (1 << WGM12) | pendingTimerClockBits;
    timerUpdatePending = false;
  }
}

void configureTimer1() {
  noInterrupts();

  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // Fast PWM mode 14: TOP=ICR1, PWM output=OC1A/D9.
  TCCR1A = (1 << WGM11);
  TCCR1B = (1 << WGM13) | (1 << WGM12);
  TIMSK1 = (1 << TOIE1);

  interrupts();
}

// f = clock / (prescaler * (TOP + 1)). Prescaler 1 covers everything down to
// 16 MHz / 65536 = 244 Hz, which is 0.76 mm/s. The velocity mode's minimum
// active speed is 0.5 mm/s (160 Hz), so a second prescaler is needed below
// that; prescaler 8 reaches 30.5 Hz, well under anything commandable.
void computeTimerSettings(float frequencyHz, uint16_t &top, uint8_t &clockBits) {
  if (frequencyHz < 1.0f) {
    frequencyHz = 1.0f;
  }

  float timerClockHz = 16000000.0f;
  clockBits = (1 << CS10);                       // prescaler 1

  if (frequencyHz < (16000000.0f / 65536.0f)) {
    timerClockHz = 2000000.0f;
    clockBits = (1 << CS11);                     // prescaler 8
  }

  uint32_t value = static_cast<uint32_t>(timerClockHz / frequencyHz) - 1UL;

  if (value < 3UL) {
    value = 3UL;
  }
  if (value > 65535UL) {
    value = 65535UL;
  }

  top = static_cast<uint16_t>(value);
}

void setPulseFrequency(float frequencyHz) {
  uint16_t top;
  uint8_t clockBits;
  computeTimerSettings(frequencyHz, top, clockBits);

  noInterrupts();

  // While the timer is clocked, hand the new period to the overflow ISR. While
  // it is stopped no overflow is coming, so it must be applied here — that is
  // also the path that starts the pulse train.
  const bool timerClocked =
      (TCCR1B & ((1 << CS12) | (1 << CS11) | (1 << CS10))) != 0;

  if (timerClocked) {
    pendingTimerTop = top;
    pendingTimerClockBits = clockBits;
    timerUpdatePending = true;
  } else {
    ICR1 = top;
    OCR1A = static_cast<uint16_t>((static_cast<uint32_t>(top) + 1UL) / 2UL);

    // Non-inverting PWM on D9, mode 14.
    TCCR1A = (1 << COM1A1) | (1 << WGM11);
    TCCR1B = (1 << WGM13) | (1 << WGM12) | clockBits;
    timerUpdatePending = false;
  }

  interrupts();
}

// ---------------- Jerk-limited S-curve motion profile ----------------

// Quintic smootherstep has zero acceleration and zero jerk at both ends of a
// velocity transition. Its maximum normalized acceleration is 1.875 and its
// maximum normalized jerk is 5.773503.
float smootherStep(float x) {
  x = constrain(x, 0.0f, 1.0f);
  return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

float rampTimeForPeakSpeed(float peakSpeedMMs) {
  const float deltaSpeed = peakSpeedMMs - activeStartSpeedMMs;
  if (deltaSpeed <= 0.001f) {
    return 0.0f;
  }

  const float accelerationLimitedTime =
      1.875f * deltaSpeed / activeMaxAccelMMs2;
  const float jerkLimitedTime =
      sqrt(5.773503f * deltaSpeed / activeMaxJerkMMs3);

  return accelerationLimitedTime > jerkLimitedTime
             ? accelerationLimitedTime
             : jerkLimitedTime;
}

// Acceleration and deceleration are symmetric. Since smootherstep averages
// 0.5 over a complete transition, their combined distance is
// (startSpeed + peakSpeed) * rampTime.
float combinedRampDistanceMM(float peakSpeedMMs) {
  return (activeStartSpeedMMs + peakSpeedMMs) *
         rampTimeForPeakSpeed(peakSpeedMMs);
}

void selectMotionParameters(bool commandUp) {
  if (commandUp) {
    activeStartSpeedMMs = UP_START_SPEED_MM_S;
    activeMaxSpeedMMs = UP_MAX_SPEED_MM_S;
    activeMaxAccelMMs2 = UP_MAX_ACCEL_MM_S2;
    activeMaxJerkMMs3 = UP_MAX_JERK_MM_S3;
  } else {
    activeStartSpeedMMs = DOWN_START_SPEED_MM_S;
    activeMaxSpeedMMs = DOWN_MAX_SPEED_MM_S;
    activeMaxAccelMMs2 = DOWN_MAX_ACCEL_MM_S2;
    activeMaxJerkMMs3 = DOWN_MAX_JERK_MM_S3;
  }
}

void planMotionProfile(bool isFinite, float distanceMM) {
  profilePeakSpeedMMs = activeMaxSpeedMMs;

  if (isFinite &&
      combinedRampDistanceMM(profilePeakSpeedMMs) > distanceMM) {
    // The requested move is too short to reach maximum speed. Find the highest
    // peak speed whose acceleration and deceleration phases fit the distance.
    float lowSpeed = activeStartSpeedMMs;
    float highSpeed = activeMaxSpeedMMs;

    for (uint8_t i = 0; i < 20; i++) {
      const float candidate = 0.5f * (lowSpeed + highSpeed);
      if (combinedRampDistanceMM(candidate) <= distanceMM) {
        lowSpeed = candidate;
      } else {
        highSpeed = candidate;
      }
    }

    profilePeakSpeedMMs = lowSpeed;
  }

  profileRampTimeS = rampTimeForPeakSpeed(profilePeakSpeedMMs);
  profileCruiseTimeS = 0.0f;

  if (isFinite) {
    const float rampDistance =
        combinedRampDistanceMM(profilePeakSpeedMMs);
    const float cruiseDistance = max(0.0f, distanceMM - rampDistance);
    profileCruiseTimeS = cruiseDistance / profilePeakSpeedMMs;
  }
}

float motionProfileSpeed(unsigned long nowUs, bool isFinite) {
  if (profileRampTimeS <= 0.0f) {
    return profilePeakSpeedMMs;
  }

  const float elapsedS =
      static_cast<float>(nowUs - profileStartUs) * 0.000001f;
  const float deltaSpeed = profilePeakSpeedMMs - activeStartSpeedMMs;

  if (elapsedS < profileRampTimeS) {
    const float phase = elapsedS / profileRampTimeS;
    return activeStartSpeedMMs + deltaSpeed * smootherStep(phase);
  }

  if (!isFinite ||
      elapsedS < (profileRampTimeS + profileCruiseTimeS)) {
    return profilePeakSpeedMMs;
  }

  const float decelerationElapsed =
      elapsedS - profileRampTimeS - profileCruiseTimeS;
  if (decelerationElapsed < profileRampTimeS) {
    const float phase = decelerationElapsed / profileRampTimeS;
    return profilePeakSpeedMMs - deltaSpeed * smootherStep(phase);
  }

  // Pulse counting remains the final authority. If timer quantization leaves a
  // few pulses after the planned profile, finish them at the entry speed.
  return activeStartSpeedMMs;
}

// ---------------- Streamed velocity mode ----------------

// Both are defined further down. The Arduino build generates prototypes by
// itself, but declaring them keeps this a valid translation unit for the
// host-side harness in tests/firmware/ as well.
void printHeight();
void stopMotion();

// Reuses smootherStep() above: quintic, so acceleration and jerk are both zero
// at each end of a velocity transition.
void planVelocityRamp(float targetMMs) {
  // A ramp already heading there must be left alone. The host refreshes an
  // unchanged command every 100 ms as a keepalive, and replanning on each of
  // those would restart the transition from wherever it had reached — the
  // velocity would approach the request asymptotically and never arrive.
  if (velRampDurationS > 0.0f && fabs(targetMMs - velRampToMMs) < 0.0001f) {
    return;
  }

  const float delta = fabs(targetMMs - velCommandedMMs);

  if (delta < 0.0001f) {
    velCommandedMMs = targetMMs;
    velRampFromMMs = targetMMs;
    velRampToMMs = targetMMs;
    velRampDurationS = 0.0f;
    return;
  }

  // Smootherstep peaks at 1.875x the mean acceleration and 5.773503x the mean
  // jerk, so these are the shortest durations that respect both limits.
  const float accelerationLimitedTime = 1.875f * delta / VEL_MAX_ACCEL_MM_S2;
  const float jerkLimitedTime = sqrt(5.773503f * delta / VEL_MAX_JERK_MM_S3);

  float duration = accelerationLimitedTime > jerkLimitedTime
                       ? accelerationLimitedTime
                       : jerkLimitedTime;
  duration = constrain(duration, VEL_RAMP_MIN_S, VEL_RAMP_MAX_S);

  velRampFromMMs = velCommandedMMs;
  velRampToMMs = targetMMs;
  velRampDurationS = duration;
  velRampStartUs = micros();
}

float velocityRampOutput(unsigned long nowUs) {
  if (velRampDurationS <= 0.0f) {
    return velRampToMMs;
  }

  const float elapsedS =
      static_cast<float>(nowUs - velRampStartUs) * 0.000001f;

  if (elapsedS >= velRampDurationS) {
    velRampDurationS = 0.0f;
    return velRampToMMs;
  }

  const float phase = elapsedS / velRampDurationS;
  return velRampFromMMs +
         (velRampToMMs - velRampFromMMs) * smootherStep(phase);
}

void stopVelocityPulses(StopReason reason) {
  noInterrupts();
  if (pulseTrainRunning) {
    stopPulseTrainFromISR(reason);
  }
  interrupts();
  digitalWrite(PUL_PIN, LOW);
}

void exitVelocityMode(bool powerOff) {
  stopVelocityPulses(STOP_USER);

  noInterrupts();
  stopReason = STOP_NONE;
  interrupts();

  motionMode = MOTION_IDLE;
  velCommandedMMs = 0.0f;
  velRampFromMMs = 0.0f;
  velRampToMMs = 0.0f;
  velRampDurationS = 0.0f;
  velPendingTargetMMs = 0.0f;
  velReversePending = false;
  velZeroSinceMs = 0;
  velTimedOut = false;
  velBlockedUp = false;
  velBlockedDown = false;

  if (powerOff) {
    setDriverPower(false);
  }

  printHeight();
}

// DIR may only change while no pulses are being generated, which is why a
// reversal ramps to zero first.
void applyVelocityDirection(bool commandUp) {
  stopVelocityPulses(STOP_NONE);
  digitalWrite(DIR_PIN, commandUp ? DIR_UP_LEVEL : DIR_DOWN_LEVEL);
  delay(10);

  noInterrupts();
  movingUp = commandUp;
  stopReason = STOP_NONE;
  interrupts();
}

// The limit switches are checked here as well as in the ISR: the ISR can only
// stop a train that is already running, and this is what stops one starting
// into a switch that is already closed.
bool limitBlocksDirection(bool commandUp) {
  if (commandUp && upperLimitActive()) {
    noInterrupts();
    positionPulses = MAX_HEIGHT_PULSES;
    positionKnown = true;
    interrupts();
    return true;
  }

  if (!commandUp && lowerLimitActive()) {
    noInterrupts();
    positionPulses = 0;
    positionKnown = true;
    interrupts();
    return true;
  }

  return false;
}

// Same question, but reported at most once per blockage. Driving away from the
// switch clears the latch, so the lift is never stuck against a limit.
bool velocityDirectionBlocked(bool commandUp) {
  bool &latched = commandUp ? velBlockedUp : velBlockedDown;

  if (!limitBlocksDirection(commandUp)) {
    latched = false;
    return false;
  }

  if (!latched) {
    latched = true;
    Serial.println(commandUp ? F("UP blocked: upper limit is active.")
                             : F("DOWN blocked: lower limit is active."));
    printHeight();
  }

  return true;
}

void startVelocityPulses(float magnitudeMMs) {
  bool running;
  bool commandUp;
  noInterrupts();
  running = pulseTrainRunning;
  commandUp = movingUp;
  interrupts();

  if (running) {
    setPulseFrequency(magnitudeMMs * PULSES_PER_MM);
    return;
  }

  if (velocityDirectionBlocked(commandUp)) {
    velCommandedMMs = 0.0f;
    velRampToMMs = 0.0f;
    velRampDurationS = 0.0f;
    velReversePending = false;
    return;
  }

  noInterrupts();
  generatedPulses = 0;
  targetPulses = 0;
  finiteMove = false;
  homingMove = false;
  stopReason = STOP_NONE;
  pulseTrainRunning = true;
  interrupts();

  setPulseFrequency(magnitudeMMs * PULSES_PER_MM);
}

// Enters velocity mode from idle. Powering the driver is the slow part, so it
// happens once here rather than on every command in the stream.
bool enterVelocityMode(bool commandUp) {
  if (velocityDirectionBlocked(commandUp)) {
    return false;
  }

  setDriverPower(true);
  delay(DRIVER_STARTUP_MS);

  if (limitBlocksDirection(commandUp)) {
    setDriverPower(false);
    Serial.println(commandUp
                       ? F("UP blocked: upper limit became active during startup.")
                       : F("DOWN blocked: lower limit became active during startup."));
    printHeight();
    return false;
  }

  digitalWrite(DIR_PIN, commandUp ? DIR_UP_LEVEL : DIR_DOWN_LEVEL);
  delay(10);

  noInterrupts();
  generatedPulses = 0;
  targetPulses = 0;
  movingUp = commandUp;
  finiteMove = false;
  homingMove = false;
  stopReason = STOP_NONE;
  pulseTrainRunning = false;
  interrupts();

  motionMode = MOTION_VELOCITY;
  velCommandedMMs = 0.0f;
  velRampFromMMs = 0.0f;
  velRampToMMs = 0.0f;
  velRampDurationS = 0.0f;
  velPendingTargetMMs = 0.0f;
  velReversePending = false;
  velZeroSinceMs = 0;
  velTimedOut = false;
  lastVelUpdateUs = micros();
  lastTelemetryUs = 0;

  Serial.println(F("Velocity mode."));
  return true;
}

void setVelocityTarget(float targetMMs) {
  const bool wantUp = targetMMs > 0.0f;
  const bool moving = fabs(velCommandedMMs) >= VEL_MIN_ACTIVE_MM_S;

  bool currentlyUp;
  noInterrupts();
  currentlyUp = movingUp;
  interrupts();

  if (fabs(targetMMs) < VEL_MIN_ACTIVE_MM_S) {
    velReversePending = false;
    planVelocityRamp(0.0f);
    return;
  }

  if (moving && wantUp != currentlyUp) {
    // Ramp through zero, then pick the request up again on the far side.
    velPendingTargetMMs = targetMMs;
    velReversePending = true;
    planVelocityRamp(0.0f);
    return;
  }

  if (velocityDirectionBlocked(wantUp)) {
    // Hold at zero rather than planning a ramp startVelocityPulses() would
    // refuse a few milliseconds later. The switch is re-read on every request,
    // so the moment it clears the next command moves the lift.
    velReversePending = false;
    planVelocityRamp(0.0f);
    return;
  }

  if (!moving && wantUp != currentlyUp) {
    applyVelocityDirection(wantUp);
  }

  velReversePending = false;
  planVelocityRamp(targetMMs);
}

// The "vel <signed mm/s>" entry point. Clamping happens here so nothing
// downstream has to trust the host.
void requestVelocity(float requestedMMs) {
  if (isnan(requestedMMs) || isinf(requestedMMs)) {
    Serial.println(F("Invalid velocity."));
    return;
  }

  requestedMMs = constrain(requestedMMs, -VEL_MAX_MM_S, VEL_MAX_MM_S);
  if (fabs(requestedMMs) < VEL_MIN_ACTIVE_MM_S) {
    requestedMMs = 0.0f;
  }

  // A velocity command supersedes a discrete move, exactly as a discrete move
  // supersedes another one.
  if (motionMode != MOTION_IDLE && motionMode != MOTION_VELOCITY) {
    stopMotion();
  }

  if (motionMode != MOTION_VELOCITY) {
    // A zero request from idle is a no-op rather than a reason to energise the
    // driver: the host streams zero whenever it is inside its deadband, and
    // cycling the relay for that would wear it out for nothing.
    if (requestedMMs == 0.0f) {
      lastVelCommandMs = millis();
      return;
    }
    if (!enterVelocityMode(requestedMMs > 0.0f)) {
      return;
    }
  }

  lastVelCommandMs = millis();
  velTimedOut = false;
  setVelocityTarget(requestedMMs);
}

void updateVelocityMode() {
  const unsigned long nowMs = millis();
  const unsigned long nowUs = micros();

  StopReason reason;
  bool running;
  noInterrupts();
  reason = stopReason;
  running = pulseTrainRunning;
  interrupts();

  // A limit reached by the ISR ends velocity mode outright. The host will see
  // the line, and its next "vel" command starts a fresh, re-checked mode.
  if (!running && (reason == STOP_UPPER_LIMIT || reason == STOP_LOWER_LIMIT ||
                   reason == STOP_SOFTWARE_LIMIT)) {
    switch (reason) {
      case STOP_UPPER_LIMIT:
        Serial.println(F("LIMIT HIT: upper limit."));
        break;
      case STOP_LOWER_LIMIT:
        Serial.println(F("LIMIT HIT: lower limit."));
        break;
      default:
        Serial.println(F("LIMIT HIT: software travel limit."));
        break;
    }
    exitVelocityMode(true);
    return;
  }

  if (!velTimedOut && (nowMs - lastVelCommandMs) > VEL_COMMAND_TIMEOUT_MS) {
    velTimedOut = true;
    velReversePending = false;
    planVelocityRamp(0.0f);
    Serial.println(F("Velocity command timeout; ramping to zero."));
  }

  if ((nowUs - lastVelUpdateUs) >= VEL_UPDATE_INTERVAL_US) {
    // Advance the ideal schedule instead of resetting it to `nowUs`. loop()
    // wakes on a 2 ms grid, so this intentionally alternates 2 ms and 4 ms
    // gaps while preserving a 324 Hz average rather than collapsing to 250 Hz.
    // Skip missed slots after a long serial read rather than replaying them as
    // a burst; the ramp itself is evaluated at the current wall-clock time.
    const unsigned long elapsedUs = nowUs - lastVelUpdateUs;
    lastVelUpdateUs +=
        (elapsedUs / VEL_UPDATE_INTERVAL_US) * VEL_UPDATE_INTERVAL_US;

    velCommandedMMs = velocityRampOutput(nowUs);
    const float magnitude = fabs(velCommandedMMs);

    if (magnitude < VEL_MIN_ACTIVE_MM_S) {
      velCommandedMMs = 0.0f;
      stopVelocityPulses(STOP_NONE);

      if (velReversePending) {
        velReversePending = false;
        applyVelocityDirection(velPendingTargetMMs > 0.0f);
        planVelocityRamp(velPendingTargetMMs);
        velPendingTargetMMs = 0.0f;
        velZeroSinceMs = 0;
      } else if (velZeroSinceMs == 0) {
        velZeroSinceMs = nowMs;
      }
    } else {
      velZeroSinceMs = 0;
      startVelocityPulses(magnitude);
    }
  }

  // Height keeps streaming while the mode holds at zero, so the host's PD
  // never has to servo against a measurement that stopped arriving.
  if (lastTelemetryUs == 0 || (nowUs - lastTelemetryUs) >= TELEMETRY_INTERVAL_US) {
    lastTelemetryUs = lastTelemetryUs == 0
                          ? nowUs
                          : lastTelemetryUs
                                + ((nowUs - lastTelemetryUs) / TELEMETRY_INTERVAL_US)
                                      * TELEMETRY_INTERVAL_US;
    printHeight();
  }

  if (velCommandedMMs == 0.0f && !velReversePending) {
    if (velTimedOut) {
      Serial.println(F("Velocity stopped: no command for 300 ms."));
      exitVelocityMode(true);
      return;
    }
    if (velZeroSinceMs != 0 && (nowMs - velZeroSinceMs) >= VEL_ZERO_IDLE_MS) {
      Serial.println(F("Velocity idle; exiting velocity mode."));
      exitVelocityMode(true);
      return;
    }
  }
}

// ---------------- Height telemetry ----------------

void printHeight() {
  int32_t pulses;
  bool known;

  noInterrupts();
  pulses = positionPulses;
  known = positionKnown;
  interrupts();

  if (!known) {
    Serial.println("Height: unknown (run home)");
    return;
  }

  const float heightMM = static_cast<float>(pulses) / PULSES_PER_MM;
  Serial.print("Height: ");
  Serial.print(heightMM, 3);
  Serial.println(" mm");
}

void printStatus() {
  Serial.print("Upper limit: ");
  Serial.println(upperLimitActive() ? "ACTIVE" : "clear");

  Serial.print("Lower limit: ");
  Serial.println(lowerLimitActive() ? "ACTIVE" : "clear");

  // The host's parser accepts exactly IDLE, UP or DOWN, so a velocity-mode
  // hold reports IDLE — which is also what it is: no pulses are being made.
  Serial.print("Motion: ");
  if (motionMode == MOTION_IDLE || !pulseTrainRunning) {
    Serial.println("IDLE");
  } else {
    Serial.println(movingUp ? "UP" : "DOWN");
  }

  Serial.print(F("Velocity: "));
  Serial.print(velCommandedMMs, 2);
  Serial.println(F(" mm/s"));

  Serial.println(PROTOCOL_CAPABILITIES);

  printHeight();
}

// ---------------- Non-blocking motion control ----------------

void finishMotion() {
  if (motionMode == MOTION_IDLE || pulseTrainRunning) {
    return;
  }

  StopReason reason;
  noInterrupts();
  reason = stopReason;
  interrupts();

  const MotionMode finishedMode = motionMode;
  motionMode = MOTION_IDLE;
  finiteMove = false;
  homingMove = false;
  digitalWrite(PUL_PIN, LOW);

  if (finishedMode == MOTION_HOMING) {
    if (reason == STOP_UPPER_LIMIT) {
      noInterrupts();
      positionPulses = MAX_HEIGHT_PULSES;
      positionKnown = true;
      interrupts();
      Serial.println("Home complete.");
    } else if (reason == STOP_USER) {
      noInterrupts();
      positionKnown = false;
      interrupts();
      Serial.println("Home stopped.");
    } else {
      noInterrupts();
      positionKnown = false;
      interrupts();
      Serial.println("Home failed: upper limit was not reached.");
    }
  } else {
    switch (reason) {
      case STOP_COMPLETE:
        Serial.println("Move complete.");
        break;
      case STOP_USER:
        Serial.println("Motion stopped by user.");
        break;
      case STOP_UPPER_LIMIT:
        Serial.println("LIMIT HIT: upper limit.");
        break;
      case STOP_LOWER_LIMIT:
        Serial.println("LIMIT HIT: lower limit.");
        break;
      case STOP_SOFTWARE_LIMIT:
        Serial.println("LIMIT HIT: software travel limit.");
        break;
      default:
        Serial.println("Motion stopped.");
        break;
    }
  }

  // Match the original safety behavior for an abort or limit event.
  if (finishedMode == MOTION_HOMING || reason != STOP_COMPLETE) {
    setDriverPower(false);
  }

  printHeight();
}

void stopMotion() {
  if (motionMode == MOTION_VELOCITY) {
    Serial.println(F("Motion stopped by user."));
    exitVelocityMode(true);
    return;
  }

  if (motionMode == MOTION_IDLE) {
    setDriverPower(false);
    return;
  }

  noInterrupts();
  if (pulseTrainRunning) {
    stopPulseTrainFromISR(STOP_USER);
  }
  interrupts();

  finishMotion();
}

bool startMotion(bool commandUp, MotionMode requestedMode, float distanceMM = 0.0f) {
  if (motionMode != MOTION_IDLE) {
    // Repeated Python teleop commands are idempotent.
    if (requestedMode == MOTION_CONTINUOUS &&
        motionMode == MOTION_CONTINUOUS &&
        movingUp == commandUp) {
      return true;
    }
    stopMotion();
  }

  if (commandUp && upperLimitActive()) {
    noInterrupts();
    positionPulses = MAX_HEIGHT_PULSES;
    positionKnown = true;
    interrupts();
    if (requestedMode == MOTION_HOMING) {
      Serial.println("Home complete.");
    } else {
      Serial.println("UP blocked: upper limit is active.");
    }
    printHeight();
    return requestedMode == MOTION_HOMING;
  }

  if (!commandUp && lowerLimitActive()) {
    noInterrupts();
    positionPulses = 0;
    positionKnown = true;
    interrupts();

    if (requestedMode == MOTION_HOMING) {
      Serial.println("Home complete.");
    } else {
      Serial.println("DOWN blocked: lower limit is active.");
    }
    printHeight();
    return requestedMode == MOTION_HOMING;
  }

  if (requestedMode == MOTION_DISTANCE || requestedMode == MOTION_HOMING) {
    if (distanceMM <= 0.0f) {
      Serial.println("Invalid distance.");
      return false;
    }
  }

  setDriverPower(true);
  delay(DRIVER_STARTUP_MS);

  // Check again after driver power-up.
  if (commandUp && upperLimitActive()) {
    setDriverPower(false);
    noInterrupts();
    positionPulses = MAX_HEIGHT_PULSES;
    positionKnown = true;
    interrupts();
    if (requestedMode == MOTION_HOMING) {
      Serial.println("Home complete.");
    } else {
      Serial.println("UP blocked: upper limit became active during startup.");
    }
    printHeight();
    return requestedMode == MOTION_HOMING;
  }

  if (!commandUp && lowerLimitActive()) {
    setDriverPower(false);
    noInterrupts();
    positionPulses = 0;
    positionKnown = true;
    interrupts();
    if (requestedMode == MOTION_HOMING) {
      Serial.println("Home complete.");
    } else {
      Serial.println("DOWN blocked: lower limit became active during startup.");
    }
    printHeight();
    return requestedMode == MOTION_HOMING;
  }

  digitalWrite(DIR_PIN, commandUp ? DIR_UP_LEVEL : DIR_DOWN_LEVEL);
  delay(10);

  uint32_t requestedPulses = 0;
  const bool isFinite =
      requestedMode == MOTION_DISTANCE || requestedMode == MOTION_HOMING;
  if (isFinite) {
    requestedPulses =
        static_cast<uint32_t>(distanceMM * PULSES_PER_MM + 0.5f);
  }

  if (requestedMode != MOTION_HOMING) {
    selectMotionParameters(commandUp);
    planMotionProfile(isFinite, distanceMM);
  }

  noInterrupts();
  generatedPulses = 0;
  targetPulses = requestedPulses;
  movingUp = commandUp;
  finiteMove = isFinite;
  homingMove = requestedMode == MOTION_HOMING;
  stopReason = STOP_NONE;
  pulseTrainRunning = true;
  interrupts();

  motionMode = requestedMode;
  lastTelemetryUs = 0;
  profileStartUs = micros();

  // All ISR-visible state is valid before Timer1 starts producing pulses.
  // Homing uses a fixed slow speed; normal motion uses the ramp profile.
  const float initialSpeedMMs =
      requestedMode == MOTION_HOMING
          ? HOMING_SPEED_MM_S
          : activeStartSpeedMMs;
  setPulseFrequency(initialSpeedMMs * PULSES_PER_MM);

  if (requestedMode == MOTION_HOMING) {
    Serial.println("Homing UP...");
  } else if (requestedMode == MOTION_CONTINUOUS) {
    Serial.print("Moving ");
    Serial.println(commandUp ? "UP" : "DOWN");
  } else {
    Serial.print("Moving ");
    Serial.print(commandUp ? "UP " : "DOWN ");
    Serial.print(distanceMM);
    Serial.println(" mm");
  }

  return true;
}

void updateMotion() {
  if (motionMode == MOTION_IDLE) {
    return;
  }

  // Velocity mode runs its own supervision: a stopped pulse train there is a
  // zero hold, not a finished move.
  if (motionMode == MOTION_VELOCITY) {
    updateVelocityMode();
    return;
  }

  if (!pulseTrainRunning) {
    finishMotion();
    return;
  }

  float speedMMs;
  if (motionMode == MOTION_HOMING) {
    speedMMs = HOMING_SPEED_MM_S;
  } else {
    speedMMs = motionProfileSpeed(micros(), finiteMove);
  }

  setPulseFrequency(speedMMs * PULSES_PER_MM);

  const unsigned long nowUs = micros();
  if (lastTelemetryUs == 0 || (nowUs - lastTelemetryUs) >= TELEMETRY_INTERVAL_US) {
    lastTelemetryUs = lastTelemetryUs == 0
                          ? nowUs
                          : lastTelemetryUs
                                + ((nowUs - lastTelemetryUs) / TELEMETRY_INTERVAL_US)
                                      * TELEMETRY_INTERVAL_US;
    printHeight();
  }
}

// ---------------- Serial commands ----------------

bool parseSignedFloat(const String &text, float &out) {
  if (text.length() == 0) {
    return false;
  }

  bool sawDigit = false;
  bool sawPoint = false;

  for (uint16_t i = 0; i < text.length(); i++) {
    const char c = text.charAt(i);
    if (c >= '0' && c <= '9') {
      sawDigit = true;
    } else if (c == '.') {
      if (sawPoint) {
        return false;
      }
      sawPoint = true;
    } else if ((c == '+' || c == '-') && i == 0) {
      continue;
    } else {
      return false;
    }
  }

  if (!sawDigit) {
    return false;
  }

  out = text.toFloat();
  return true;
}

void processCommand(String command) {
  command.trim();
  command.toLowerCase();

  if (command.length() == 0) {
    return;
  }

  if (command == "status") {
    printStatus();
    return;
  }

  if (command == "stop" || command == "x") {
    stopMotion();
    return;
  }

  if (command == "home") {
    startMotion(true, MOTION_HOMING, HOMING_TRAVEL_MM);
    return;
  }

  if (command == "power on") {
    setDriverPower(true);
    Serial.println("Driver power ON.");
    return;
  }

  if (command == "power off") {
    stopMotion();
    setDriverPower(false);
    Serial.println("Driver power OFF.");
    return;
  }

  const int separator = command.indexOf(' ');
  String action = command;
  String argument = "";
  bool hasDistance = false;
  float distanceMM = DEFAULT_MOVE_MM;

  if (separator > 0) {
    action = command.substring(0, separator);
    argument = command.substring(separator + 1);
    argument.trim();
    distanceMM = argument.toFloat();
    hasDistance = true;
  }

  if (action == "vel") {
    // Parsed strictly: String::toFloat() answers 0.0 for anything it does not
    // understand, and a silent zero here would read as a legitimate hold and
    // keep refreshing the command timeout.
    float requested = 0.0f;
    if (!hasDistance || !parseSignedFloat(argument, requested)) {
      Serial.println(F("Invalid velocity."));
      return;
    }
    requestVelocity(requested);
    return;
  }

  if (action == "up") {
    startMotion(
        true,
        hasDistance ? MOTION_DISTANCE : MOTION_CONTINUOUS,
        distanceMM);
  } else if (action == "down") {
    startMotion(
        false,
        hasDistance ? MOTION_DISTANCE : MOTION_CONTINUOUS,
        distanceMM);
  } else {
    Serial.println(
        F("Commands: up, down, stop, home, up 200, down 200, vel 12.5, status, "
          "power on, power off"));
  }
}

// ---------------- Arduino setup/loop ----------------

void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(DRIVER_RELAY_PIN, OUTPUT);

  pinMode(UPPER_LIMIT_PIN, INPUT_PULLUP);
  pinMode(LOWER_LIMIT_PIN, INPUT_PULLUP);

  digitalWrite(PUL_PIN, LOW);
  digitalWrite(DIR_PIN, DIR_DOWN_LEVEL);
  setDriverPower(false);

  Serial.begin(115200);
  Serial.setTimeout(20);

  configureTimer1();
  delay(500);

  // If the controller boots on a known endpoint, establish height immediately.
  if (lowerLimitActive()) {
    positionPulses = 0;
    positionKnown = true;
  } else if (upperLimitActive()) {
    positionPulses = MAX_HEIGHT_PULSES;
    positionKnown = true;
  }

  Serial.println("Lift controller ready.");
  // The host reads this to decide whether it may stream velocity. An older
  // sketch does not print it, and the host then stays on up / down / stop.
  Serial.println(PROTOCOL_CAPABILITIES);
  Serial.println(F("Python commands: up, down, stop, home, vel <signed mm/s>"));
  Serial.println("Optional commands: up 200, down 200, status, power on, power off");
  printHeight();
}

void loop() {
  // updateMotion is non-blocking, so Python's line-based "stop" command is
  // processed promptly even while the step pulse train is running.
  updateMotion();

  if (Serial.available() > 0) {
    const String command = Serial.readStringUntil('\n');
    processCommand(command);
  }

  delay(2);
}
