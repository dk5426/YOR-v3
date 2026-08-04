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

// Allow relay, driver and brake time to become ready.
constexpr unsigned long DRIVER_STARTUP_MS = 500;

// Python's reader parses lines in the exact form "Height: <number> mm".
constexpr unsigned long TELEMETRY_INTERVAL_MS = 50;

constexpr int32_t MAX_HEIGHT_PULSES =
    static_cast<int32_t>(MAX_HEIGHT_MM * PULSES_PER_MM + 0.5f);

// ---------------- Motion state ----------------

enum MotionMode : uint8_t {
  MOTION_IDLE,
  MOTION_CONTINUOUS,
  MOTION_DISTANCE,
  MOTION_HOMING,
};

enum StopReason : uint8_t {
  STOP_NONE,
  STOP_COMPLETE,
  STOP_USER,
  STOP_UPPER_LIMIT,
  STOP_LOWER_LIMIT,
  STOP_SOFTWARE_LIMIT,
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
unsigned long lastTelemetryMs = 0;

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

void stopPulseTrainFromISR(StopReason reason) {
  // Stop Timer1 clock.
  TCCR1B &= ~((1 << CS12) | (1 << CS11) | (1 << CS10));

  // Disconnect OC1A from D9.
  TCCR1A &= ~((1 << COM1A1) | (1 << COM1A0));

  // Force D9 LOW using direct port access.
  PORTB &= ~(1 << PB1);

  stopReason = reason;
  pulseTrainRunning = false;
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

void setPulseFrequency(float frequencyHz) {
  if (frequencyHz < 1.0f) {
    frequencyHz = 1.0f;
  }

  // f = 16 MHz / (prescaler * (TOP + 1)); prescaler 1 is used.
  uint32_t top = static_cast<uint32_t>(16000000.0f / frequencyHz) - 1UL;

  if (top < 3UL) {
    top = 3UL;
  }
  if (top > 65535UL) {
    top = 65535UL;
  }

  noInterrupts();

  ICR1 = static_cast<uint16_t>(top);
  OCR1A = static_cast<uint16_t>((top + 1UL) / 2UL);

  // Non-inverting PWM on D9, mode 14, prescaler 1.
  TCCR1A = (1 << COM1A1) | (1 << WGM11);
  TCCR1B = (1 << WGM13) | (1 << WGM12) | (1 << CS10);

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

  Serial.print("Motion: ");
  if (motionMode == MOTION_IDLE) {
    Serial.println("IDLE");
  } else {
    Serial.println(movingUp ? "UP" : "DOWN");
  }

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
  lastTelemetryMs = 0;
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

  const unsigned long now = millis();
  if (lastTelemetryMs == 0 || (now - lastTelemetryMs) >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryMs = now;
    printHeight();
  }
}

// ---------------- Serial commands ----------------

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
  bool hasDistance = false;
  float distanceMM = DEFAULT_MOVE_MM;

  if (separator > 0) {
    action = command.substring(0, separator);
    distanceMM = command.substring(separator + 1).toFloat();
    hasDistance = true;
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
        "Commands: up, down, stop, home, up 200, down 200, status, power on, power off");
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
  Serial.println("Python commands: up, down, stop, home");
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
