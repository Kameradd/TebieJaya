import time
import board
import digitalio
import serial
SERIAL_PORT = '/dev/ttyTHS1'  # Jetson J1010 UART port
BAUDRATE = 115200
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Serial connection established on {SERIAL_PORT} at {BAUDRATE} baud.")
except Exception as e:
    print(f"Failed to open serial port {SERIAL_PORT}: {e}")


try:
    pir_sensor = digitalio.DigitalInOut(board.D17)
    pir_sensor.direction = digitalio.Direction.INPUT
except Exception as e:
    print("Failed to init PIR sensor: {}".format(e))
    pir_sensor = None

try:
    sound_sensor = digitalio.DigitalInOut(board.D27)
    sound_sensor.direction = digitalio.Direction.INPUT
except Exception as e:
    print("Failed to init sound sensor: {}".format(e))
    sound_sensor = None

print("Sensors initialized. Starting readings...")

motion_detected = False
sound_detected = False

last_serial = 0.0
SERIAL_MIN_INTERVAL = 2.0 

while True:
    try:

        if pir_sensor:
            current_motion = pir_sensor.value
            if current_motion and not motion_detected:
                print("Motion detected")
            elif not current_motion and motion_detected:
                print("Motion stopped.")
            motion_detected = current_motion
        
        if sound_sensor:
            current_sound = not sound_sensor.value
            if current_sound and not sound_detected:
                print("Sound detected")
            elif not current_sound and sound_detected:
                print("Sound stopped")
            sound_detected = current_sound

        # Read temperature and humidity from ESP32 via serial
        now = time.monotonic()
        if ser and (now - last_serial) >= SERIAL_MIN_INTERVAL:
            last_serial = now
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    # Expecting format: TEMP:25.3,HUM:60.2
                    if line.startswith('TEMP:') and 'HUM:' in line:
                        try:
                            parts = line.split(',')
                            temp_c = float(parts[0].split(':')[1])
                            humidity = float(parts[1].split(':')[1])
                            temp_f = temp_c * 9.0 / 5.0 + 32.0
                            print(f"Temp: {temp_c:.1f} C / {temp_f:.1f} F    Humidity: {humidity:.1f}% ")
                        except Exception as parse_error:
                            print(f"Serial parse error: {parse_error} | Raw: {line}")
                    else:
                        print(f"Unexpected serial data: {line}")
            except Exception as error:
                print(f"Serial read error: {error}")

        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting program.")
        break
    except Exception as e:
        print("An unexpected error occurred: {}".format(e))
        break

# Cleanup sensors
if pir_sensor:
    pir_sensor.deinit()
if sound_sensor:
    sound_sensor.deinit()
if ser:
    ser.close()
print("GPIO and serial cleanup complete.")