import cv2
import numpy as np
import time
import json
import os
"""
NAVIGATION CONTROL SYSTEM USING COLOR DETECTION

Navigation of AUV through the gate and obstacle avoidance using color detection and PID control.
This script captures video input, processes each frame to detect obstacles and gates based on color,
and uses PID controllers to adjust the AUV's movement accordingly.


PID Class: Implements a simple PID controller for controlling the AUV's position
Trackbar Creation: Creates trackbars for tuning HSV color thresholds for obstacle and gate detection.
Loading configuration: Loads HSV threshold values and PID parameters from a JSON configuration file.
Main Processing Loop: Captures frames, processes them for obstacle and gate detection, computes PID outputs,
and displays the results along with control information.

"""


# Loading configuration from JSON
with open("Navigation/Config/auv_config.json", "r") as f:
    config = json.load(f)


class PID:
    def __init__(self, kp=config["PID"]["Kp"], ki=config["PID"]["Ki"], kd=config["PID"]["Kd"]):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.setpoint = 0
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    # function to compute PID output
    def compute(self, current_value):
        error = self.setpoint - current_value
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt > 0.01:  # Sample time
            # PID calculation
            p_term = self.Kp * error
            self.integral += error * dt
            i_term = self.Ki * self.integral
            d_term = self.Kd * (error - self.last_error) / dt
            
            output = p_term + i_term + d_term
            output = max(-400, min(400, output))
            
            self.last_error = error
            self.last_time = current_time
            return output
        return 0
    
    def reset(self):
        self.integral = 0
        self.last_error = 0

# Trackbar callback (required but does nothing)
def nothing(x):
    pass

# Initialize video capture
print("=" * 60)
print("ROV VISION CONTROL SYSTEM")
print("=" * 60)
video_path = input("\nEnter video path (or press Enter for webcam): ").strip()

cap = cv2.VideoCapture(video_path if video_path else 0)
if not cap.isOpened():
    print("Error: Could not open video source!")
    exit()

# Get video dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video!")
    exit()

height, width = frame.shape[:2]
print(f"\nVideo: {width}x{height}")
print(f"Target center: ({width//2}, {height//2})")

# Create windows and trackbars
cv2.namedWindow('Controls')
cv2.namedWindow('Main View')
cv2.namedWindow('Obstacle Mask')
cv2.namedWindow('Gate Mask')

# Obstacle detection trackbars
cv2.createTrackbar('Obs_H_Low', 'Controls', 0, 179, nothing)
cv2.createTrackbar('Obs_H_High', 'Controls', 179, 179, nothing)
cv2.createTrackbar('Obs_S_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('Obs_S_High', 'Controls', 255, 255, nothing)
cv2.createTrackbar('Obs_V_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('Obs_V_High', 'Controls', 255, 255, nothing)

# Gate detection trackbars
cv2.createTrackbar('Gate_H_Low', 'Controls', 0, 179, nothing)
cv2.createTrackbar('Gate_H_High', 'Controls', 179, 179, nothing)
cv2.createTrackbar('Gate_S_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('Gate_S_High', 'Controls', 255, 255, nothing)
cv2.createTrackbar('Gate_V_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('Gate_V_High', 'Controls', 255, 255, nothing)

# set initial trackbar positions from config
cv2.setTrackbarPos('Obs_H_Low', 'Controls', config["Obstacle_HSV"]["H_low"])
cv2.setTrackbarPos('Obs_H_High', 'Controls', config["Obstacle_HSV"]["H_high"])
cv2.setTrackbarPos('Obs_S_Low', 'Controls', config["Obstacle_HSV"]["S_low"])
cv2.setTrackbarPos('Obs_S_High', 'Controls', config["Obstacle_HSV"]["S_high"])
cv2.setTrackbarPos('Obs_V_Low', 'Controls', config["Obstacle_HSV"]["V_low"])
cv2.setTrackbarPos('Obs_V_High', 'Controls', config["Obstacle_HSV"]["V_high"])

cv2.setTrackbarPos('Gate_H_Low', 'Controls', config["Gate_HSV"]["H_low"])
cv2.setTrackbarPos('Gate_H_High', 'Controls', config["Gate_HSV"]["H_high"])
cv2.setTrackbarPos('Gate_S_Low', 'Controls', config["Gate_HSV"]["S_low"])
cv2.setTrackbarPos('Gate_S_High', 'Controls', config["Gate_HSV"]["S_high"])
cv2.setTrackbarPos('Gate_V_Low', 'Controls', config["Gate_HSV"]["V_low"])
cv2.setTrackbarPos('Gate_V_High', 'Controls', config["Gate_HSV"]["V_high"])


# Initialize PID controllers
x_pid = PID(kp=0.7, ki=0.0, kd=0.1)
y_pid = PID(kp=0.7, ki=0.0, kd=0.1)
x_pid.setpoint = width // 2
y_pid.setpoint = height // 2

print("\n" + "=" * 60)
print("CONTROLS:")
print("  ESC   - Exit")
print("  SPACE - Pause/Resume")
print("  R     - Reset PID")
print("  Adjust trackbars to tune color detection")
print("=" * 60 + "\n")

frame_count = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        
        # Loop video
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        display = frame.copy()
        
        # Get trackbar values
        obs_h_low = cv2.getTrackbarPos('Obs_H_Low', 'Controls')
        obs_h_high = cv2.getTrackbarPos('Obs_H_High', 'Controls')
        obs_s_low = cv2.getTrackbarPos('Obs_S_Low', 'Controls')
        obs_s_high = cv2.getTrackbarPos('Obs_S_High', 'Controls')
        obs_v_low = cv2.getTrackbarPos('Obs_V_Low', 'Controls')
        obs_v_high = cv2.getTrackbarPos('Obs_V_High', 'Controls')
        
        gate_h_low = cv2.getTrackbarPos('Gate_H_Low', 'Controls')
        gate_h_high = cv2.getTrackbarPos('Gate_H_High', 'Controls')
        gate_s_low = cv2.getTrackbarPos('Gate_S_Low', 'Controls')
        gate_s_high = cv2.getTrackbarPos('Gate_S_High', 'Controls')
        gate_v_low = cv2.getTrackbarPos('Gate_V_Low', 'Controls')
        gate_v_high = cv2.getTrackbarPos('Gate_V_High', 'Controls')
            
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Draw crosshairs
        cv2.line(display, (width//2, 0), (width//2, height), (0, 0, 0), 2)
        cv2.line(display, (0, height//2), (width, height//2), (0, 0, 0), 2)
        
        # OBSTACLE DETECTION
        obs_mask = cv2.inRange(hsv, 
                               np.array([obs_h_low, obs_s_low, obs_v_low]),
                               np.array([obs_h_high, obs_s_high, obs_v_high]))
        
        kernel = np.ones((5, 5), np.uint8)
        obs_mask = cv2.dilate(obs_mask, kernel)
        
        obs_contours, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_in_path = False
        for contour in obs_contours:
            if cv2.contourArea(contour) > 3000:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w//2, y + h//2
                
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.circle(display, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(display, "OBSTACLE", (cx-40, cy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Check if in center
                if abs(cx - width//2) < width//3:
                    obstacle_in_path = True
        
        # GATE DETECTION
        gate_mask = cv2.inRange(hsv,
                                np.array([gate_h_low, gate_s_low, gate_v_low]),
                                np.array([gate_h_high, gate_s_high, gate_v_high]))
        
        gate_mask = cv2.dilate(gate_mask, kernel)
        
        gate_contours, _ = cv2.findContours(gate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Default PWM (neutral)
        pwm_lr = 1500
        pwm_ud = 1500
        pid_x = 0
        pid_y = 0
        
        gate_found = False
        for contour in gate_contours:
            if cv2.contourArea(contour) > 3000:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w//2, y + h//2
                
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)
                cv2.putText(display, "GATE", (cx-20, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate PID
                pid_x = x_pid.compute(cx)
                pid_y = y_pid.compute(cy)
                
                # Apply to PWM
                pwm_lr = int(1500 + pid_x)
                pwm_ud = int(1500 + pid_y)
                pwm_lr = max(1100, min(1900, pwm_lr))
                pwm_ud = max(1100, min(1900, pwm_ud))
                
                # Draw tracking line
                cv2.line(display, (width//2, height//2), (cx, cy), (255, 255, 0), 2)
                
                gate_found = True
                break
        
        # Status display
        status_y = 30
        cv2.putText(display, f"Frame: {frame_count}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(display, f"PWM L/R: {pwm_lr}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(display, f"PWM U/D: {pwm_ud}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        
        if gate_found:
            cv2.putText(display, "STATUS: TRACKING GATE", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif obstacle_in_path:
            cv2.putText(display, "STATUS: OBSTACLE IN PATH!", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, "STATUS: SEARCHING...", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show windows
        cv2.imshow('Main View', display)
        cv2.imshow('Obstacle Mask', obs_mask)
        cv2.imshow('Gate Mask', gate_mask)
        
        # Console output
        if frame_count % 10 == 0:
            print(f"Frame {frame_count:5d} | L/R: {pwm_lr:4d} | U/D: {pwm_ud:4d} | "
                  f"PID_X: {pid_x:6.1f} | PID_Y: {pid_y:6.1f}")
    
    # Keyboard controls
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord('r') or key == ord('R'):
        x_pid.reset()
        y_pid.reset()
        print("PID Reset")



cap.release()
cv2.destroyAllWindows()
print("\nProgram ended.")