import cv2
import numpy as np
import time

# Simulated ROS publishers (for testing without ROS)
class SimPublisher:
    def __init__(self, topic_name):
        self.topic = topic_name
        self.last_value = None
    
    def publish(self, value):
        self.last_value = value

pub_lr = SimPublisher('pwm_lr_topic')
pub_ud = SimPublisher('pwm_ud_topic')

# Simple PID Controller
class PID:
    def __init__(self):
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0
        self.setpoint = 0.0
        self.sample_time = 0.01
        self.output_limits = (-400, 400)
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def __call__(self, current_value):
        error = self.setpoint - current_value
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt >= self.sample_time:
            # Proportional term
            p_term = self.Kp * error
            
            # Integral term
            self.integral += error * dt
            i_term = self.Ki * self.integral
            
            # Derivative term
            derivative = (error - self.last_error) / dt if dt > 0 else 0
            d_term = self.Kd * derivative
            
            # Calculate output
            output = p_term + i_term + d_term
            
            # Apply limits
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
            
            self.last_error = error
            self.last_time = current_time
            
            return output
        return 0

# Trackbar callback
def callback(x):
    pass

# Create window first
cv2.namedWindow('image')
cv2.namedWindow('Image')
cv2.namedWindow('Mask - Obstacle')
cv2.namedWindow('Mask - Gate')

# Create trackbars for obstacle detection
cv2.createTrackbar('lowH', 'image', 0, 179, callback)
cv2.createTrackbar('highH', 'image', 179, 179, callback)
cv2.createTrackbar('lowS', 'image', 0, 255, callback)
cv2.createTrackbar('highS', 'image', 255, 255, callback)
cv2.createTrackbar('lowV', 'image', 0, 255, callback)
cv2.createTrackbar('highV', 'image', 255, 255, callback)

# Create trackbars for gate detection
cv2.createTrackbar('lowH_gate', 'image', 0, 179, callback)
cv2.createTrackbar('highH_gate', 'image', 179, 179, callback)
cv2.createTrackbar('lowS_gate', 'image', 0, 255, callback)
cv2.createTrackbar('highS_gate', 'image', 255, 255, callback)
cv2.createTrackbar('lowV_gate', 'image', 0, 255, callback)
cv2.createTrackbar('highV_gate', 'image', 255, 255, callback)

# PID Setup
TargetX = 320  # Will be updated based on actual frame size
TargetY = 240
P = 0.7
I = 0.0
D = 0.1

x_pid = PID()
x_pid.Kp = P
x_pid.Kd = D
x_pid.Ki = I
x_pid.setpoint = TargetX
x_pid.sample_time = 0.01
x_pid.output_limits = (-400, 400)

y_pid = PID()
y_pid.Kp = P
y_pid.Kd = D
y_pid.Ki = I
y_pid.setpoint = TargetY
y_pid.sample_time = 0.01
y_pid.output_limits = (-400, 400)

# Prompt user for video file
print("=" * 60)
print("ROV VISION CONTROL SYSTEM - VIDEO MODE")
print("=" * 60)
print("\nEnter the path to your test video file:")
print("(Press Enter to use webcam instead)")
video_path = input("> ").strip()

if video_path == "":
    cap = cv2.VideoCapture(0)
    print("\nUsing webcam...")
else:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nError: Could not open video file: {video_path}")
        print("Falling back to webcam...")
        cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source!")
    exit()

# Get first frame to determine dimensions
ret, img = cap.read()
if not ret:
    print("Error: Could not read from video source!")
    cap.release()
    exit()

height, width = img.shape[:2]
TargetX = width // 2
TargetY = height // 2
x_pid.setpoint = TargetX
y_pid.setpoint = TargetY

print(f"\nVideo dimensions: {width}x{height}")
print(f"Target center: ({TargetX}, {TargetY})")
print("\n" + "=" * 60)
print("CONTROLS:")
print("=" * 60)
print("ESC - Exit program")
print("SPACE - Pause/Resume")
print("R - Reset PID controllers")
print("Adjust trackbars in 'image' window to tune color detection")
print("=" * 60)

frame_count = 0
paused = False
running = True

while running:
    if not paused:
        ret, img = cap.read()
        
        # Loop video if it ends
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
            if not ret:
                break
        
        frame_count += 1
        
        # Read trackbar positions
        ilowH = cv2.getTrackbarPos('lowH', 'image')
        ihighH = cv2.getTrackbarPos('highH', 'image')
        ilowS = cv2.getTrackbarPos('lowS', 'image')
        ihighS = cv2.getTrackbarPos('highS', 'image')
        ilowV = cv2.getTrackbarPos('lowV', 'image')
        ihighV = cv2.getTrackbarPos('highV', 'image')
        
        ilowH_gate = cv2.getTrackbarPos('lowH_gate', 'image')
        ihighH_gate = cv2.getTrackbarPos('highH_gate', 'image')
        ilowS_gate = cv2.getTrackbarPos('lowS_gate', 'image')
        ihighS_gate = cv2.getTrackbarPos('highS_gate', 'image')
        ilowV_gate = cv2.getTrackbarPos('lowV_gate', 'image')
        ihighV_gate = cv2.getTrackbarPos('highV_gate', 'image')
        
        # Initialize PWM values (neutral)
        pwm_lr = 1500
        pwm_ud = 1500
        x_output = 0
        y_output = 0
        
        # Create working copies
        frame = img.copy()
        frame2 = img.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Draw target crosshairs on frame2
        cv2.line(frame2, (TargetX, 0), (TargetX, height), (0, 0, 0), 2)
        cv2.line(frame2, (0, TargetY), (width, TargetY), (0, 0, 0), 2)
        
        # === OBSTACLE DETECTION ===
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        
        # Gamma correction
        frame_corrected = np.array(255 * (frame / 255) ** 0.6, dtype='uint8')
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours for obstacles
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000 and len(contour) >= 5:  # Need at least 5 points for minAreaRect
                x, y, w, h = cv2.boundingRect(contour)
                ccx = int(x + (w / 2))
                ccy = int(y + (h / 2))
                
                # Draw rotated rectangle
                try:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    if len(box) > 0:
                        cv2.drawContours(frame2, [box], 0, (0, 0, 255), 2)
                except:
                    # Fallback to regular rectangle if rotated rect fails
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                cv2.circle(frame2, (ccx, ccy), 5, (255, 0, 0), -1)
                cv2.putText(frame2, "OBSTACLE", (ccx - 40, ccy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Check if obstacle is in center path
                margin = width // 3
                if TargetX - margin <= ccx <= TargetX + margin:
                    obstacle_detected = True
                    cv2.putText(frame2, "OBSTACLE IN PATH!", (50, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # === GATE DETECTION ===
        lower_hsv_gate = np.array([ilowH_gate, ilowS_gate, ilowV_gate])
        higher_hsv_gate = np.array([ihighH_gate, ihighS_gate, ihighV_gate])
        mask_gate = cv2.inRange(hsv, lower_hsv_gate, higher_hsv_gate)
        
        # Dilate gate mask
        dilated_gate = cv2.dilate(mask_gate, kernel, iterations=1)
        
        # Find contours for gates
        contours_gate, _ = cv2.findContours(dilated_gate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gate_detected = False
        for contour_gate in contours_gate:
            area_gate = cv2.contourArea(contour_gate)
            if area_gate > 3000 and len(contour_gate) >= 5:  # Need at least 5 points for minAreaRect
                x_gate, y_gate, w_gate, h_gate = cv2.boundingRect(contour_gate)
                ccx_gate = int(x_gate + (w_gate / 2))
                ccy_gate = int(y_gate + (h_gate / 2))
                
                # Draw rotated rectangle
                try:
                    rect_gate = cv2.minAreaRect(contour_gate)
                    box_gate = cv2.boxPoints(rect_gate)
                    box_gate = np.int0(box_gate)
                    if len(box_gate) > 0:
                        cv2.drawContours(frame2, [box_gate], 0, (0, 255, 0), 2)
                except:
                    # Fallback to regular rectangle if rotated rect fails
                    cv2.rectangle(frame2, (x_gate, y_gate), 
                                (x_gate + w_gate, y_gate + h_gate), (0, 255, 0), 2)
                
                cv2.circle(frame2, (ccx_gate, ccy_gate), 5, (255, 255, 0), -1)
                cv2.putText(frame2, "GATE", (ccx_gate - 20, ccy_gate - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate PID output
                x_output = x_pid(ccx_gate)
                y_output = y_pid(ccy_gate)
                
                # Apply PID to PWM (FIXED)
                pwm_lr = int(1500 + x_output)
                pwm_ud = int(1500 + y_output)
                
                # Clamp PWM values
                pwm_lr = max(1100, min(1900, pwm_lr))
                pwm_ud = max(1100, min(1900, pwm_ud))
                
                gate_detected = True
                
                # Draw error lines
                cv2.line(frame2, (TargetX, TargetY), (ccx_gate, ccy_gate), (255, 255, 0), 2)
                
                # Only process first (largest) gate
                break
        
        # Publish PWM values
        pub_lr.publish(pwm_lr)
        pub_ud.publish(pwm_ud)
        
        # Draw status information
        status_y = 30
        cv2.putText(frame2, f"Frame: {frame_count}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame2, f"PWM L/R: {pwm_lr}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame2, f"PWM U/D: {pwm_ud}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame2, f"PID X: {x_output:.1f}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame2, f"PID Y: {y_output:.1f}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        
        if gate_detected:
            cv2.putText(frame2, "STATUS: TRACKING GATE", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif obstacle_detected:
            cv2.putText(frame2, "STATUS: OBSTACLE DETECTED", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame2, "STATUS: SEARCHING...", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display windows
        cv2.imshow("Image", frame2)
        cv2.imshow("Mask - Obstacle", mask)
        cv2.imshow("Mask - Gate", mask_gate)
        
        # Console output
        if frame_count % 10 == 0:  # Print every 10 frames
            print(f"Frame {frame_count:5d} | PWM L/R: {pwm_lr:4d} | PWM U/D: {pwm_ud:4d} | "
                  f"PID X: {x_output:6.1f} | PID Y: {y_output:6.1f}")
    
    # Handle keyboard input
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        running = False
    elif key == ord(' '):  # SPACE
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord('r') or key == ord('R'):  # Reset PID
        x_pid.integral = 0
        y_pid.integral = 0
        x_pid.last_error = 0
        y_pid.last_error = 0
        print("PID controllers reset")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nProgram terminated.")