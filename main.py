import cv2
import numpy as np
from ultralytics import YOLO

class ParticleFilterTracker:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = None
        self.weights = None
        self.roi = None
        self.reference_roi = None

    def initialize(self, roi, frame):
        x, y, w, h = roi
        self.roi = roi
        self.reference_roi = cv2.resize(frame[y:y+h, x:x+w], (w, h))  # Store initial ROI as reference
        self.particles = np.empty((self.num_particles, 4), dtype=np.float32)
        self.particles[:, 0] = x + w * np.random.rand(self.num_particles)
        self.particles[:, 1] = y + h * np.random.rand(self.num_particles)
        self.particles[:, 2] = w * np.random.rand(self.num_particles)
        self.particles[:, 3] = h * np.random.rand(self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self):
        noise = np.random.randn(self.num_particles, 4) * 5  # Adjust noise scale as needed
        self.particles += noise

    def update(self, frame):
        if self.roi is None:
            return None

        x, y, w, h = self.roi
        self.reference_roi = cv2.resize(frame[y:y+h, x:x+w], (w, h))  # Update reference ROI with current frame

        self.weights = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            particle = self.particles[i]
            px, py, pw, ph = particle
            x1, y1, x2, y2 = int(px), int(py), int(px + pw), int(py + ph)

            # Handle potential out-of-frame coordinates
            if (x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]):
                roi_frame = frame[y1:y2, x1:x2]
                if roi_frame.size > 0:
                    roi_frame = cv2.resize(roi_frame, (w, h))  # Resize to match reference dimensions

                    # Ensure correct broadcasting for MSE calculation (expand if necessary)
                    if roi_frame.ndim < self.reference_roi.ndim:
                        roi_frame = np.expand_dims(roi_frame, axis=-1)  # Add channel dimension if missing
                    elif roi_frame.shape[-1] > self.reference_roi.shape[-1]:
                        self.reference_roi = np.expand_dims(self.reference_roi, axis=-1)  # Add channel dimension if missing

                    mse = np.sum((roi_frame - self.reference_roi) ** 2) / np.prod(roi_frame.shape)
                    self.weights[i] = np.exp(-mse)

        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Resample particles
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Estimate the position of the object
        mean_particle = np.mean(self.particles, axis=0)

        # Update reference ROI with a weighted average of current and previous ROI
        alpha = 0.5  # Adjust weight between current and previous ROI
        self.roi = tuple(map(lambda x, y: int(alpha * x + (1 - alpha) * y), self.roi, mean_particle))

        return mean_particle

def main():
    # Load the trained YOLOv8 model
    model = YOLO('E:\\AiTec Internship 2024\\particle filter\\Ahsan Pen.pt')  # Replace with your model path

    # Initialize video capture
    video_path = 'E:\\AiTec Internship 2024\\particle filter\\inputvideo.mp4'  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    tracker_initialized = False
    tracker = ParticleFilterTracker()
    roi = None
    class_name = None
    conf = None
    lost_track_count = 0  # Counter for lost track

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames are available

        if not tracker_initialized:
            # Perform inference
            results = model(frame)

            # Extract bounding boxes and confidence scores
            detections = results[0].boxes

            # Clear frame before drawing
            display_frame = frame.copy()

            # Check if any object is detected
            if len(detections) == 0:
                cv2.putText(display_frame, 'No object detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Use the first detected object for tracking
                detection = detections[0]
                x1, y1, x2, y2 = detection.xyxy[0]
                conf = detection.conf[0]
                cls = detection.cls[0]

                if conf > 0.5:  # Only consider detections with confidence > 0.5
                    roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    class_name = model.names[int(cls)]
                    tracker.initialize(roi, frame)
                    tracker_initialized = True
                else:
                    cv2.putText(display_frame, 'No object detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Predict and update the tracker
            tracker.predict()
            estimated_roi = tracker.update(frame)

            if estimated_roi is not None:
                # Draw bounding box and display confidence and class
                x, y, w, h = estimated_roi
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                lost_track_count = 0  # Reset lost track counter
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                lost_track_count += 1
                if lost_track_count > 30:  # Re-initialize tracking after 30 frames
                    tracker_initialized = False
                    lost_track_count = 0

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
