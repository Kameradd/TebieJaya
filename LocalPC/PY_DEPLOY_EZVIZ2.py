# save as yolo_rtsp_tracker_final.py
import sys
import os
import time
from collections import deque
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
from pathlib import Path
import torch
import boxmot
from datetime import datetime
import threading

# Reduce RTSP latency for OpenCV FFmpeg backend
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

# ++++++++++ Thread-Safe Video Stream Reader (Unchanged) ++++++++++
class VideoStream:
    """A thread-safe class to read frames from an RTSP stream."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
            raise IOError(f"Cannot open video stream at {src}")
            
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.grabbed else None
        return frame

    def isOpened(self):
        return self.stream.isOpened()

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# -------- RTSP CAMERA CONFIG --------
VERIFICATION_CODE = "ILBUCZ"
IP_ADDRESS = "192.168.10.64"
RTSP_URL = f"rtsp://admin:{VERIFICATION_CODE}@{IP_ADDRESS}:554"

# -------- Paths --------
# Use absolute paths for reliability
CWD = os.path.dirname(os.path.realpath(__file__))
YOLO_WEIGHTS = os.path.join(CWD, "best.pt")
OUTPUT_VIDEO = os.path.join(CWD, "goat_tracking_output_live.mp4")
OUTPUT_CSV = os.path.join(CWD, "goat_tracks_live.csv")
REID_WEIGHTS = Path(os.path.join(CWD, "boxmot/reid_weights/osnet_x0_25_msmt17.pt"))
TRACKER_CONFIG = Path(os.path.join(os.path.dirname(boxmot.__file__), "configs/trackers/strongsort.yaml"))

# -------- Parameters --------
STAND_RATIO = 1.2
USE_TEMP_PROXY = True
TEMP_PROXY_A = 0.05
TEMP_PROXY_B = 20.0

# (All utility functions like bbox_to_int, compute_color_histogram, etc. are the same as original)
# -------- Utility functions --------
def bbox_to_int(bbox):
    x1, y1, x2, y2 = bbox
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def posture_from_wh(w, h):
    if w <= 0: return "unknown"
    return "standing" if (h / w) >= STAND_RATIO else "sitting"

def compute_color_histogram(img, bbox, size=(32,32)):
    x1,y1,x2,y2 = bbox_to_int(bbox)
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img-1, x2), min(h_img-1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(256, dtype=float)
    try:
        crop = cv2.resize(crop, size)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except:
        return np.zeros(256, dtype=float)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# (All tracker helper classes and functions are the same)
# -------- Tracker init helper --------
def init_boxmot_strongsort(device="cuda", reid_weights=REID_WEIGHTS):
    from boxmot import create_tracker
    print("Using tracker config:", TRACKER_CONFIG, TRACKER_CONFIG.exists())
    print("Using ReID weights:", reid_weights, reid_weights.exists())
    try:
        tr = create_tracker(
            tracker_type="strongsort",
            tracker_config=TRACKER_CONFIG,
            reid_weights=reid_weights,
            device=device
        )
        print("✅ Initialized BoxMOT StrongSORT tracker.")
        return tr
    except Exception as e:
        print(f"❌ Failed to init BoxMOT StrongSORT: {e}")
        return None

# -------- Fallback simple tracker --------
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=60, max_distance=100):
        self.next_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, dets, frame=None):
        if dets is None or len(dets)==0:
            to_delete=[]
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid]+=1
                if self.disappeared[oid]>self.max_disappeared:
                    to_delete.append(oid)
            for oid in to_delete:
                del self.objects[oid]; del self.disappeared[oid]
            return np.empty((0,5))
        dets=np.asarray(dets)
        centroids=np.array([[(r[0]+r[2])/2.0,(r[1]+r[3])/2.0] for r in dets])
        if len(self.objects)==0:
            outputs=[]
            for i,row in enumerate(dets):
                tid=self.next_id; self.next_id+=1
                self.objects[tid]=centroids[i]; self.disappeared[tid]=0
                outputs.append([row[0],row[1],row[2],row[3],float(tid)])
            return np.array(outputs)
        object_ids=list(self.objects.keys())
        object_centroids=np.array([self.objects[oid] for oid in object_ids])
        D=np.linalg.norm(object_centroids[:,None,:]-centroids[None,:,:],axis=2)
        rows=D.min(axis=1).argsort()
        cols=D.argmin(axis=1)[rows]
        assigned_rows, assigned_cols=set(),set()
        outputs=[]
        for r,c in zip(rows,cols):
            if r in assigned_rows or c in assigned_cols: continue
            if D[r,c]>self.max_distance: continue
            oid=object_ids[r]; self.objects[oid]=centroids[c]; self.disappeared[oid]=0
            assigned_rows.add(r); assigned_cols.add(c)
            row=dets[c]; outputs.append([row[0],row[1],row[2],row[3],float(oid)])
        for i in range(len(dets)):
            if i in assigned_cols: continue
            row=dets[i]; tid=self.next_id; self.next_id+=1
            self.objects[tid]=centroids[i]; self.disappeared[tid]=0
            outputs.append([row[0],row[1],row[2],row[3],float(tid)])
        for idx_r, oid in enumerate(object_ids):
            if idx_r not in assigned_rows:
                self.disappeared[oid]+=1
                if self.disappeared[oid]>self.max_disappeared:
                    del self.objects[oid]; del self.disappeared[oid]
        return np.array(outputs)

# -------- Tracker update generic --------
def update_tracker_generic(tracker_obj, detections, frame):
    if tracker_obj is None: return np.empty((0,5))
    try:
        if hasattr(tracker_obj,"update_tracks"): return tracker_obj.update_tracks(detections,frame=frame)
        if hasattr(tracker_obj,"update"):
            try: return tracker_obj.update(detections,frame)
            except TypeError: return tracker_obj.update(detections)
        if hasattr(tracker_obj,"track"): return tracker_obj.track(detections,frame)
        if callable(tracker_obj): return tracker_obj(detections,frame)
    except Exception as e:
        print("Tracker update exception:",e)
        return np.empty((0,5))
    return np.empty((0,5))

# -------- Parse tracks generic --------
def parse_tracks_generic(tracks_out):
    parsed=[]
    if tracks_out is None: return parsed
    if isinstance(tracks_out,(list,tuple,np.ndarray)) and len(tracks_out)>0:
        if isinstance(tracks_out,np.ndarray) and tracks_out.dtype!=object:
            for row in tracks_out:
                row=np.array(row)
                if row.size>=5:
                    tid=int(row[4])
                    bbox=(float(row[0]),float(row[1]),float(row[2]),float(row[3]))
                    parsed.append((tid,bbox))
            return parsed
        for t in tracks_out:
            try:
                tid=None
                if hasattr(t,"track_id"): tid=int(t.track_id)
                elif hasattr(t,"id"): tid=int(t.id)
                elif hasattr(t,"trackId"): tid=int(t.trackId)
                elif hasattr(t,"idx"): tid=int(t.idx)
                bbox=None
                if hasattr(t,"tlbr"): bbox=t.tlbr
                elif hasattr(t,"to_tlbr"): bbox=t.to_tlbr()
                elif hasattr(t,"tlwh"):
                    tlwh=t.tlwh; bbox=(tlwh[0],tlwh[1],tlwh[0]+tlwh[2],tlwh[1]+tlwh[3])
                elif hasattr(t,"bbox"): bbox=t.bbox
                if bbox is None and hasattr(t,"__len__"):
                    try:
                        arr=np.array(t)
                        if arr.size>=5:
                            bbox=(float(arr[0]),float(arr[1]),float(arr[2]),float(arr[3]))
                            if tid is None: tid=int(arr[4])
                    except: pass
                if bbox is not None and tid is not None: parsed.append((tid,tuple(map(float,bbox))))
            except: continue
        return parsed
    return parsed

# -------- Main pipeline --------
def run_pipeline(yolo_weights, video_source, output_video_path, output_csv_path,
                 use_boxmot=True, device="cpu", conf_thres=0.25):
    
    from boxmot import create_tracker # Keep import here to avoid circular dependency issues
    
    print("Loading YOLO model:", yolo_weights)
    yolo = YOLO(yolo_weights)

    TRACKER_OBJ = None
    if use_boxmot:
        if not REID_WEIGHTS.exists():
            print("⚠️ ReID weights not found, fallback to SimpleCentroid tracker.")
            TRACKER_OBJ=SimpleCentroidTracker()
        else:
            TRACKER_OBJ = init_boxmot_strongsort(device=device, reid_weights=REID_WEIGHTS)
        if TRACKER_OBJ is None:
            print("BoxMOT StrongSORT init failed, falling back to simple tracker.")
            TRACKER_OBJ = SimpleCentroidTracker()
    else:
        TRACKER_OBJ = SimpleCentroidTracker()

    goat_data = {}
    
    while True: 
        stream = None
        try:
            print(f"🔄 Connecting to video source: {video_source}")
            stream = VideoStream(video_source).start()
            time.sleep(2.0)

            if not stream.isOpened() or stream.read() is None:
                raise IOError("Stream is not open or first frame is empty.")

            print("✅ Stream opened successfully! Starting processing...")
            print("Press 'q' in the video window to quit.")
            
            first_frame = stream.read()
            height, width, _ = first_frame.shape
            
            OUTPUT_FPS = 15.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, OUTPUT_FPS, (width, height))

            frame_idx = 0
            start_time = time.time()
            
            while True:
                frame = stream.read()
                if frame is None:
                    print("‼️ Empty frame received, possible stream loss.")
                    time.sleep(0.5)
                    break
                
                frame_idx += 1
                video_time = time.time() - start_time

                results = yolo(frame, imgsz=640, conf=conf_thres, max_det=100, verbose=False)
                r = results[0]

                dets = []
                for box in r.boxes:
                    xy = box.xyxy.cpu().numpy().reshape(-1)
                    conf = safe_float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    dets.append([xy[0], xy[1], xy[2], xy[3], conf, cls])
                dets_arr = np.array(dets) if dets else np.zeros((0, 6))

                tracks_out = update_tracker_generic(TRACKER_OBJ, dets_arr, frame)
                parsed = parse_tracks_generic(tracks_out)
                
                # <<< MODIFIED/RESTORED SECTION: Process and draw in one loop >>>
                for tid, bbox in parsed:
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    
                    # --- Data Gathering (Restored from Original) ---
                    posture = posture_from_wh(w, h)
                    
                    if tid not in goat_data:
                        goat_data[tid] = {
                            "first_seen_time": video_time,
                            "last_seen_time": video_time,
                            "events": [],
                            "current_posture": posture,
                            "since_time": video_time,
                            "estimated_temp_history": [] # <<< RE-ADDED
                        }
                    else:
                        g = goat_data[tid]
                        prev = g["current_posture"]
                        if posture != prev:
                            duration = video_time - g["since_time"]
                            g["events"].append({"time": video_time, "from": prev, "to": posture, "duration_prev": duration})
                            g["current_posture"] = posture
                            g["since_time"] = video_time
                        g["last_seen_time"] = video_time

                    # <<< RE-ADDED: Temperature Proxy Calculation >>>
                    if USE_TEMP_PROXY:
                        xi1_crop, yi1_crop, xi2_crop, yi2_crop = bbox_to_int(bbox)
                        crop = frame[yi1_crop:yi2_crop, xi1_crop:xi2_crop]
                        if crop.size > 0:
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            v_mean = float(np.mean(hsv[:, :, 2]))
                            temp_proxy = TEMP_PROXY_A * v_mean + TEMP_PROXY_B
                            goat_data[tid]["estimated_temp_history"].append((video_time, temp_proxy))

                    # --- Drawing Logic (Restored from Original) ---
                    xi1, yi1, xi2, yi2 = bbox_to_int(bbox)
                    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
                    
                    # Draw ID with background
                    label = f"ID:{tid}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (xi1, max(0, yi1 - text_h - 8)), (xi1 + text_w, yi1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (xi1, yi1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    # <<< RE-ADDED: Draw Posture and Temperature >>>
                    g_draw = goat_data.get(tid, None)
                    posture_label = g_draw["current_posture"] if g_draw else "?"
                    cv2.putText(frame, posture_label, (xi1, yi2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    if g_draw and g_draw["estimated_temp_history"]:
                        last_temp = g_draw["estimated_temp_history"][-1][1]
                        cv2.putText(frame, f"T:{last_temp:.1f}", (xi1, yi2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow('YOLO Live Tracking', frame)
                out.write(frame)

                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, stopping the application.")
                    stream.stop()
                    out.release()
                    cv2.destroyAllWindows()
                    save_and_exit(goat_data, output_csv_path, output_video_path)
                    return
            
        except (IOError, cv2.error) as e:
            print(f"❌ Connection Error: {e}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break
        finally:
            if stream:
                stream.stop()
            if 'out' in locals() and out.isOpened():
                out.release()
            cv2.destroyAllWindows()
    
    save_and_exit(goat_data, output_csv_path, output_video_path)

# (save_and_exit function is the same, but restored to include temperature)
def save_and_exit(goat_data, output_csv_path, output_video_path):
    """Builds and saves the CSV report before exiting."""
    rows = []
    for tid, g in goat_data.items():
        total_time_seen = g.get("last_seen_time", g["first_seen_time"]) - g["first_seen_time"]
        for ev in g["events"]:
            rows.append({"id": tid, "event_time_s": ev["time"], "from": ev["from"], "to": ev["to"], "duration_prev_s": ev["duration_prev"]})
        
        # <<< RE-ADDED: Add temperature to summary >>>
        avg_temp = np.mean([t for (_, t) in g["estimated_temp_history"]]) if g["estimated_temp_history"] else None
        
        rows.append({
            "id": tid, "event_time_s": "summary",
            "first_seen_s": g["first_seen_time"],
            "last_seen_s": g.get("last_seen_time", g["first_seen_time"]),
            "total_time_seen_s": total_time_seen,
            "current_posture": g["current_posture"],
            "last_posture_since_s": g["since_time"],
            "avg_temp_proxy": avg_temp
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        print("✅ Data saved to:", output_csv_path)
    else:
        print("No data collected to save.")
    print("Outputs video saved to:", output_video_path)

# -------- Run --------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        from boxmot import create_tracker
        run_pipeline(YOLO_WEIGHTS, RTSP_URL, OUTPUT_VIDEO, OUTPUT_CSV, use_boxmot=True, device=device, conf_thres=0.4)
    except ImportError as e:
        print(f"Could not import boxmot, running with simple tracker. Error: {e}")
        run_pipeline(YOLO_WEIGHTS, RTSP_URL, OUTPUT_VIDEO, OUTPUT_CSV, use_boxmot=False, device=device, conf_thres=0.4)