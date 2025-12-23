import numpy as np
from typing import Optional
from dataclasses import dataclass
import time


@dataclass
class FrenetTrackedObject:
    id: int
    s: float
    vs: float
    d: float
    vd: float
    x: float
    y: float
    state: np.ndarray
    covariance: np.ndarray
    last_seen: float
    age: int
    is_static: bool = False
    position_history: list = None
    vx: float = 0.0
    vy: float = 0.0
    
    def __post_init__(self):
        if self.position_history is None:
            self.position_history = []
    
    def get_speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)
    
    def get_velocity_magnitude_frenet(self) -> float:
        return np.sqrt(self.vs**2 + self.vd**2)


class FrenetEKF:
    def __init__(self, dt: float = 0.1, 
                 p_vs: float = 0.5, p_d: float = 0.8, p_vd: float = 0.8,
                 process_noise_s: float = 0.1, process_noise_vs: float = 0.5,
                 process_noise_d: float = 0.1, process_noise_vd: float = 0.5,
                 meas_noise_s: float = 0.2, meas_noise_vs: float = 0.5,
                 meas_noise_d: float = 0.2, meas_noise_vd: float = 0.5,
                 track_length: float = 100.0, is_circular_track: bool = False):
        self.dt = dt
        self.p_vs = p_vs
        self.p_d = p_d
        self.p_vd = p_vd
        self.track_length = track_length
        self.is_circular_track = is_circular_track
        
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [0, 0, 0],
            [p_vs, 0, 0],
            [0, p_d, 0],
            [0, 0, p_vd]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],  
            [0, 0, 1, 0] 
        ])
        
        self.Q = np.diag([
            process_noise_s,
            process_noise_vs,
            process_noise_d,
            process_noise_vd
        ])
        
        self.R = np.diag([
            meas_noise_s,
            meas_noise_d
        ])
        
    def predict(self, state: np.ndarray, P: np.ndarray) -> tuple:
        state_pred = self.F @ state
        
        if self.is_circular_track:
            state_pred[0] = state_pred[0] % self.track_length
        
        P_pred = self.F @ P @ self.F.T + self.Q
        
        return state_pred, P_pred
        
    def update(self, state_pred: np.ndarray, P_pred: np.ndarray,
               measurement: np.ndarray) -> tuple:
        innovation = measurement - self.H @ state_pred
        
        if self.is_circular_track:
            innovation[0] = self._normalize_s(innovation[0])
        
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        state = state_pred + K @ innovation
        if self.is_circular_track:
            state[0] = state[0] % self.track_length
        
        P = (np.eye(4) - K @ self.H) @ P_pred
        
        return state, P
        
    def _normalize_s(self, s_diff: float) -> float:
        if s_diff > self.track_length / 2:
            return s_diff - self.track_length
        elif s_diff < -self.track_length / 2:
            return s_diff + self.track_length
        return s_diff


class StaticDynamicClassifier:
    def __init__(self, window_size: int = 10, 
                 static_threshold: float = 0.15,
                 vote_threshold: int = 7):
        self.window_size = window_size
        self.static_threshold = static_threshold
        self.vote_threshold = vote_threshold
        
    def classify(self, position_history: list) -> bool:
        if len(position_history) < self.window_size:
            return False
            
        recent_positions = np.array([[p[0], p[1]] for p in position_history[-self.window_size:]])
        
        std_s = np.std(recent_positions[:, 0])
        std_d = np.std(recent_positions[:, 1])
        
        total_std = np.sqrt(std_s**2 + std_d**2)
        
        return total_std < self.static_threshold


class FrenetObjectTracker:
    def __init__(self, dt: float = 0.1, max_distance: float = 2.0,
                 max_age: int = 5, track_length: float = 100.0,
                 vs_target: float = 5.0, los_distance: float = 10.0,
                 is_circular_track: bool = False,
                 **ekf_params):
        self.dt = dt
        self.max_distance = max_distance
        self.max_age = max_age
        self.track_length = track_length
        self.vs_target = vs_target
        self.los_distance = los_distance
        self.is_circular_track = is_circular_track
        
        self.ekf = FrenetEKF(dt=dt, track_length=track_length, 
                            is_circular_track=is_circular_track, **ekf_params)
        self.classifier = StaticDynamicClassifier()
        
        self.tracks = {}
        self.next_id = 0
        self.previous_detections = {}
        
    def _initialize_track(self, s: float, d: float, 
                         x: float, y: float, vs_init: float = 0.0, vd_init: float = 0.0) -> FrenetTrackedObject:
        state = np.array([s, vs_init, d, vd_init])
        
        covariance = np.diag([0.1, 2.0, 0.1, 2.0])
        
        track = FrenetTrackedObject(
            id=self.next_id,
            s=s, vs=vs_init, d=d, vd=vd_init,
            x=x, y=y,
            state=state,
            covariance=covariance,
            last_seen=time.time(),
            age=0,
            position_history=[(s, d, time.time())]
        )
        
        self.next_id += 1
        return track
        
    def _is_in_los(self, track: FrenetTrackedObject, ego_s: float) -> bool:
        s_diff = abs(track.s - ego_s)
        if self.is_circular_track:
            s_diff = min(s_diff, self.track_length - s_diff)
        return s_diff < self.los_distance
        
    def update(self, detections_frenet: list, ego_s: float = 0.0,
               frenet_converter = None) -> list:
        current_time = time.time()
        
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            
            state_pred, P_pred = self.ekf.predict(track.state, track.covariance)
            
            track.state = state_pred
            track.covariance = P_pred
            track.s = state_pred[0]
            track.vs = state_pred[1]
            track.d = state_pred[2]
            track.vd = state_pred[3]
            
            if frenet_converter is not None:
                track.x, track.y = frenet_converter.frenet_to_cartesian(track.s, track.d)
                
            track.age += 1
            
        associations = self._associate_detections(detections_frenet)
        
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            s, d, x, y = detections_frenet[det_idx]
            
            measurement = np.array([s, d])
            
            state_new, P_new = self.ekf.update(track.state, track.covariance, measurement)
            
            track.state = state_new
            track.covariance = P_new
            track.s = state_new[0]
            track.vs = state_new[1]
            track.d = state_new[2]
            track.vd = state_new[3]
            track.x = x
            track.y = y
            track.last_seen = current_time
            track.age = 0
            
            if frenet_converter is not None:
                track.vx, track.vy = frenet_converter.frenet_velocity_to_cartesian(
                    track.s, track.d, track.vs, track.vd
                )
            
            track.position_history.append((s, d, current_time))
            if len(track.position_history) > 20:
                track.position_history.pop(0)
                
            track.is_static = self.classifier.classify(track.position_history)
            
        unassociated = [i for i in range(len(detections_frenet)) 
                       if i not in associations.values()]
        for det_idx in unassociated:
            s, d, x, y = detections_frenet[det_idx]
            
            vs_init, vd_init = 0.0, 0.0
            best_match_dist = float('inf')
            
            for prev_key, (prev_s, prev_d, prev_time) in self.previous_detections.items():
                dist = np.sqrt((s - prev_s)**2 + (d - prev_d)**2)
                dt = current_time - prev_time
                
                if dist < 1.0 and dt > 0.01 and dt < 0.5 and dist < best_match_dist:
                    best_match_dist = dist
                    vs_init = (s - prev_s) / dt
                    vd_init = (d - prev_d) / dt
            
            new_track = self._initialize_track(s, d, x, y, vs_init, vd_init)
            self.tracks[new_track.id] = new_track
            
            self.previous_detections[f"{s:.2f}_{d:.2f}"] = (s, d, current_time)
        
        self.previous_detections = {
            k: v for k, v in self.previous_detections.items() 
            if current_time - v[2] < 1.0
        }
            
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]
                
        return list(self.tracks.values())
        
    def _associate_detections(self, detections_frenet: list) -> dict:
        if not self.tracks or not detections_frenet:
            return {}
            
        track_ids = list(self.tracks.keys())
        track_positions = np.array([[self.tracks[tid].s, self.tracks[tid].d] 
                                    for tid in track_ids])
        detection_positions = np.array([[det[0], det[1]] for det in detections_frenet])
        
        cost_matrix = np.zeros((len(track_ids), len(detections_frenet)))
        for i, track_pos in enumerate(track_positions):
            for j, det_pos in enumerate(detection_positions):
                s_diff = abs(track_pos[0] - det_pos[0])
                if self.is_circular_track:
                    s_diff = min(s_diff, self.track_length - s_diff)
                
                d_diff = abs(track_pos[1] - det_pos[1])
                cost_matrix[i, j] = np.sqrt(s_diff**2 + d_diff**2)
                
        associations = {}
        
        for _ in range(min(len(track_ids), len(detections_frenet))):
            if cost_matrix.size == 0:
                break
                
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            track_idx, det_idx = min_idx
            
            if cost_matrix[track_idx, det_idx] < self.max_distance:
                associations[track_ids[track_idx]] = det_idx
                cost_matrix[track_idx, :] = np.inf
                cost_matrix[:, det_idx] = np.inf
            else:
                break
                
        return associations
        
    def get_active_tracks(self) -> list:
        return [track for track in self.tracks.values() if track.age < 2]
        
    def get_dynamic_tracks(self) -> list:
        return [track for track in self.tracks.values() 
                if not track.is_static and track.age < 2]
    
    def update_track_length(self, track_length: float):
        self.track_length = track_length
        self.ekf.track_length = track_length
