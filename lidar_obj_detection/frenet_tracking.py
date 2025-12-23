import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class OpponentState:
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
    position_buffer: List[Tuple[float, float]] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0


class OpponentEKF:
    def __init__(self, dt: float = 0.1, track_length: float = 100.0):
        self.dt = dt
        self.track_length = track_length
        
        self.P_vs = 0.2
        self.P_d = 0.02
        self.P_vd = 0.2
        
        self.F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.B = np.array([
            [0.0, 0.0, 0.0],
            [self.P_vs, 0.0, 0.0],
            [0.0, self.P_d, 0.0],
            [0.0, 0.0, self.P_vd]
        ])
        
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        Q1 = np.array([
            [1.95e-7, 1.56e-5],
            [1.56e-5, 1.25e-3]
        ])
        Q2 = np.array([
            [7.81e-7, 6.25e-5],
            [6.25e-5, 5e-3]
        ])
        self.Q = np.block([
            [Q1, np.zeros((2, 2))],
            [np.zeros((2, 2)), Q2]
        ])
        
        self.R = np.diag([0.002, 0.2, 0.002, 0.2])
        
    def predict(self, state: np.ndarray, P: np.ndarray, u: np.ndarray, in_los: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        state_pred = self.F @ state + self.B @ u
        
        state_pred[0] = state_pred[0] % self.track_length
        
        P_pred = self.F @ P @ self.F.T + self.Q
        
        return state_pred, P_pred
        
    def update(self, state_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = z - self.H @ state_pred
        
        y[0] = self._normalize_s_residual(y[0])
        
        S = self.H @ P_pred @ self.H.T + self.R
        
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        state = state_pred + K @ y
        
        state[0] = state[0] % self.track_length
        
        P = (np.eye(4) - K @ self.H) @ P_pred
        
        return state, P
        
    def _normalize_s_residual(self, s_residual: float) -> float:
        if s_residual > self.track_length / 2:
            return s_residual - self.track_length
        elif s_residual < -self.track_length / 2:
            return s_residual + self.track_length
        return s_residual


class StaticDynamicClassifier:
    def __init__(self, buffer_size: int = 10, std_threshold: float = 0.17):
        self.buffer_size = buffer_size
        self.std_threshold = std_threshold
        
    def classify(self, position_buffer: List[Tuple[float, float]]) -> bool:
        if len(position_buffer) < self.buffer_size:
            return False
            
        recent_positions = np.array(position_buffer[-self.buffer_size:])
        
        std_s = np.std(recent_positions[:, 0])
        std_d = np.std(recent_positions[:, 1])
        
        total_std = np.sqrt(std_s**2 + std_d**2)
        
        return total_std < self.std_threshold


class OpponentTracker:
    def __init__(self, dt: float = 0.1, track_length: float = 100.0, 
                 max_association_dist: float = 2.0, max_age: int = 5,
                 v_target_ratio: float = 0.6):
        self.dt = dt
        self.track_length = track_length
        self.max_association_dist = max_association_dist
        self.max_age = max_age
        self.v_target_ratio = v_target_ratio
        
        self.ekf = OpponentEKF(dt=dt, track_length=track_length)
        self.classifier = StaticDynamicClassifier()
        
        self.tracks = {}
        self.next_id = 0
        
    def _initialize_track(self, s: float, d: float, vs: float, vd: float, 
                         x: float, y: float) -> OpponentState:
        state = np.array([s, vs, d, vd])
        
        P = np.diag([0.1, 2.0, 0.1, 2.0])
        
        track = OpponentState(
            id=self.next_id,
            s=s, vs=vs, d=d, vd=vd,
            x=x, y=y,
            state=state,
            covariance=P,
            last_seen=time.time(),
            age=0,
            position_buffer=[(s, d)]
        )
        
        self.next_id += 1
        return track
        
    def update(self, detections: List[Tuple[float, float, float, float, float, float]], 
               ego_s: float = 0.0, ego_vs: float = 5.0, 
               v_target_profile: Optional[np.ndarray] = None,
               frenet_converter = None) -> List[OpponentState]:
        
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            
            in_los = len(detections) > 0
            
            if in_los:
                u = np.array([0.0, -track.d, -track.vd])
            else:
                if v_target_profile is not None:
                    s_idx = int(track.s) % len(v_target_profile)
                    v_target = v_target_profile[s_idx] * self.v_target_ratio
                else:
                    v_target = ego_vs * self.v_target_ratio
                    
                u = np.array([(v_target - track.vs), -track.d, -track.vd])
            
            state_pred, P_pred = self.ekf.predict(track.state, track.covariance, u, in_los)
            
            track.state = state_pred
            track.covariance = P_pred
            track.s = state_pred[0]
            track.vs = state_pred[1]
            track.d = state_pred[2]
            track.vd = state_pred[3]
            
            if frenet_converter is not None:
                track.x, track.y = frenet_converter.frenet_to_cartesian(track.s, track.d)
                
            track.age += 1
            
        associations = self._associate_detections(detections)
        
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            s, d, vs, vd, x, y = detections[det_idx]
            
            z = np.array([s, vs, d, vd])
            
            state_new, P_new = self.ekf.update(track.state, track.covariance, z)
            
            track.state = state_new
            track.covariance = P_new
            track.s = state_new[0]
            track.vs = state_new[1]
            track.d = state_new[2]
            track.vd = state_new[3]
            track.x = x
            track.y = y
            track.last_seen = time.time()
            track.age = 0
            
            track.position_buffer.append((s, d))
            if len(track.position_buffer) > 20:
                track.position_buffer.pop(0)
                
            track.is_static = self.classifier.classify(track.position_buffer)
            
        unassociated_detections = [i for i in range(len(detections)) 
                                   if i not in associations.values()]
        
        for det_idx in unassociated_detections:
            s, d, vs, vd, x, y = detections[det_idx]
            new_track = self._initialize_track(s, d, vs, vd, x, y)
            self.tracks[new_track.id] = new_track
            
        to_remove = [tid for tid, track in self.tracks.items() 
                    if track.age > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
                
        return list(self.tracks.values())
        
    def _associate_detections(self, detections: List[Tuple[float, float, float, float, float, float]]) -> dict:
        if not self.tracks or not detections:
            return {}
            
        track_ids = list(self.tracks.keys())
        
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                s_det, d_det = det[0], det[1]
                
                s_diff = abs(track.s - s_det)
                s_diff = min(s_diff, self.track_length - s_diff)
                
                d_diff = abs(track.d - d_det)
                
                cost_matrix[i, j] = np.sqrt(s_diff**2 + d_diff**2)
                
        associations = {}
        
        for _ in range(min(len(track_ids), len(detections))):
            if cost_matrix.size == 0:
                break
                
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            track_idx, det_idx = min_idx
            
            if cost_matrix[track_idx, det_idx] < self.max_association_dist:
                associations[track_ids[track_idx]] = det_idx
                
                cost_matrix[track_idx, :] = np.inf
                cost_matrix[:, det_idx] = np.inf
            else:
                break
                
        return associations
        
    def get_dynamic_opponents(self) -> List[OpponentState]:
        return [track for track in self.tracks.values() 
                if not track.is_static and track.age < 3]
    
    def get_closest_opponent(self, ego_s: float) -> Optional[OpponentState]:
        dynamic_opponents = self.get_dynamic_opponents()
        if not dynamic_opponents:
            return None
            
        min_dist = float('inf')
        closest = None
        for opp in dynamic_opponents:
            s_diff = abs(opp.s - ego_s)
            s_diff = min(s_diff, self.track_length - s_diff)
            if s_diff < min_dist:
                min_dist = s_diff
                closest = opp
                
        return closest
