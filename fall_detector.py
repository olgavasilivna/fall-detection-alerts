import math
from typing import Tuple, Dict, List, Optional
import numpy as np
from collections import deque

class FallDetector:

    default_config = {
        # Basic thresholds
        'min_confidence': 0.5,  # Lower confidence requirement
        'min_consecutive_frames': 1,  # Single frame detection
        'window_size': 5,  # Smaller window for faster response

        # Ratio analysis
        'ratio_change_threshold': 0.2,
        'min_ratio_samples': 3,
        'max_ratio': 2.0,

        # Secondary thresholds
        'min_torso_angle': 80,
        'min_fall_score': 0.3,
        'min_velocity': 0.05,

        # Weights for scoring
        'confidence_weight': 0.1,
        'velocity_weight': 0.1,
        'angle_weight': 0.2,
        'position_weight': 0.2,
        'ratio_weight': 0.8,

        # Debug settings
        'debug': True,

        # Recovery detection settings
        'recovery_window_size': 20,
        'min_recovery_frames': 15,
        'recovery_angle_threshold': 30,
        'recovery_velocity_threshold': 0.01,
        'recovery_position_threshold': 0.3,
        'recovery_ratio_threshold': 0.2,
        'recovery_timeout': 300,
        'ema_alpha': 0.2,
        'min_baseline_samples': 30,
        'baseline_update_rate': 0.1,
        'epsilon': 1e-6,
        'baseline_history_length': 100,

        # Added parameters
        'confirmation_frames': 2,
        'no_motion_timeout': 5,

        # Alert settings
        'alert_cooldown_frames': 30,  # Number of frames to wait before allowing another alert
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fall detector with configurable parameters.

        Args:
            config: Optional dictionary of configuration parameters. If None, use default_config.
        """
        # Use default configuration if no custom config is provided
        if config is None:
            self.config = self.default_config
        else:
            self.config = self.default_config.copy()  # Start with default
            self.config.update(config)  # Override with custom config

        # State tracking
        self.fall_frame_counter = 0
        self.last_fall_state = False
        self.previous_center_y = None
        self.no_condition_frames = 0
        self.alert_sent = False  # Track if alert has been sent for current fall
        self.recovery_alert_sent = False  # Track if recovery alert has been sent
        self.alert_cooldown_counter = 0  # Counter for alert cooldown

        # Recovery state tracking
        self.is_fallen = False
        self.recovery_frame_counter = 0
        self.recovery_timeout_counter = 0
        self.recovery_metrics_history = deque(maxlen=self.config['recovery_window_size'])

        # Hysteresis counters for recovery
        self.recovery_angle_frames = 0
        self.recovery_velocity_frames = 0
        self.recovery_position_frames = 0
        self.recovery_ratio_frames = 0

        # Baseline calibration state
        self.is_calibrating = True
        self.baseline_samples = []

        # Sliding windows for smoothing
        self.score_history = deque(maxlen=self.config['window_size'])
        self.angle_history = deque(maxlen=self.config['window_size'])
        self.velocity_history = deque(maxlen=self.config['window_size'])
        self.ratio_history = deque(maxlen=self.config['window_size'])
        self.baseline_ratio_history = deque(maxlen=self.config['baseline_history_length'])

        # Hysteresis tracking
        self.angle_above_threshold_frames = 0
        self.velocity_above_threshold_frames = 0
        self.suspicious_ratio_frames = 0

        # Added: Confirmation tracking
        self.confirmation_counter = 0
        self.is_actually_fallen = False
        self.no_motion_counter = 0

    def normalize_velocity(self, velocity: float) -> float:
        """Normalize velocity with sigmoid function to reduce sensitivity to small movements."""
        k = 100  # Increased from 50 for steeper sigmoid
        x0 = self.config['min_velocity']  # Center of sigmoid
        return 1 / (1 + np.exp(-k * (abs(velocity) - x0)))

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle with sigmoid function."""
        k = 0.2  # Increased from 0.1 for steeper sigmoid
        x0 = self.config['min_torso_angle']  # Center of sigmoid
        return 1 / (1 + np.exp(-k * (angle - x0)))

    def normalize_position(self, head_y: float, hip_y: float) -> float:
        """Normalize head-hip position difference."""
        diff = hip_y - head_y
        return min(max(diff / 0.2, 0), 1)  # Normalize to 0-1 range

    def normalize_ratio(self, ratio: float) -> float:
        """Normalize ratio based on baseline average."""
        if not self.baseline_ratio_history:
            return 0.0

        baseline_avg = np.mean(self.baseline_ratio_history)
        if baseline_avg == 0:
            return 0.0

        # Limit ratio to a max value for normalization
        ratio = min(ratio, self.config['max_ratio'])

        # Normalize based on deviation from baseline
        deviation = abs(ratio - baseline_avg) / (baseline_avg+self.config['epsilon'])
        # Use sigmoid to smooth the transition
        k = 10.0  # Increased from 5.0 for steeper sigmoid
        x0 = self.config['ratio_change_threshold']  # Center of sigmoid
        return 1 / (1 + np.exp(-k * (deviation - x0)))

    def calculate_fall_score(self, metrics: Dict) -> float:
        """
        Calculate a weighted fall score based on multiple indicators.

        Args:
            metrics: Dictionary containing current frame metrics

        Returns:
            float: Fall score between 0 and 1
        """
        # Normalize all metrics
        velocity_score = self.normalize_velocity(metrics['vertical_velocity'])
        angle_score = self.normalize_angle(metrics['torso_angle'])
        position_score = self.normalize_position(metrics['head_y'], metrics['hip_y'])
        ratio_score = self.normalize_ratio(metrics['current_ratio'])

        # Debug output for normalized scores
        if self.config['debug']:
            print(f"\nNormalized Scores:")
            print(f"  Velocity: {velocity_score:.3f}")
            print(f"  Angle: {angle_score:.3f}")
            print(f"  Position: {position_score:.3f}")
            print(f"  Ratio: {ratio_score:.3f}")

        # Calculate weighted score with confidence as a multiplier
        base_score = (
            velocity_score * self.config['velocity_weight'] +
            angle_score * self.config['angle_weight'] +
            position_score * self.config['position_weight'] +
            ratio_score * self.config['ratio_weight']
        )

        # Confidence acts as a multiplier to the base score
        confidence_factor = metrics['confidence'] * self.config['confidence_weight']
        total_score = base_score * (1 + confidence_factor)

        return min(total_score, 1.0)  # Cap at 1.0

    def is_ratio_suspicious(self, current_ratio: float) -> bool:
        """Check if current ratio is suspiciously different from baseline."""
        if len(self.baseline_ratio_history) < self.config['min_ratio_samples']:
            # Not enough samples yet, add to baseline
            self.baseline_ratio_history.append(current_ratio)
            return False

        baseline_avg = np.mean(self.baseline_ratio_history)
        ratio_change = abs(current_ratio - baseline_avg) / baseline_avg

        # Only update baseline if ratio is not suspicious, not calibrating, and not falling
        if ratio_change < self.config['ratio_change_threshold'] and not self.is_fallen and not self.is_calibrating:
            self.baseline_ratio_history.append(current_ratio)

        return ratio_change >= self.config['ratio_change_threshold']

    def calculate_ema(self, data: List[float], alpha: float) -> List[float]:
        """
        Calculate Exponential Moving Average for a list of values.

        Args:
            data: List of values to calculate EMA for
            alpha: Smoothing factor (0 < alpha < 1)

        Returns:
            List of EMA values
        """
        if not data:
            return []  # Handle empty list

        ema = [data[0]]  # Initialize with the first data point
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        return ema

    def update_baseline_ratio(self, current_ratio: float) -> None:
        """
        Update the baseline ratio using exponential smoothing.
        This is a simplified version of a Kalman filter approach, where we use
        exponential smoothing to gradually adapt the baseline to new measurements.

        Args:
            current_ratio: Current ratio value to incorporate into baseline
        """
        if self.is_calibrating:
            self.baseline_samples.append(current_ratio)
            if len(self.baseline_samples) >= self.config['min_baseline_samples']:
                self.is_calibrating = False
                self.baseline_ratio_history.extend(self.baseline_samples)
                self.baseline_samples.clear()
                if self.config['debug']:
                    print("Baseline calibration complete. Starting normal operation.")
        else:
            # Only update baseline if we're not in a fall state
            if not self.is_fallen and self.baseline_ratio_history:
                current_baseline = np.mean(self.baseline_ratio_history)
                updated_baseline = (1 - self.config['baseline_update_rate']) * current_baseline + \
                                 self.config['baseline_update_rate'] * current_ratio
                self.baseline_ratio_history.append(updated_baseline)

                if self.config['debug']:
                    print(f"Baseline updated: {current_baseline:.3f} -> {updated_baseline:.3f}")

    def is_recovery_condition_met(self, metrics: Dict) -> bool:
        """
        Determine if current frame indicates recovery from a fall (with hysteresis).

        Args:
            metrics: Dictionary containing current frame metrics

        Returns:
            bool: True if recovery conditions are met
        """
        if not self.is_fallen:
            return False

        # Check basic confidence threshold
        if metrics['confidence'] < self.config['min_confidence']:
            return False

        # Calculate recovery metrics
        torso_angle_ok = metrics['torso_angle'] < self.config['recovery_angle_threshold']
        velocity_ok = abs(metrics['vertical_velocity']) < self.config['recovery_velocity_threshold']
        position_ok = self.normalize_position(metrics['head_y'], metrics['hip_y']) < self.config['recovery_position_threshold']

        # Check ratio against baseline with epsilon to prevent division by zero
        baseline_avg = np.mean(self.baseline_ratio_history) if self.baseline_ratio_history else 0
        ratio_change = abs(metrics['current_ratio'] - baseline_avg) / (baseline_avg + self.config['epsilon']) if baseline_avg != 0 else 0
        ratio_ok = ratio_change < self.config['recovery_ratio_threshold']

        # Store current metrics for history
        self.recovery_metrics_history.append({
            'torso_angle': metrics['torso_angle'],
            'velocity': abs(metrics['vertical_velocity']),
            'position': self.normalize_position(metrics['head_y'], metrics['hip_y']),
            'ratio': ratio_change
        })

        # Update hysteresis counters
        if torso_angle_ok:
            self.recovery_angle_frames = min(self.recovery_angle_frames + 1, self.config['min_recovery_frames'])
        else:
            self.recovery_angle_frames = max(0, self.recovery_angle_frames - 2)  # More aggressive decrement

        if velocity_ok:
            self.recovery_velocity_frames = min(self.recovery_velocity_frames + 1, self.config['min_recovery_frames'])
        else:
            self.recovery_velocity_frames = max(0, self.recovery_velocity_frames - 2)

        if position_ok:
            self.recovery_position_frames = min(self.recovery_position_frames + 1, self.config['min_recovery_frames'])
        else:
            self.recovery_position_frames = max(0, self.recovery_position_frames - 2)

        if ratio_ok:
            self.recovery_ratio_frames = min(self.recovery_ratio_frames + 1, self.config['min_recovery_frames'])
        else:
            self.recovery_ratio_frames = max(0, self.recovery_ratio_frames - 2)

        # Calculate EMA for each metric
        if len(self.recovery_metrics_history) >= self.config['min_recovery_frames']:
            angle_values = [m['torso_angle'] for m in self.recovery_metrics_history]
            velocity_values = [m['velocity'] for m in self.recovery_metrics_history]
            position_values = [m['position'] for m in self.recovery_metrics_history]
            ratio_values = [m['ratio'] for m in self.recovery_metrics_history]

            ema_angles = self.calculate_ema(angle_values, self.config['ema_alpha'])
            ema_velocities = self.calculate_ema(velocity_values, self.config['ema_alpha'])
            ema_positions = self.calculate_ema(position_values, self.config['ema_positions'])
            ema_ratios = self.calculate_ema(ratio_values, self.config['ema_ratios'])

            # Check if all metrics have met the hysteresis requirement
            consistent_recovery = (
                self.recovery_angle_frames >= self.config['min_recovery_frames'] and
                self.recovery_velocity_frames >= self.config['min_recovery_frames'] and
                self.recovery_position_frames >= self.config['min_recovery_frames'] and
                self.recovery_ratio_frames >= self.config['min_recovery_frames']
            )

            if consistent_recovery:
                self.recovery_frame_counter += 1
            else:
                self.recovery_frame_counter = 0  # Reset if not all conditions are met

            # Debug output
            if self.config['debug']:
                print(f"\nRecovery Analysis:")
                print(f"Torso Angle: {metrics['torso_angle']:.1f}° (EMA: {ema_angles[-1]:.1f}°)")
                print(f"Velocity: {abs(metrics['vertical_velocity']):.4f} (EMA: {ema_velocities[-1]:.4f})")
                print(f"Position: {self.normalize_position(metrics['head_y'], metrics['hip_y']):.3f} (EMA: {ema_positions[-1]:.3f})")
                print(f"Ratio Change: {ratio_change:.3f} (EMA: {ema_ratios[-1]:.3f})")
                print(f"Hysteresis Counters:")
                print(f"  Angle: {self.recovery_angle_frames}/{self.config['min_recovery_frames']}")
                print(f"  Velocity: {self.recovery_velocity_frames}/{self.config['min_recovery_frames']}")
                print(f"  Position: {self.recovery_position_frames}/{self.config['min_recovery_frames']}")
                print(f"  Ratio: {self.recovery_ratio_frames}/{self.config['min_recovery_frames']}")
                print(f"Recovery Progress: {self.recovery_frame_counter}/{self.config['min_recovery_frames']}")

            if self.recovery_frame_counter >= self.config['min_recovery_frames']:
                self.is_fallen = False
                self.recovery_frame_counter = 0
                self.recovery_metrics_history.clear()

                # Reset hysteresis counters
                self.recovery_angle_frames = 0
                self.recovery_velocity_frames = 0
                self.recovery_position_frames = 0
                self.recovery_ratio_frames = 0

                return True

        # Increment timeout counter
        self.recovery_timeout_counter += 1
        if self.recovery_timeout_counter >= self.config['recovery_timeout']:
            # Reset recovery state if timeout reached
            self.is_fallen = False
            self.recovery_frame_counter = 0
            self.recovery_metrics_history.clear()
            self.recovery_timeout_counter = 0

            # Reset hysteresis counters
            self.recovery_angle_frames = 0
            self.recovery_velocity_frames = 0
            self.recovery_position_frames = 0
            self.recovery_ratio_frames = 0

        return False

    def is_fall_condition_met(self, metrics: Dict) -> Tuple[bool, float]:
        """Determine if current frame indicates a fall."""
        # Check basic confidence threshold
        if metrics['confidence'] < self.config['min_confidence']:
            return False, metrics['confidence']

        # Track previous state
        previous_fall_state = self.is_fallen

        # Update sliding windows
        self.score_history.append(metrics.get('fall_score', 0))
        self.angle_history.append(metrics['torso_angle'])
        self.velocity_history.append(abs(metrics['vertical_velocity']))
        self.ratio_history.append(metrics['current_ratio'])

        # Calculate smoothed metrics
        avg_ratio = np.mean(self.ratio_history) if self.ratio_history else 0

        # Simple ratio check - fall when ratio is below 2.0
        ratio_condition = avg_ratio < 2.0

        # Debug output
        if self.config['debug']:
            print(f"\nFall Detection Analysis:")
            print(f"Current Ratio: {avg_ratio:.2f}")
            print(f"Ratio Threshold: 2.0")
            print(f"Ratio Condition: {ratio_condition}")
            print(f"Raw Ratio: {metrics['current_ratio']:.2f}")

        # Primary fall detection based on ratio
        if ratio_condition:
            self.fall_frame_counter += 1
            if self.config['debug']:
                print(f"Ratio condition met, incrementing counter: {self.fall_frame_counter}")
            
            if self.fall_frame_counter >= self.config['min_consecutive_frames']:
                self.is_fallen = True
                # Send alert on state change
                if not previous_fall_state:
                    if self.config['debug']:
                        print("ALERT: Fall detected!")
                    self.alert_sent = True
                    self.recovery_alert_sent = False
        else:
            self.fall_frame_counter = max(0, self.fall_frame_counter - 1)
            if self.fall_frame_counter == 0 and self.is_fallen:
                self.is_fallen = False
                # Send recovery alert on state change
                if previous_fall_state:
                    if self.config['debug']:
                        print("ALERT: Person has recovered from fall!")
                    self.recovery_alert_sent = True
                    self.alert_sent = False

        # Set final fall state
        is_fall = self.is_fallen

        # Debug output
        if self.config['debug']:
            print(f"\nFinal State:")
            print(f"Fall Frame Counter: {self.fall_frame_counter}")
            print(f"Is Fallen: {self.is_fallen}")
            print(f"Final Fall State: {is_fall}")
            print(f"Alert Sent: {self.alert_sent}")
            print(f"Recovery Alert Sent: {self.recovery_alert_sent}")

        return is_fall, metrics['confidence']

    def get_score_history(self) -> List[float]:
        """
        Get the history of fall scores.

        Returns:
            List of recent fall scores
        """
        return list(self.score_history)

    def reset(self):
        """Reset all state variables."""
        self.fall_frame_counter = 0
        self.last_fall_state = False
        self.previous_center_y = None
        self.no_condition_frames = 0
        self.is_fallen = False
        self.recovery_frame_counter = 0
        self.recovery_timeout_counter = 0
        self.recovery_metrics_history.clear()
        self.score_history.clear()
        self.angle_history.clear()
        self.velocity_history.clear()
        self.ratio_history.clear()
        self.angle_above_threshold_frames = 0
        self.velocity_above_threshold_frames = 0
        self.suspicious_ratio_frames = 0
        self.baseline_ratio_history.clear()

        # Reset alert flags and cooldown
        self.alert_sent = False
        self.recovery_alert_sent = False
        self.alert_cooldown_counter = 0

        # Reset recovery hysteresis counters
        self.recovery_angle_frames = 0
        self.recovery_velocity_frames = 0
        self.recovery_position_frames = 0
        self.recovery_ratio_frames = 0

        # Reset calibration state
        self.is_calibrating = True
        self.baseline_samples.clear()

        # Reset confirmation tracking
        self.confirmation_counter = 0
        self.is_actually_fallen = False
        self.no_motion_counter = 0