import numpy as np
from typing import Dict, List, Tuple, Optional, Any
try:
    from scipy import signal
    from scipy.stats import entropy
    from scipy.spatial.distance import cdist
except ImportError:
    pass
import warnings

def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if float(np.ptp(x)) < 1e-06 or float(np.ptp(y)) < 1e-06:
        return 1.0 - min(1.0, abs(float(np.mean(x)) - float(np.mean(y))))
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)

class SymbolicRewardCalculator:

    def __init__(self, config):
        self.config = config
        self.weights = config.get_symbolic_weights()

    def compute_beat_alignment(self, g_vis: np.ndarray, g_sym: np.ndarray, delta_max: Optional[float]=None) -> float:
        if delta_max is None:
            delta_max = self.config.delta_max

        def ncc(x, y, shift):
            if shift == 0:
                x_shifted = x
            elif shift > 0:
                x_shifted = np.pad(x, (shift, 0), mode='constant')[:len(y)]
            else:
                x_shifted = np.pad(x, (0, -shift), mode='constant')[-len(y):]
            if len(x_shifted) != len(y):
                min_len = min(len(x_shifted), len(y))
                x_shifted = x_shifted[:min_len]
                y = y[:min_len]
            x_std = float(x_shifted.std())
            y_std = float(y.std())
            if x_std < 1e-08 and y_std < 1e-08:
                closeness = 1.0 - min(1.0, abs(float(x_shifted.mean()) - float(y.mean())))
                return 2.0 * closeness - 1.0
            if x_std < 1e-08 or y_std < 1e-08:
                return -1.0
            x_norm = (x_shifted - x_shifted.mean()) / x_std
            y_norm = (y - y.mean()) / y_std
            return np.dot(x_norm, y_norm) / len(x_norm)
        max_shift = int(delta_max * len(g_vis))
        shifts = np.arange(-max_shift, max_shift + 1)
        ncc_values = [ncc(g_vis, g_sym, s) for s in shifts]
        r_beat = max(ncc_values) if ncc_values else 0.0
        return np.clip((r_beat + 1) / 2, 0, 1)

    def compute_phrase_consistency(self, B_vis: List[int], B_sym: List[int]) -> float:
        if not B_vis and (not B_sym):
            return 1.0
        if not B_vis or not B_sym:
            return 0.0
        set_vis = set(B_vis)
        set_sym = set(B_sym)
        if not set_vis or not set_sym:
            return 0.0
        tolerance = 2
        matches = 0
        for b_vis in set_vis:
            for b_sym in set_sym:
                if abs(b_vis - b_sym) <= tolerance:
                    matches += 1
                    break
        precision = matches / len(set_vis) if set_vis else 0.0
        recall = matches / len(set_sym) if set_sym else 0.0
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def compute_key_stability(self, key_posterior: np.ndarray) -> float:
        if key_posterior.size == 0:
            return 0.0
        try:
            from scipy.stats import entropy as scipy_entropy
            entropy_func = scipy_entropy
        except ImportError:

            def entropy_func(p):
                p = p / (p.sum() + 1e-08)
                p = p[p > 0]
                return -np.sum(p * np.log(p + 1e-08))
        if key_posterior.ndim == 1:
            key_posterior = key_posterior.reshape(1, -1)
        entropies = [entropy_func(p) for p in key_posterior]
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(key_posterior.shape[1]) if key_posterior.ndim > 1 else np.log(len(key_posterior))
        r_key = 1.0 - mean_entropy / (max_entropy + 1e-08)
        return np.clip(r_key, 0, 1)

    def compute_note_diversity(self, pitches: np.ndarray, durations: np.ndarray) -> float:

        def compute_entropy_normalized(values):
            if len(values) == 0:
                return 0.0
            unique, counts = np.unique(values, return_counts=True)
            probs = counts / len(values)
            try:
                from scipy.stats import entropy as scipy_entropy
                h = scipy_entropy(probs)
            except ImportError:
                probs = probs[probs > 0]
                h = -np.sum(probs * np.log(probs + 1e-08))
            max_h = np.log(len(unique)) if len(unique) > 1 else 1.0
            return h / (max_h + 1e-08) if max_h > 0 else 0.0
        h_pitch = compute_entropy_normalized(pitches)
        h_duration = compute_entropy_normalized(durations)
        r_div = (h_pitch + h_duration) / 2.0
        return np.clip(r_div, 0, 1)

    def compute_playability(self, violations: int, total_notes: int) -> float:
        if total_notes == 0:
            return 0.0
        violation_ratio = violations / total_notes
        r_play = 1.0 - violation_ratio
        return np.clip(r_play, 0, 1)

    def compute_symbolic_reward(self, beat_data: Optional[Dict]=None, phrase_data: Optional[Dict]=None, key_data: Optional[Dict]=None, diversity_data: Optional[Dict]=None, playability_data: Optional[Dict]=None) -> float:
        rewards = {}
        if beat_data:
            rewards['beat'] = self.compute_beat_alignment(beat_data['g_vis'], beat_data['g_sym'])
        else:
            rewards['beat'] = 0.0
        if phrase_data:
            rewards['phrase'] = self.compute_phrase_consistency(phrase_data['B_vis'], phrase_data['B_sym'])
        else:
            rewards['phrase'] = 0.0
        if key_data:
            rewards['key'] = self.compute_key_stability(key_data['key_posterior'])
        else:
            rewards['key'] = 0.0
        if diversity_data:
            rewards['div'] = self.compute_note_diversity(diversity_data['pitches'], diversity_data['durations'])
        else:
            rewards['div'] = 0.0
        if playability_data:
            rewards['play'] = self.compute_playability(playability_data['violations'], playability_data['total_notes'])
        else:
            rewards['play'] = 0.0
        R_sym = sum((self.weights[k] * rewards[k] for k in self.weights.keys()))
        return np.clip(R_sym, 0, 1)

class SymbolicAudioConsistencyReward:

    def __init__(self, config):
        self.config = config
        self.weights = config.get_consistency_weights()

    def compute_onset_alignment(self, e_sym: np.ndarray, e_aud: np.ndarray) -> float:

        def normalized_dtw(x, y):
            n, m = (len(x), len(y))
            if n == 0 or m == 0:
                return 1.0
            x_norm = (x - x.mean()) / (x.std() + 1e-08)
            y_norm = (y - y.mean()) / (y.std() + 1e-08)
            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(x_norm[i - 1] - y_norm[j - 1])
                    dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            normalized_dist = dtw[n, m] / (n + m)
            return normalized_dist
        norm_dtw = normalized_dtw(e_sym, e_aud)
        r_onset = 1.0 - np.clip(norm_dtw, 0, 1)
        return r_onset

    def compute_dynamic_correlation(self, rms_aud: np.ndarray, vel_sym: np.ndarray) -> float:
        if len(rms_aud) != len(vel_sym) or len(rms_aud) == 0:
            return 0.0
        corr = _safe_corrcoef(rms_aud, vel_sym)
        r_dyn = max(0.0, corr)
        return r_dyn

    def compute_color_correlation(self, C_spec: np.ndarray, w: np.ndarray, hf_ratio: np.ndarray, b: np.ndarray) -> float:

        def positive_correlation(x, y):
            if len(x) != len(y) or len(x) == 0:
                return 0.0
            corr = _safe_corrcoef(x, y)
            return max(0.0, corr)
        rho_spec = positive_correlation(C_spec, w)
        rho_hf = positive_correlation(hf_ratio, b)
        r_color = 0.5 * (rho_spec + rho_hf)
        return np.clip(r_color, 0, 1)

    def compute_consistency_reward(self, onset_data: Optional[Dict]=None, dynamic_data: Optional[Dict]=None, color_data: Optional[Dict]=None) -> float:
        rewards = {}
        if onset_data:
            rewards['onset'] = self.compute_onset_alignment(onset_data['e_sym'], onset_data['e_aud'])
        else:
            rewards['onset'] = 0.0
        if dynamic_data:
            rewards['dyn'] = self.compute_dynamic_correlation(dynamic_data['rms_aud'], dynamic_data['vel_sym'])
        else:
            rewards['dyn'] = 0.0
        if color_data:
            rewards['color'] = self.compute_color_correlation(color_data['C_spec'], color_data['w'], color_data['hf_ratio'], color_data['b'])
        else:
            rewards['color'] = 0.0
        R_sym_aud = sum((self.weights[k] * rewards[k] for k in self.weights.keys()))
        return np.clip(R_sym_aud, 0, 1)

class MultiSourceRewardAggregator:

    def __init__(self, config):
        self.config = config
        self.alpha_vis = np.array(config.alpha_vis)
        self.beta_mus = np.array(config.beta_mus)
        self.gamma_cons = config.gamma_cons if config.consistency_enabled else 0.0
        self._normalize_weights()

    def _normalize_weights(self):
        total = self.alpha_vis.sum() + self.beta_mus.sum() + self.gamma_cons
        if total > 0:
            self.alpha_vis = self.alpha_vis / total
            self.beta_mus = self.beta_mus / total
            self.gamma_cons = self.gamma_cons / total

    def update_weights(self, alpha_vis: np.ndarray, beta_mus: np.ndarray, gamma_cons: float):
        self.alpha_vis = np.array(alpha_vis)
        self.beta_mus = np.array(beta_mus)
        self.gamma_cons = gamma_cons if self.config.consistency_enabled else 0.0
        self._normalize_weights()

    def compute_visual_rewards(self, motion_rhythm: float, color_tone: float, novelty_thematic: float) -> float:
        rewards = np.array([motion_rhythm, color_tone, novelty_thematic])
        group_weight = float(self.alpha_vis.sum())
        if group_weight <= 0:
            return 0.0
        R_vis = np.dot(self.alpha_vis, rewards) / group_weight
        return np.clip(R_vis, 0, 1)

    def compute_musicality_rewards(self, tonal_stability: float, harmonic_progression: float, rhythmic_density: float, structural_symmetry: float) -> float:
        rewards = np.array([tonal_stability, harmonic_progression, rhythmic_density, structural_symmetry])
        group_weight = float(self.beta_mus.sum())
        if group_weight <= 0:
            return 0.0
        R_mus = np.dot(self.beta_mus, rewards) / group_weight
        return np.clip(R_mus, 0, 1)

    def compute_consistency_reward_kl(self, p_y: np.ndarray, q_y: np.ndarray) -> float:
        p_y = p_y / (p_y.sum() + 1e-08)
        q_y = q_y / (q_y.sum() + 1e-08)
        try:
            from scipy.stats import entropy as scipy_entropy
            kl = scipy_entropy(p_y, q_y)
        except ImportError:
            kl = np.sum(p_y * np.log((p_y + 1e-08) / (q_y + 1e-08)))
        max_kl = 10.0
        R_cons = 1.0 - np.clip(kl / max_kl, 0, 1)
        return R_cons

    def aggregate_rewards(self, R_vis: float, R_mus: float, R_cons: float) -> float:
        R_total = self.alpha_vis.sum() * R_vis + self.beta_mus.sum() * R_mus + self.gamma_cons * R_cons
        return np.clip(R_total, 0, 1)
