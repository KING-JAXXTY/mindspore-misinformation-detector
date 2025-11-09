"""
Video Deepfake Detection using MindSpore
Analyzes video frames for manipulation, face swaps, and synthetic content
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import threading
import traceback
import cv2
import tempfile
import os

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="CPU")
mindspore_lock = threading.Lock()

class MindSporeVideoDeepfakeDetector(nn.Cell):
    """CNN for video deepfake detection - analyzes frame patterns"""
    
    def __init__(self):
        super(MindSporeVideoDeepfakeDetector, self).__init__()
        # Input: 64 frame features (spatial + temporal)
        self.dense1 = nn.Dense(64, 128)
        self.dense2 = nn.Dense(128, 64)
        self.dense3 = nn.Dense(64, 32)
        self.dense4 = nn.Dense(32, 2)  # Real vs Fake
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return self.softmax(x)

# Global model instance
detector_model = None

def initialize_model():
    """Initialize the MindSpore deepfake detection model"""
    global detector_model
    if detector_model is None:
        with mindspore_lock:
            detector_model = MindSporeVideoDeepfakeDetector()
            # Initialize with random weights (in production, load pre-trained weights)
            dummy_input = Tensor(np.random.randn(1, 64).astype(np.float32))
            _ = detector_model(dummy_input)
    return detector_model

def extract_video_features(video_path, max_frames=30):
    """Extract advanced features from video frames for deepfake detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames evenly
        frame_indices = np.linspace(0, max(0, total_frames - 1), min(max_frames, total_frames), dtype=int)
        
        features = []
        prev_frame = None
        frame_count = 0
        current_idx = 0
        
        # Detection metrics
        temporal_inconsistencies = 0
        face_region_anomalies = 0
        compression_artifacts = []
        color_inconsistencies = []
        
        # Authentic video characteristics
        motion_blur_scores = []
        noise_grain_levels = []
        lighting_consistency = []
        lens_distortion_detected = False
        
        while cap.isOpened() and current_idx < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == frame_indices[current_idx]:
                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
                # Spatial features
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                
                # Edge analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Frequency domain analysis
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                high_freq_energy = np.sum(magnitude_spectrum[int(gray.shape[0]*0.4):, int(gray.shape[1]*0.4):]) / magnitude_spectrum.size
                
                # Color distribution
                color_variance = np.var(hsv[:,:,0])
                saturation_mean = np.mean(hsv[:,:,1])
                
                # Texture analysis
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                texture_variance = np.var(laplacian)
                
                # Face region detection
                h, w = gray.shape
                face_region = gray[int(h*0.2):int(h*0.7), int(w*0.3):int(w*0.7)]
                if face_region.size > 0:
                    face_sharpness = cv2.Laplacian(face_region, cv2.CV_64F).var()
                    face_contrast = np.std(face_region)
                else:
                    face_sharpness = 0
                    face_contrast = 0
                
                # 7. COMPRESSION ARTIFACT DETECTION
                # Deepfakes often have double compression artifacts
                block_size = 8
                block_artifacts = 0
                for by in range(0, h - block_size, block_size):
                    for bx in range(0, w - block_size, block_size):
                        block = gray[by:by+block_size, bx:bx+block_size]
                        block_std = np.std(block)
                        if block_std < 5:  # Overly smooth blocks (compression/generation artifact)
                            block_artifacts += 1
                compression_artifacts.append(block_artifacts / max(1, (h*w) // (block_size*block_size)))
                
                motion_blur_scores.append(motion_blur_score)
                
                # Temporal consistency
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    motion_intensity = np.mean(frame_diff)
                    
                    if motion_intensity > 50:
                        temporal_inconsistencies += 1
                    
                    # Motion blur analysis
                    if motion_intensity > 5:
                        blur_variance = cv2.Laplacian(frame_diff, cv2.CV_64F).var()
                        if 10 <= blur_variance <= 100:
                            motion_blur_scores.append(1.0)
                        elif blur_variance < 5:
                            motion_blur_scores.append(0.0)  # Too perfect (AI)
                        else:
                            motion_blur_scores.append(0.5)  # Excessive blur
                else:
                    motion_intensity = 0
                
                prev_frame = gray.copy()
                
                # Color channel correlation
                b, g, r = cv2.split(frame)
                rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
                color_inconsistencies.append(abs(rg_corr))
                
                # Noise grain analysis
                noise_estimate = np.std(gray - cv2.GaussianBlur(gray, (5,5), 0))
                if 2.0 <= noise_estimate <= 15.0:
                    noise_grain_levels.append(1.0)
                elif noise_estimate < 1.5:
                    noise_grain_levels.append(0.0)
                else:
                    noise_grain_levels.append(0.5)
                
                # Lighting consistency
                luminance = lab[:,:,0]
                luminance_std = np.std(luminance)
                luminance_mean = np.mean(luminance)
                light_ratio = luminance_std / max(1, luminance_mean)
                if 0.2 <= light_ratio <= 0.6:
                    lighting_consistency.append(1.0)
                elif light_ratio < 0.1:
                    lighting_consistency.append(0.0)
                else:
                    lighting_consistency.append(0.5)
                
                # Lens distortion detection
                if not lens_distortion_detected and current_idx == 0:
                    border_width = int(w * 0.1)
                    left_border = gray[:, :border_width]
                    right_border = gray[:, -border_width:]
                    
                    left_edges = cv2.Canny(left_border, 50, 150)
                    right_edges = cv2.Canny(right_border, 50, 150)
                    edge_curvature = np.sum(left_edges) + np.sum(right_edges)
                    
                    if edge_curvature > 100:
                        lens_distortion_detected = True
                
                # Aggregate frame features
                frame_features = [
                    mean_intensity / 255.0,
                    std_intensity / 128.0,
                    edge_density,
                    high_freq_energy,
                    color_variance / 180.0,
                    saturation_mean / 255.0,
                    min(1.0, texture_variance / 1000),
                    min(1.0, face_sharpness / 500),
                    face_contrast / 128.0,
                    motion_intensity / 100.0
                ]
                
                features.append(frame_features)
                current_idx += 1
            
            frame_count += 1
        
        cap.release()
        
        if len(features) == 0:
            return None
        
        # Calculate aggregate metrics
        avg_compression = np.mean(compression_artifacts) if compression_artifacts else 0
        avg_color_inconsist = np.mean(color_inconsistencies) if color_inconsistencies else 0
        temporal_inconsist_rate = temporal_inconsistencies / max(1, len(features) - 1)
        
        # === ENHANCED: Calculate authentic video characteristics ===
        natural_motion_blur = np.mean(motion_blur_scores) if motion_blur_scores else 0.5
        natural_noise_grain = np.mean(noise_grain_levels) if noise_grain_levels else 0.5
        natural_lighting = np.mean(lighting_consistency) if lighting_consistency else 0.5
        has_lens_distortion = 1.0 if lens_distortion_detected else 0.0
        
        # Flatten frame features and pad/truncate to consistent size
        features_flat = np.array(features).flatten()
        if len(features_flat) < 60:
            features_flat = np.pad(features_flat, (0, 60 - len(features_flat)), mode='constant')
        else:
            features_flat = features_flat[:60]
        
        # Add aggregate features (4 features to make 64 total)
        features_flat = np.append(features_flat, [
            duration / 60.0,                    # Normalized duration (minutes)
            avg_compression,                     # Compression artifact score
            avg_color_inconsist,                 # Color inconsistency
            temporal_inconsist_rate              # Temporal inconsistency rate
        ])
        
        # Return features and ENHANCED metrics including authentic characteristics
        return (features_flat, fps, total_frames, duration, 
                temporal_inconsistencies, avg_compression, avg_color_inconsist,
                natural_motion_blur, natural_noise_grain, natural_lighting, has_lens_distortion)
        
    except Exception as e:
        print(f"[Video Feature Extraction Error]: {str(e)}")
        return None

def analyze_video_with_mindspore(video_path):
    """
    Analyze video for deepfake/manipulation using MindSpore
    Enhanced with baseline profiling for authentic video detection
    Returns: dict with judgment, confidence, explanation
    """
    try:
        # === METADATA VERIFICATION (Camera/Device Fingerprinting) ===
        metadata_authentic = False
        codec_signature = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_signature = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()
            
            # Common real camera codecs: H264 (avc1), H265 (hevc), MJPEG
            real_codecs = ['avc1', 'h264', 'H264', 'hevc', 'hvc1', 'MJPG', 'mjpg']
            # AI-generated often use: VP8, VP9, or unusual codecs
            if any(codec in codec_signature for codec in real_codecs):
                metadata_authentic = True
        except:
            pass
        
        # Extract features
        result = extract_video_features(video_path)
        if result is None:
            return {
                'judgment': 'ERROR',
                'confidence': 0,
                'manipulation_score': 0,
                'explanation': 'Failed to process video file',
                'summary': 'ERROR - Could not analyze video'
            }
        
        # Unpack ENHANCED results with authentic video characteristics
        (features, fps, total_frames, duration, temporal_inconsist, compression_score, color_inconsist,
         motion_blur_natural, noise_grain_natural, lighting_natural, lens_distortion) = result
        
        # Initialize model
        model = initialize_model()
        
        # MindSpore prediction
        with mindspore_lock:
            input_tensor = Tensor(features.reshape(1, -1).astype(np.float32))
            prediction = model(input_tensor)
            prediction_np = prediction.asnumpy()[0]
            
            real_score = float(prediction_np[0])
            fake_score = float(prediction_np[1])
        
        # === BASELINE PROFILING FOR AUTHENTIC VIDEOS ===
        # Real recorded videos have specific characteristics
        authentic_indicators = []
        authenticity_score = 0  # 0-10 scale (higher = more likely real)
        
        # 1. NATURAL FPS RANGES (most cameras: 24, 25, 30, 60 fps)
        natural_fps_ranges = [(23, 26), (29, 31), (59, 61)]
        is_natural_fps = any(low <= fps <= high for low, high in natural_fps_ranges)
        if is_natural_fps:
            authentic_indicators.append(f"Natural camera frame rate: {fps:.1f} FPS")
            authenticity_score += 2
        
        # 2. REASONABLE DURATION (real recordings typically > 3 seconds)
        if duration >= 3:
            authentic_indicators.append(f"Normal video length: {duration:.1f}s")
            authenticity_score += 1
        if duration >= 10:
            authenticity_score += 1  # Longer videos are more likely authentic
        
        # 3. CONSISTENT FRAME COUNT (real cameras maintain steady frame rate)
        expected_frames = fps * duration
        frame_consistency = total_frames / expected_frames if expected_frames > 0 else 0
        if 0.95 <= frame_consistency <= 1.05:
            authentic_indicators.append("Consistent frame pacing (camera characteristic)")
            authenticity_score += 2
        
        # 4. LOW COMPRESSION ARTIFACTS (real videos have normal compression)
        if compression_score < 0.15:
            authentic_indicators.append("Normal compression levels")
            authenticity_score += 1
        
        # 5. NATURAL COLOR CORRELATION (real cameras have predictable color patterns)
        if 0.4 <= color_inconsist <= 0.85:
            authentic_indicators.append("Natural color characteristics")
            authenticity_score += 2
        
        # === ENHANCED AUTHENTIC CHARACTERISTICS ===
        
        # 6. NATURAL MOTION BLUR (real cameras capture motion blur during movement)
        if motion_blur_natural >= 0.7:
            authentic_indicators.append("Natural motion blur detected (real camera physics)")
            authenticity_score += 2
        elif motion_blur_natural >= 0.5:
            authentic_indicators.append("Motion blur present")
            authenticity_score += 1
        
        # 7. SENSOR NOISE GRAIN (real cameras have characteristic sensor noise)
        if noise_grain_natural >= 0.7:
            authentic_indicators.append("Natural sensor noise pattern (camera characteristic)")
            authenticity_score += 2
        elif noise_grain_natural >= 0.5:
            authentic_indicators.append("Sensor noise detected")
            authenticity_score += 1
        
        # 8. LIGHTING CONSISTENCY (real videos follow physical lighting laws)
        if lighting_natural >= 0.7:
            authentic_indicators.append("Consistent lighting physics (natural illumination)")
            authenticity_score += 2
        elif lighting_natural >= 0.5:
            authentic_indicators.append("Lighting appears natural")
            authenticity_score += 1
        
        # 9. LENS DISTORTION (real camera lenses have geometric distortion)
        if lens_distortion > 0.5:
            authentic_indicators.append("Lens distortion detected (real optical system)")
            authenticity_score += 2
        
        # 10. CODEC METADATA (real cameras use specific codecs)
        if metadata_authentic:
            authentic_indicators.append(f"Authentic codec signature: {codec_signature} (real camera encoding)")
            authenticity_score += 2
        
        # === DEEPFAKE DETECTION (Red Flags) ===
        red_flags = []
        severity_score = 0  # 0-10 scale
        
        # Check for AI-generated characteristics (opposite of authentic)
        
        # UNNATURAL MOTION (too perfect or missing motion blur)
        if motion_blur_natural < 0.3:
            red_flags.append("Unnatural motion patterns (missing real camera motion blur)")
            severity_score += 2
        
        # TOO CLEAN/SMOOTH (AI videos lack sensor noise)
        if noise_grain_natural < 0.3:
            red_flags.append("Unnaturally clean frames (missing sensor noise - AI indicator)")
            severity_score += 3
        
        # IMPOSSIBLE LIGHTING (physics-defying illumination)
        if lighting_natural < 0.3:
            red_flags.append("Inconsistent lighting physics (impossible illumination patterns)")
            severity_score += 3
        
        # NO LENS DISTORTION (perfect geometry = not a real lens)
        if lens_distortion < 0.5 and duration > 2:
            red_flags.append("No lens distortion detected (suggests software rendering)")
            severity_score += 2
        
        # SUSPICIOUS CODEC (non-standard encoding)
        if not metadata_authentic and codec_signature:
            red_flags.append(f"Unusual codec: {codec_signature} (not typical camera encoding)")
            severity_score += 2
        
        # 1. FPS ANALYSIS (deepfakes often have non-standard frame rates)
        if fps < 15:
            red_flags.append(f"Very low frame rate: {fps:.1f} FPS (possible deepfake generation)")
            severity_score += 3
        elif not is_natural_fps and (fps < 20 or fps > 65):
            red_flags.append(f"Non-standard frame rate: {fps:.1f} FPS (not typical camera output)")
            severity_score += 2
        
        # 2. DURATION CHECK (very short videos are suspicious)
        if duration < 1:
            red_flags.append(f"Extremely short video: {duration:.1f}s (common in fake clips)")
            severity_score += 3
        elif duration < 3:
            red_flags.append(f"Very short duration: {duration:.1f}s (suspicious)")
            severity_score += 1
        
        # 3. FRAME CONSISTENCY (deepfakes often have frame drops)
        if frame_consistency < 0.7:
            red_flags.append(f"Significant frame loss: {frame_consistency:.1%} consistency (possible re-encoding)")
            severity_score += 3
        elif frame_consistency < 0.9:
            red_flags.append(f"Frame inconsistency detected: {frame_consistency:.1%}")
            severity_score += 1
        
        # 4. TEMPORAL INCONSISTENCY (critical deepfake indicator)
        if temporal_inconsist > 8:
            red_flags.append(f"Severe temporal inconsistency: {temporal_inconsist} sudden jumps (strong deepfake indicator)")
            severity_score += 4
        elif temporal_inconsist > 5:
            red_flags.append(f"High temporal inconsistency: {temporal_inconsist} jumps")
            severity_score += 2
        elif temporal_inconsist > 3:
            red_flags.append(f"Moderate temporal inconsistencies: {temporal_inconsist} jumps")
            severity_score += 1
        
        # 5. COMPRESSION ARTIFACTS (double compression signature)
        if compression_score > 0.35:
            red_flags.append(f"Excessive compression artifacts: {compression_score:.1%} (strong re-encoding signature)")
            severity_score += 3
        elif compression_score > 0.2:
            red_flags.append(f"High compression artifacts: {compression_score:.1%}")
            severity_score += 1
        
        # 6. COLOR INCONSISTENCY (synthetic video indicator)
        if color_inconsist > 0.95:
            red_flags.append(f"Abnormal color correlation: {color_inconsist:.2f} (AI generation indicator)")
            severity_score += 3
        elif color_inconsist < 0.25:
            red_flags.append(f"Unusual color decorrelation: {color_inconsist:.2f} (possible manipulation)")
            severity_score += 2
        
        # === ENHANCED INTELLIGENT SCORING WITH MULTI-FACTOR ANALYSIS ===
        
        # Start with MindSpore prediction
        base_manipulation = fake_score * 100
        
        # Calculate authenticity bonus (real videos get negative manipulation score adjustment)
        # Maximum possible authenticity_score is now 20 (10 original + 10 enhanced features)
        authenticity_bonus = authenticity_score * 5  # Up to -100 points for highly authentic videos
        
        # Calculate suspicion penalty from red flags
        # Maximum possible severity_score is now ~25+ (original + enhanced checks)
        suspicion_penalty = severity_score * 5  # Up to +125 points for highly suspicious videos
        
        # TIERED ANALYSIS with enhanced thresholds
        if authenticity_score >= 12:
            # VERY STRONG authentic indicators (12+/20) - highly likely real recording
            # Trust authentic characteristics heavily
            manipulation_score = max(0, base_manipulation * 0.2 - authenticity_bonus * 0.8 + suspicion_penalty * 0.3)
        elif authenticity_score >= 8:
            # STRONG authentic indicators (8-11/20) - likely real recording
            # Reduce base manipulation significantly
            manipulation_score = max(0, base_manipulation * 0.3 - authenticity_bonus * 0.6 + suspicion_penalty * 0.5)
        elif severity_score >= 10:
            # VERY STRONG deepfake indicators (10+) - highly likely fake
            # Trust suspicion signals heavily
            manipulation_score = min(100, base_manipulation * 0.4 + suspicion_penalty * 0.8)
        elif severity_score >= 7:
            # STRONG deepfake indicators (7-9) - likely fake
            # Increase base manipulation significantly  
            manipulation_score = min(100, base_manipulation * 0.5 + suspicion_penalty * 0.6)
        else:
            # BALANCED analysis - use all signals with equal weight
            manipulation_score = base_manipulation * 0.5 - authenticity_bonus * 0.4 + suspicion_penalty * 0.5
        
        # Clamp to valid range
        manipulation_score = max(0, min(100, manipulation_score))
        
        # Determine judgment with improved thresholds
        if manipulation_score >= 65:
            judgment = "FAKE"
            confidence = 80 + (manipulation_score - 65) * 0.4
        elif manipulation_score >= 35:
            judgment = "SUSPICIOUS"
            confidence = 60 + (manipulation_score - 35) * 0.6
        else:
            judgment = "AUTHENTIC"
            confidence = 75 + (35 - manipulation_score) * 0.6
        
        confidence = min(95, max(55, confidence))
        
        # Generate comprehensive explanation
        explanation_parts = [
            f"Video Analysis: {total_frames} frames @ {fps:.1f} FPS ({duration:.1f}s)",
            f"MindSpore AI Score: {manipulation_score:.1f}% manipulation probability",
            f"Confidence Level: {confidence:.1f}%"
        ]
        
        # Show authentic indicators first (if any)
        if authentic_indicators:
            explanation_parts.append("\nAuthentic Characteristics Detected:")
            for indicator in authentic_indicators:
                explanation_parts.append(f"  • {indicator}")
            explanation_parts.append(f"  Authenticity Score: {authenticity_score}/10")
        
        # Then show red flags (if any)
        if red_flags:
            explanation_parts.append("\nSuspicious Indicators Found:")
            for flag in red_flags:
                explanation_parts.append(f"  • {flag}")
            explanation_parts.append(f"  Suspicion Level: {severity_score}/10")
        
        # Final conclusion
        if judgment == "AUTHENTIC":
            if authenticity_score >= 6:
                explanation_parts.append("\nCONCLUSION: Strong authentic video characteristics detected")
                explanation_parts.append("   This appears to be a genuine recording with natural camera properties.")
            else:
                explanation_parts.append("\nCONCLUSION: No significant manipulation detected")
                explanation_parts.append("   Video appears authentic with normal characteristics.")
        elif judgment == "SUSPICIOUS":
            explanation_parts.append("\nCONCLUSION: Possible manipulation or quality issues")
            explanation_parts.append("   Video shows some unusual characteristics. Could be edited or low-quality.")
        else:
            explanation_parts.append("\nCONCLUSION: High probability of deepfake or synthetic content")
            explanation_parts.append("   Multiple indicators suggest AI generation or heavy manipulation.")
        
        explanation = "\n".join(explanation_parts)
        
        # Summary for bubbles
        if judgment == "AUTHENTIC":
            summary = f"{judgment} - {confidence:.0f}% confidence\n{total_frames} frames | {authenticity_score}/10 authentic score"
        elif judgment == "SUSPICIOUS":
            summary = f"{judgment} - {confidence:.0f}% confidence\n{total_frames} frames | {severity_score}/10 suspicion level"
        else:
            summary = f"{judgment} - {confidence:.0f}% confidence\n{total_frames} frames | {severity_score}/10 deepfake indicators"
        
        return {
            'judgment': judgment,
            'confidence': round(confidence, 1),
            'manipulation_score': round(manipulation_score, 1),
            'explanation': explanation,
            'summary': summary,
            'video_info': {
                'fps': round(fps, 1),
                'frames': total_frames,
                'duration': round(duration, 1)
            },
            'authenticity_score': authenticity_score,
            'authentic_indicators': authentic_indicators,
            'red_flags': red_flags,
            'severity_score': severity_score
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[MindSpore Video Analysis Error]: {str(e)}\n{error_trace}")
        return {
            'judgment': 'ERROR',
            'confidence': 0,
            'manipulation_score': 0,
            'explanation': f'Analysis failed: {str(e)}',
            'summary': 'ERROR - Analysis failed'
        }

if __name__ == "__main__":
    # Test the detector
    print("MindSpore Video Deepfake Detector initialized")
    print("Model ready for video analysis")
