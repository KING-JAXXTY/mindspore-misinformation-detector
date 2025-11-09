"""
MindSpore Image Analysis Module
==============================

Advanced image analysis for deepfake and manipulation detection using MindSpore.
Includes CNN models for authenticity detection and metadata analysis.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import logging
import io
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Image processing libraries
try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not available. Image analysis will be limited.")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available. Advanced image analysis disabled.")

logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    """Extract features from images for manipulation detection"""
    
    def __init__(self):
        # Suspicious editing patterns with weights
        self.manipulation_indicators = {
            'compression_artifacts': 0.3,
            'inconsistent_lighting': 0.4,
            'edge_inconsistencies': 0.35,
            'color_inconsistencies': 0.3,
            'metadata_inconsistencies': 0.5,
            'noise_patterns': 0.25,
            'clone_stamp_patterns': 0.45,
            'jpeg_grid_anomalies': 0.35,
            'unnatural_sharpening': 0.3,
            'color_transfer': 0.4
        }
        
        # Common editing software signatures in EXIF
        self.editing_software = [
            'photoshop', 'gimp', 'pixlr', 'affinity', 'paint.net',
            'photoscape', 'lightroom', 'illustrator', 'indesign',
            'snapseed', 'vsco', 'facetune', 'beautycam', 'meitu',
            'picsart', 'canva', 'fotor', 'befunky', 'remove.bg',
            'cutout', 'photopea', 'sketch', 'procreate', 'krita'
        ]
        
        # AI generation software signatures
        self.ai_generation_tools = [
            'midjourney', 'dalle', 'stable diffusion', 'stablediffusion',
            'artbreeder', 'nightcafe', 'deepai', 'wombo', 'dream',
            'playground', 'leonardo', 'firefly', 'synthesia'
        ]
        
        # Common camera manufacturers for authenticity verification
        self.genuine_camera_brands = [
            'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
            'leica', 'pentax', 'hasselblad', 'phase one', 'samsung',
            'apple', 'google', 'huawei', 'xiaomi', 'oppo', 'vivo'
        ]
    
    def extract_image_features(self, image_data: bytes, filename: str = "") -> Dict[str, float]:
        """Extract comprehensive image features"""
        
        features = {
            # Basic image properties
            'file_size': len(image_data),
            'has_exif': 0.0,
            'has_gps': 0.0,
            'has_camera_info': 0.0,
            'has_software_info': 0.0,
            
            # Image quality metrics
            'resolution_score': 0.0,
            'compression_quality': 0.0,
            'noise_level': 0.0,
            'blur_detection': 0.0,
            
            # Manipulation indicators
            'metadata_consistency': 1.0,
            'compression_artifacts': 0.0,
            'edge_consistency': 1.0,
            'lighting_consistency': 1.0,
            'color_consistency': 1.0,
            
            # Technical features
            'aspect_ratio': 1.0,
            'bit_depth': 8.0,
            'color_space': 1.0,
            'histogram_irregularities': 0.0,
        }
        
        if not HAS_PIL:
            logger.warning("PIL not available, using basic analysis")
            return features
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image properties
            width, height = image.size
            features['resolution_score'] = min(1.0, (width * height) / (1920 * 1080))
            features['aspect_ratio'] = width / height if height > 0 else 1.0
            
            # Extract EXIF data
            exif_data = self._extract_exif_data(image)
            features.update(self._analyze_exif_data(exif_data))
            
            # Convert to RGB for analysis
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Analyze image quality
            features.update(self._analyze_image_quality(img_array))
            
            # Detect manipulation indicators
            features.update(self._detect_manipulation_indicators(img_array))
            
            # Analyze color and lighting
            features.update(self._analyze_color_lighting(img_array))
            
            # Advanced detection methods
            features['ai_generation_score'] = self._detect_ai_generation_artifacts(img_array)
            features['splicing_score'] = self._detect_splicing_boundaries(img_array)
            features['resolution_inconsistency'] = self._detect_resolution_inconsistency(img_array)
            features['perspective_inconsistency'] = self._detect_perspective_inconsistency(img_array)
            features['natural_camera_score'] = self._check_natural_camera_properties(features)
            
        except Exception as e:
            logger.error(f"Image feature extraction error: {e}")
        
        return features
    
    def _extract_exif_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF metadata from image"""
        exif_data = {}
        
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
        except Exception as e:
            logger.warning(f"EXIF extraction failed: {e}")
        
        return exif_data
    
    def _analyze_exif_data(self, exif_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze EXIF data for authenticity indicators"""
        features = {
            'has_exif': 1.0 if exif_data else 0.0,
            'has_gps': 0.0,
            'has_camera_info': 0.0,
            'has_software_info': 0.0,
            'metadata_consistency': 1.0
        }
        
        if not exif_data:
            return features
        
        # Check for GPS data
        if any(key in exif_data for key in ['GPSInfo', 'GPS']):
            features['has_gps'] = 1.0
        
        # Check for camera information
        camera_tags = ['Make', 'Model', 'LensModel', 'FocalLength']
        if any(tag in exif_data for tag in camera_tags):
            features['has_camera_info'] = 1.0
        
        # Check for software information
        software_tags = ['Software', 'ProcessingSoftware', 'CreatedBy']
        if any(tag in exif_data for tag in software_tags):
            features['has_software_info'] = 1.0
        
        # Analyze metadata consistency
        features['metadata_consistency'] = self._check_metadata_consistency(exif_data)
        
        return features
    
    def _check_metadata_consistency(self, exif_data: Dict[str, Any]) -> float:
        """Check for metadata inconsistencies that might indicate tampering"""
        consistency_score = 1.0
        issues_found = []
        
        # Check timestamp consistency
        date_tags = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        dates = [exif_data.get(tag) for tag in date_tags if tag in exif_data]
        
        if len(dates) > 1:
            date_strings = [str(d) for d in dates if d]
            if len(set(date_strings)) > 1:
                consistency_score -= 0.25
                issues_found.append('timestamp_mismatch')
        
        # Check for editing software
        software = exif_data.get('Software', '').lower()
        processing_software = exif_data.get('ProcessingSoftware', '').lower()
        
        if any(edit_tool in software for edit_tool in self.editing_software):
            consistency_score -= 0.4
            issues_found.append('editing_software_detected')
        
        if any(edit_tool in processing_software for edit_tool in self.editing_software):
            consistency_score -= 0.35
            issues_found.append('processing_software_detected')
        
        # Check for AI generation tools
        if any(ai_tool in software or ai_tool in processing_software 
               for ai_tool in self.ai_generation_tools):
            consistency_score -= 0.6
            issues_found.append('ai_generation_software')
        
        # Check for missing critical metadata in modern cameras
        critical_tags = ['Make', 'Model', 'DateTime']
        missing_critical = sum(1 for tag in critical_tags if tag not in exif_data)
        if missing_critical > 0 and len(exif_data) > 2:
            consistency_score -= 0.15 * missing_critical
            issues_found.append('missing_camera_metadata')
        
        # Check for GPS data removal (suspicious if no GPS but has detailed camera info)
        has_camera_info = all(tag in exif_data for tag in ['Make', 'Model'])
        has_gps = any(tag in exif_data for tag in ['GPSInfo', 'GPS', 'GPSLatitude'])
        if has_camera_info and not has_gps:
            # Many cameras have GPS, removal might indicate privacy scrubbing or tampering
            consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _analyze_image_quality(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze image quality metrics"""
        features = {
            'compression_quality': 0.5,
            'noise_level': 0.0,
            'blur_detection': 0.0,
        }
        
        try:
            # Calculate noise level (variance of Laplacian)
            if HAS_OPENCV:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
                features['noise_level'] = min(1.0, noise_level / 1000)
                
                # Blur detection
                blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
                features['blur_detection'] = 1.0 - min(1.0, blur_measure / 500)
            else:
                # Simplified noise detection without OpenCV
                gray = np.mean(img_array, axis=2)
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                noise_estimate = np.mean(grad_x) + np.mean(grad_y)
                features['noise_level'] = min(1.0, noise_estimate / 50)
                
        except Exception as e:
            logger.warning(f"Image quality analysis error: {e}")
        
        return features
    
    def _detect_manipulation_indicators(self, img_array: np.ndarray) -> Dict[str, float]:
        """Detect various manipulation indicators with advanced techniques"""
        features = {
            'compression_artifacts': 0.0,
            'edge_consistency': 1.0,
            'histogram_irregularities': 0.0,
            'clone_detection_score': 0.0,
            'unnatural_sharpening': 0.0,
            'jpeg_grid_anomalies': 0.0,
            'noise_inconsistency': 0.0,
        }
        
        try:
            # Detect compression artifacts (JPEG blocking)
            features['compression_artifacts'] = self._detect_jpeg_artifacts(img_array)
            
            # Edge consistency analysis
            features['edge_consistency'] = self._analyze_edge_consistency(img_array)
            
            # Histogram analysis
            features['histogram_irregularities'] = self._analyze_histogram_irregularities(img_array)
            
            # Clone stamp detection
            features['clone_detection_score'] = self._detect_clone_patterns(img_array)
            
            # Unnatural sharpening detection
            features['unnatural_sharpening'] = self._detect_unnatural_sharpening(img_array)
            
            # JPEG grid anomaly detection
            features['jpeg_grid_anomalies'] = self._detect_jpeg_grid_anomalies(img_array)
            
            # Noise consistency across regions
            features['noise_inconsistency'] = self._detect_noise_inconsistency(img_array)
            
        except Exception as e:
            logger.warning(f"Manipulation detection error: {e}")
        
        return features
    
    def _detect_jpeg_artifacts(self, img_array: np.ndarray) -> float:
        """Detect JPEG compression artifacts"""
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Look for 8x8 block patterns typical of JPEG compression
            block_size = 8
            artifact_score = 0.0
            
            height, width = gray.shape
            blocks_y = height // block_size
            blocks_x = width // block_size
            
            for by in range(blocks_y - 1):
                for bx in range(blocks_x - 1):
                    # Extract 8x8 blocks
                    block1 = gray[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
                    block2 = gray[(by+1)*block_size:(by+2)*block_size, bx*block_size:(bx+1)*block_size]
                    
                    # Calculate discontinuity at block boundaries
                    boundary_diff = np.mean(np.abs(block1[-1, :] - block2[0, :]))
                    artifact_score += boundary_diff
            
            # Normalize
            total_boundaries = blocks_y * blocks_x
            if total_boundaries > 0:
                artifact_score = min(1.0, artifact_score / (total_boundaries * 50))
            
            return artifact_score
            
        except Exception as e:
            logger.warning(f"JPEG artifact detection error: {e}")
            return 0.0
    
    def _analyze_edge_consistency(self, img_array: np.ndarray) -> float:
        """Analyze edge consistency for tampering detection"""
        try:
            if HAS_OPENCV:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Analyze edge distribution
                edge_density = np.sum(edges > 0) / edges.size
                
                # Look for unnatural edge patterns
                # High edge density in small regions might indicate splicing
                kernel = np.ones((20, 20), np.uint8)
                edge_dilated = cv2.dilate(edges, kernel, iterations=1)
                region_density = np.sum(edge_dilated > 0) / edge_dilated.size
                
                consistency_score = 1.0 - abs(edge_density - region_density)
                return max(0.0, consistency_score)
            else:
                # Simplified edge analysis without OpenCV
                gray = np.mean(img_array, axis=2)
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                
                edge_strength = np.mean(grad_x) + np.mean(grad_y)
                # Assume consistent if edge strength is moderate
                return 1.0 - min(1.0, abs(edge_strength - 10) / 20)
                
        except Exception as e:
            logger.warning(f"Edge consistency analysis error: {e}")
            return 0.5
    
    def _analyze_histogram_irregularities(self, img_array: np.ndarray) -> float:
        """Analyze color histogram for irregularities"""
        try:
            irregularity_score = 0.0
            
            # Analyze each color channel
            for channel in range(3):
                hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 255))
                
                # Look for unnatural spikes or gaps in histogram
                hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
                differences = np.abs(hist - hist_smooth)
                spike_score = np.sum(differences > np.std(differences) * 2) / 256
                
                irregularity_score += spike_score
            
            return min(1.0, irregularity_score / 3)
            
        except Exception as e:
            logger.warning(f"Histogram analysis error: {e}")
            return 0.0
    
    def _detect_clone_patterns(self, img_array: np.ndarray) -> float:
        """Detect clone stamp patterns (copy-paste within image)"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Sample patches from the image
            patch_size = 32
            stride = 16
            clone_score = 0.0
            comparisons = 0
            
            patches = []
            positions = []
            
            for y in range(0, height - patch_size, stride):
                for x in range(0, width - patch_size, stride):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    patches.append(patch)
                    positions.append((y, x))
            
            # Compare patches to find suspiciously similar regions
            for i in range(min(len(patches), 50)):  # Limit comparisons
                for j in range(i+1, min(len(patches), 50)):
                    # Skip if patches are adjacent
                    y1, x1 = positions[i]
                    y2, x2 = positions[j]
                    distance = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                    
                    if distance < patch_size * 2:
                        continue
                    
                    # Calculate similarity
                    diff = np.abs(patches[i].astype(float) - patches[j].astype(float))
                    similarity = 1.0 - (np.mean(diff) / 255.0)
                    
                    if similarity > 0.95:  # Very high similarity
                        clone_score += 1.0
                    
                    comparisons += 1
            
            if comparisons > 0:
                return min(1.0, clone_score / max(1, comparisons * 0.01))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Clone detection error: {e}")
            return 0.0
    
    def _detect_unnatural_sharpening(self, img_array: np.ndarray) -> float:
        """Detect oversharpening or unnatural edge enhancement"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(float)
            
            # Detect oversharpened edges (halos around edges)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate the variance of the Laplacian
            lap_var = np.var(laplacian)
            
            # High variance might indicate oversharpening
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # If high Laplacian variance but low actual edge density = oversharpening
            if lap_var > 1000 and edge_density < 0.05:
                return min(1.0, lap_var / 5000)
            
            # Look for sharpening halos (bright/dark rings around edges)
            kernel = np.ones((3,3), np.float32) / 9
            blurred = cv2.filter2D(gray, -1, kernel)
            diff = gray - blurred
            
            # Excessive differences indicate sharpening
            sharpening_score = np.mean(np.abs(diff)) / 50
            
            return min(1.0, sharpening_score)
            
        except Exception as e:
            logger.warning(f"Sharpening detection error: {e}")
            return 0.0
    
    def _detect_jpeg_grid_anomalies(self, img_array: np.ndarray) -> float:
        """Detect JPEG grid anomalies from double compression or splicing"""
        try:
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            height, width = gray.shape
            block_size = 8
            
            # Analyze multiple grid offsets to detect misaligned JPEG grids
            anomaly_scores = []
            
            for offset_y in [0, 4]:
                for offset_x in [0, 4]:
                    grid_score = 0.0
                    blocks_checked = 0
                    
                    for by in range(offset_y, height - block_size, block_size):
                        for bx in range(offset_x, width - block_size, block_size):
                            if by + block_size < height and bx + block_size < width:
                                block = gray[by:by+block_size, bx:bx+block_size]
                                
                                # Check for block boundary discontinuities
                                if bx + block_size * 2 < width:
                                    next_block = gray[by:by+block_size, bx+block_size:bx+block_size*2]
                                    boundary_diff = np.mean(np.abs(block[:, -1] - next_block[:, 0]))
                                    grid_score += boundary_diff
                                    blocks_checked += 1
                    
                    if blocks_checked > 0:
                        anomaly_scores.append(grid_score / blocks_checked)
            
            if anomaly_scores:
                # If different offsets give very different scores, suggests splicing
                anomaly_variance = np.var(anomaly_scores)
                return min(1.0, anomaly_variance / 100)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"JPEG grid anomaly detection error: {e}")
            return 0.0
    
    def _detect_noise_inconsistency(self, img_array: np.ndarray) -> float:
        """Detect inconsistent noise patterns across the image"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(float)
            height, width = gray.shape
            
            # Divide image into regions
            region_size = min(height, width) // 4
            noise_levels = []
            
            for y in range(0, height - region_size, region_size):
                for x in range(0, width - region_size, region_size):
                    region = gray[y:y+region_size, x:x+region_size]
                    
                    # Estimate noise level using high-frequency component
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    noise_level = np.var(laplacian)
                    noise_levels.append(noise_level)
            
            if len(noise_levels) > 1:
                # Calculate variance in noise levels
                # Spliced/composited images have inconsistent noise
                noise_variance = np.var(noise_levels) / (np.mean(noise_levels) + 1)
                return min(1.0, noise_variance / 10)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Noise inconsistency detection error: {e}")
            return 0.0
    
    def _analyze_color_lighting(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze color and lighting consistency"""
        features = {
            'lighting_consistency': 1.0,
            'color_consistency': 1.0,
        }
        
        try:
            # Analyze lighting consistency across the image
            features['lighting_consistency'] = self._analyze_lighting_consistency(img_array)
            
            # Analyze color consistency
            features['color_consistency'] = self._analyze_color_consistency(img_array)
            
        except Exception as e:
            logger.warning(f"Color/lighting analysis error: {e}")
        
        return features
    
    def _analyze_lighting_consistency(self, img_array: np.ndarray) -> float:
        """Analyze lighting consistency across the image"""
        try:
            # Convert to grayscale for luminance analysis
            luminance = np.mean(img_array, axis=2)
            
            # Divide image into regions and analyze luminance distribution
            height, width = luminance.shape
            region_size = min(height, width) // 4
            
            regional_means = []
            for y in range(0, height - region_size, region_size):
                for x in range(0, width - region_size, region_size):
                    region = luminance[y:y+region_size, x:x+region_size]
                    regional_means.append(np.mean(region))
            
            if len(regional_means) > 1:
                # Calculate consistency as inverse of variance
                variance = np.var(regional_means)
                consistency = 1.0 / (1.0 + variance / 100)
                return min(1.0, consistency)
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Lighting consistency analysis error: {e}")
            return 0.5
    
    def _analyze_color_consistency(self, img_array: np.ndarray) -> float:
        """Analyze color consistency for splicing detection"""
        try:
            # Analyze color distribution in different regions
            height, width, channels = img_array.shape
            region_size = min(height, width) // 3
            
            color_stats = []
            for y in range(0, height - region_size, region_size):
                for x in range(0, width - region_size, region_size):
                    region = img_array[y:y+region_size, x:x+region_size]
                    
                    # Calculate mean and std for each channel
                    region_stats = []
                    for c in range(channels):
                        region_stats.extend([np.mean(region[:, :, c]), np.std(region[:, :, c])])
                    
                    color_stats.append(region_stats)
            
            if len(color_stats) > 1:
                color_stats = np.array(color_stats)
                
                # Calculate consistency across regions
                consistency_scores = []
                for stat_idx in range(color_stats.shape[1]):
                    stat_variance = np.var(color_stats[:, stat_idx])
                    consistency_scores.append(1.0 / (1.0 + stat_variance / 1000))
                
                return min(1.0, np.mean(consistency_scores))
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Color consistency analysis error: {e}")
            return 0.5
    
    def _detect_ai_generation_artifacts(self, img_array: np.ndarray) -> float:
        """Detect AI generation artifacts (GAN/diffusion model patterns)"""
        try:
            # AI-generated images often have:
            # 1. Overly smooth textures in unnatural areas
            # 2. Repeated patterns at certain frequencies
            # 3. Unusual color gradients
            
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            height, width = gray.shape
            
            ai_score = 0.0
            
            # Check for unnatural smoothness in texture regions
            patch_size = 32
            smoothness_scores = []
            
            for y in range(0, min(height - patch_size, 200), patch_size):
                for x in range(0, min(width - patch_size, 200), patch_size):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    variance = np.var(patch)
                    smoothness_scores.append(variance)
            
            if smoothness_scores:
                # AI images often have unnaturally consistent variance
                variance_of_variance = np.var(smoothness_scores)
                if variance_of_variance < 100:  # Too consistent
                    ai_score += 0.3
            
            # Check for repeated patterns (common in GANs)
            if HAS_OPENCV:
                # Use frequency domain analysis
                small_gray = cv2.resize(gray.astype(np.uint8), (256, 256))
                f_transform = np.fft.fft2(small_gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                # AI-generated images often have unusual frequency patterns
                center = magnitude.shape[0] // 2
                high_freq = magnitude[center+20:, center+20:]
                low_freq = magnitude[center-20:center+20, center-20:center+20]
                
                freq_ratio = np.mean(high_freq) / (np.mean(low_freq) + 1)
                if freq_ratio < 0.1:  # Unnaturally low high-frequency content
                    ai_score += 0.4
            
            return min(1.0, ai_score)
            
        except Exception as e:
            logger.warning(f"AI generation detection error: {e}")
            return 0.0
    
    def _detect_splicing_boundaries(self, img_array: np.ndarray) -> float:
        """Detect image splicing by analyzing boundaries"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Detect edges
            edges = cv2.Canny(gray, 30, 100)
            
            # Look for suspiciously straight boundaries (common in splicing)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
            
            splice_score = 0.0
            
            if lines is not None and len(lines) > 10:
                # Too many straight lines might indicate splicing
                splice_score += min(0.5, len(lines) / 100)
            
            # Check for discontinuities in texture at potential boundaries
            # Divide into 4 quadrants and compare
            h_mid, w_mid = height // 2, width // 2
            quadrants = [
                gray[0:h_mid, 0:w_mid],
                gray[0:h_mid, w_mid:width],
                gray[h_mid:height, 0:w_mid],
                gray[h_mid:height, w_mid:width]
            ]
            
            texture_features = []
            for quad in quadrants:
                if quad.size > 0:
                    # Calculate texture using Laplacian variance
                    laplacian = cv2.Laplacian(quad, cv2.CV_64F)
                    texture_features.append(np.var(laplacian))
            
            if len(texture_features) == 4:
                texture_variance = np.var(texture_features)
                if texture_variance > 10000:  # High variance between quadrants
                    splice_score += 0.4
            
            return min(1.0, splice_score)
            
        except Exception as e:
            logger.warning(f"Splicing detection error: {e}")
            return 0.0
    
    def _detect_resolution_inconsistency(self, img_array: np.ndarray) -> float:
        """Detect resolution inconsistencies (common in composites)"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Divide into regions and check sharpness
            region_size = min(height, width) // 4
            sharpness_scores = []
            
            for y in range(0, height - region_size, region_size):
                for x in range(0, width - region_size, region_size):
                    region = gray[y:y+region_size, x:x+region_size]
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    sharpness = laplacian.var()
                    sharpness_scores.append(sharpness)
            
            if len(sharpness_scores) > 1:
                # High variance in sharpness suggests composite
                sharpness_variance = np.var(sharpness_scores)
                mean_sharpness = np.mean(sharpness_scores)
                
                if mean_sharpness > 0:
                    coefficient_of_variation = np.sqrt(sharpness_variance) / mean_sharpness
                    return min(1.0, coefficient_of_variation / 2)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Resolution inconsistency detection error: {e}")
            return 0.0
    
    def _detect_perspective_inconsistency(self, img_array: np.ndarray) -> float:
        """Detect inconsistent perspectives (spliced objects)"""
        try:
            if not HAS_OPENCV:
                return 0.0
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect lines and analyze vanishing points
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) < 10:
                return 0.0
            
            # Check for multiple conflicting vanishing points
            angles = []
            for line in lines[:50]:
                rho, theta = line[0]
                angles.append(theta)
            
            # Cluster angles
            angles = np.array(angles)
            angle_variance = np.var(angles)
            
            # High variance in line angles might indicate inconsistent perspective
            if angle_variance > 0.5:
                return min(1.0, angle_variance / 2)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Perspective inconsistency detection error: {e}")
            return 0.0
    
    def _check_natural_camera_properties(self, features: Dict[str, float]) -> float:
        """Check if image has natural camera properties (not screenshot/edited)"""
        score = 0.0
        
        # Original camera metadata
        if features['has_exif'] and features['has_camera_info']:
            score += 0.4
        
        # GPS data (phones usually have this)
        if features['has_gps']:
            score += 0.2
        
        # Natural aspect ratio (common camera ratios)
        aspect_ratio = features.get('aspect_ratio', 1.0)
        common_ratios = [16/9, 4/3, 3/2, 1/1]
        if any(abs(aspect_ratio - ratio) < 0.05 for ratio in common_ratios):
            score += 0.2
        
        # Natural resolution (common camera resolutions)
        resolution = features.get('resolution_score', 0)
        if 0.3 < resolution < 1.2:  # Not too low, not suspiciously high
            score += 0.2
        
        return min(1.0, score)

class MindSporeImageClassifier(nn.Cell):
    """MindSpore CNN for image manipulation detection - NO DROPOUT (crashes removed)"""
    
    def __init__(self, input_size=20, num_classes=4):
        super(MindSporeImageClassifier, self).__init__()
        
        # Feature processing network
        self.features = nn.SequentialCell([
            nn.Dense(input_size, 128),
            nn.ReLU(),
            
            nn.Dense(128, 256),
            nn.ReLU(),
            
            nn.Dense(256, 128),
            nn.ReLU(),
            
            nn.Dense(128, 64),
            nn.ReLU(),
        ])
        
        self.classifier = nn.Dense(64, num_classes)
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        probabilities = self.softmax(logits)
        return probabilities

class ImageAnalyzer:
    """Main image analyzer using MindSpore"""
    
    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.model = None
        self._initialize_model()
        logger.info(f"ImageAnalyzer initialized. PIL: {HAS_PIL}, OpenCV: {HAS_OPENCV}")
    
    def _initialize_model(self):
        """Initialize the MindSpore model"""
        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            
            self.model = MindSporeImageClassifier(input_size=20, num_classes=4)
            
            # Load pre-trained weights if available
            model_path = Path(__file__).parent.parent / 'models' / 'image_classifier.ckpt'
            if model_path.exists():
                param_dict = ms.load_checkpoint(str(model_path))
                ms.load_param_into_net(self.model, param_dict)
                logger.info("Loaded pre-trained image model weights")
            else:
                logger.info("Using randomly initialized image model weights")
            
            self.model.set_train(False)
            
        except Exception as e:
            logger.error(f"Image model initialization error: {e}")
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return HAS_PIL  # Need at least PIL for basic image processing
    
    def analyze(self, image_file) -> Dict[str, Any]:
        """Analyze image for manipulation and authenticity"""
        try:
            # Read image data
            image_file.seek(0)
            image_data = image_file.read()
            image_file.seek(0)
            
            if not image_data:
                return {
                    "judgment": "No Data",
                    "explanation": "Could not read image data.",
                    "confidence": 0.0,
                    "reliability_score": 50,
                    "is_misinformation": False,
                    "features": {}
                }
            
            # Extract features
            features = self.feature_extractor.extract_image_features(image_data, image_file.filename)
            
            # Normalize features for neural network
            feature_vector = self._normalize_features(features)
            
            # Get neural network prediction if model is available
            nn_prediction = None
            if self.model is not None:
                try:
                    input_tensor = Tensor(feature_vector.reshape(1, -1), ms.float32)
                    probabilities = self.model(input_tensor)
                    nn_prediction = probabilities.asnumpy()[0]
                    logger.debug(f"[ML] Image MindSpore prediction: {nn_prediction}")
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"[WARNING] Image MindSpore prediction failed: {e}")
                    
                    # If graph execution error or signal abort, try to recover
                    if "Model execution error" in error_msg or "graph_scheduler" in error_msg or "signal is aborted" in error_msg:
                        logger.info("[RECOVERY] Attempting to reinitialize Image analyzer model...")
                        try:
                            self._initialize_model()
                            logger.info("[RECOVERY] Image model reinitialized successfully")
                        except Exception as reinit_error:
                            logger.error(f"[RECOVERY] Failed to reinitialize Image model: {reinit_error}")
                            self.model = None
            
            # Rule-based analysis
            rule_based_result = self._rule_based_image_analysis(features, image_data)
            
            # Combine results
            final_result = self._combine_image_predictions(rule_based_result, nn_prediction, features)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "judgment": "No Data",
                "explanation": f"Image analysis failed: {str(e)}",
                "confidence": 0.0,
                "reliability_score": 50,
                "is_misinformation": False,
                "features": {}
            }
    
    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features for neural network input"""
        key_features = [
            'file_size', 'has_exif', 'has_gps', 'has_camera_info', 'has_software_info',
            'resolution_score', 'compression_quality', 'noise_level', 'blur_detection',
            'metadata_consistency', 'compression_artifacts', 'edge_consistency',
            'lighting_consistency', 'color_consistency', 'aspect_ratio', 'bit_depth',
            'color_space', 'histogram_irregularities'
        ]
        
        values = []
        for feature in key_features:
            value = features.get(feature, 0)
            
            # Apply appropriate normalization
            if feature == 'file_size':
                value = min(value / (10 * 1024 * 1024), 1.0)  # Normalize to 10MB max
            elif feature == 'aspect_ratio':
                value = min(value / 3.0, 1.0)  # Normalize unusual aspect ratios
            elif feature == 'bit_depth':
                value = min(value / 24.0, 1.0)  # Normalize to 24-bit max
            
            values.append(value)
        
        # Pad or truncate to exactly 20 features
        while len(values) < 20:
            values.append(0.0)
        values = values[:20]
        
        return np.array(values, dtype=np.float32)
    
    def _rule_based_image_analysis(self, features: Dict[str, float], image_data: bytes) -> Dict[str, Any]:
        """Enhanced rule-based image authenticity analysis with improved logic and detailed explanations"""
        
        authenticity_score = 1.0
        red_flags = []
        positive_indicators = []
        manipulation_severity = 0  # Track severity: 0=none, 1=minor, 2=moderate, 3=severe
        
        # Positive indicators (increase authenticity) - stronger weights for genuine camera metadata
        if features['has_exif'] and features['has_camera_info']:
            authenticity_score += 0.25
            positive_indicators.append("Original camera EXIF metadata present")
        
        if features['has_gps']:
            authenticity_score += 0.1
            positive_indicators.append("GPS location data embedded")
        
        if features['metadata_consistency'] > 0.85:
            authenticity_score += 0.2
            positive_indicators.append("Metadata timestamps and properties are consistent")
        elif features['metadata_consistency'] > 0.7:
            authenticity_score += 0.1
            positive_indicators.append("Metadata appears mostly consistent")
        
        if features['edge_consistency'] > 0.85:
            authenticity_score += 0.15
            positive_indicators.append("Edge patterns show natural consistency")
        
        if features['lighting_consistency'] > 0.85 and features['color_consistency'] > 0.85:
            authenticity_score += 0.2
            positive_indicators.append("Lighting and color distribution appear natural")
        
        if features.get('noise_inconsistency', 0) < 0.15:
            authenticity_score += 0.15
            positive_indicators.append("Noise patterns consistent across image regions")
        
        if features['compression_artifacts'] < 0.3 and features.get('jpeg_grid_anomalies', 0) < 0.2:
            authenticity_score += 0.1
            positive_indicators.append("Natural compression patterns detected")
        
        # Check natural camera properties
        natural_score = features.get('natural_camera_score', 0)
        if natural_score > 0.7:
            authenticity_score += 0.15
            positive_indicators.append("Image shows natural camera characteristics")
        
        # Negative indicators (decrease authenticity) - tiered by severity
        
        # AI GENERATION DETECTION (highest severity)
        ai_score = features.get('ai_generation_score', 0)
        if ai_score > 0.6:
            authenticity_score -= 0.6
            manipulation_severity = 3
            red_flags.append(f"CRITICAL: AI generation artifacts detected ({ai_score:.0%}) - likely synthetic/generated image")
        elif ai_score > 0.4:
            authenticity_score -= 0.35
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"AI generation patterns detected ({ai_score:.0%}) - possible synthetic elements")
        
        # SPLICING DETECTION (high severity)
        splice_score = features.get('splicing_score', 0)
        if splice_score > 0.6:
            authenticity_score -= 0.5
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"CRITICAL: Image splicing detected ({splice_score:.0%}) - multiple images combined")
        elif splice_score > 0.4:
            authenticity_score -= 0.3
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Splicing indicators found ({splice_score:.0%}) - possible composite image")
        
        # RESOLUTION & PERSPECTIVE INCONSISTENCIES (high severity for composites)
        resolution_inconsist = features.get('resolution_inconsistency', 0)
        if resolution_inconsist > 0.5:
            authenticity_score -= 0.4
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"CRITICAL: Resolution inconsistency ({resolution_inconsist:.0%}) - different image sources combined")
        elif resolution_inconsist > 0.3:
            authenticity_score -= 0.2
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Resolution variations detected ({resolution_inconsist:.0%})")
        
        perspective_inconsist = features.get('perspective_inconsistency', 0)
        if perspective_inconsist > 0.5:
            authenticity_score -= 0.35
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Perspective inconsistency ({perspective_inconsist:.0%}) - conflicting vanishing points detected")
        
        # CRITICAL indicators (high severity - likely FAKE)
        if features.get('clone_detection_score', 0) > 0.6:
            authenticity_score -= 0.5
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"CRITICAL: Clone stamp patterns detected ({features['clone_detection_score']:.0%}) - duplicated regions found within image")
        elif features.get('clone_detection_score', 0) > 0.35:
            authenticity_score -= 0.3
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Clone stamp patterns detected ({features['clone_detection_score']:.0%}) - possible content removal or duplication")
        
        if features.get('jpeg_grid_anomalies', 0) > 0.5:
            authenticity_score -= 0.4
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"CRITICAL: JPEG grid anomalies ({features['jpeg_grid_anomalies']:.0%}) - double compression or splicing detected")
        elif features.get('jpeg_grid_anomalies', 0) > 0.3:
            authenticity_score -= 0.25
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"JPEG grid anomalies detected ({features['jpeg_grid_anomalies']:.0%}) - possible multiple edits")
        
        if features.get('noise_inconsistency', 0) > 0.6:
            authenticity_score -= 0.4
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"CRITICAL: Noise pattern inconsistency ({features['noise_inconsistency']:.0%}) - composited elements with different noise levels")
        elif features.get('noise_inconsistency', 0) > 0.4:
            authenticity_score -= 0.25
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Noise pattern inconsistency detected ({features['noise_inconsistency']:.0%}) - possible composite image")
        
        # SEVERE indicators (likely MANIPULATED)
        if features['metadata_consistency'] < 0.4:
            authenticity_score -= 0.4
            manipulation_severity = max(manipulation_severity, 3)
            red_flags.append(f"Severe metadata inconsistencies (score: {features['metadata_consistency']:.0%}) - editing software detected or timestamps mismatched")
        elif features['metadata_consistency'] < 0.6:
            authenticity_score -= 0.2
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Metadata inconsistencies found (score: {features['metadata_consistency']:.0%}) - possible editing")
        
        if features['edge_consistency'] < 0.5:
            authenticity_score -= 0.35
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Edge inconsistencies detected ({features['edge_consistency']:.0%}) - possible splicing or copy-paste manipulation")
        elif features['edge_consistency'] < 0.7:
            authenticity_score -= 0.15
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Minor edge irregularities ({features['edge_consistency']:.0%}) - possible minor edits")
        
        if features['lighting_consistency'] < 0.5:
            authenticity_score -= 0.3
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Lighting inconsistency ({features['lighting_consistency']:.0%}) - unnatural lighting patterns across regions")
        elif features['lighting_consistency'] < 0.7:
            authenticity_score -= 0.1
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Minor lighting variations detected ({features['lighting_consistency']:.0%})")
        
        if features['color_consistency'] < 0.5:
            authenticity_score -= 0.3
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Color inconsistency ({features['color_consistency']:.0%}) - color distribution varies unnaturally")
        elif features['color_consistency'] < 0.7:
            authenticity_score -= 0.1
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Minor color adjustments detected ({features['color_consistency']:.0%})")
        
        # MODERATE indicators (likely EDITED)
        if features['compression_artifacts'] > 0.75:
            authenticity_score -= 0.35
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"HIGH compression artifacts detected ({features['compression_artifacts']:.0%}) - suggests multiple saves/re-encodings")
        elif features['compression_artifacts'] > 0.5:
            authenticity_score -= 0.15
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Moderate compression artifacts ({features['compression_artifacts']:.0%}) - possible re-saving")
        
        if features['histogram_irregularities'] > 0.6:
            authenticity_score -= 0.25
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Histogram irregularities ({features['histogram_irregularities']:.0%}) - color adjustments or filters applied")
        elif features['histogram_irregularities'] > 0.4:
            authenticity_score -= 0.1
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Minor histogram adjustments detected ({features['histogram_irregularities']:.0%})")
        
        if features.get('unnatural_sharpening', 0) > 0.6:
            authenticity_score -= 0.3
            manipulation_severity = max(manipulation_severity, 2)
            red_flags.append(f"Unnatural sharpening detected ({features['unnatural_sharpening']:.0%}) - excessive edge enhancement applied")
        elif features.get('unnatural_sharpening', 0) > 0.4:
            authenticity_score -= 0.1
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append(f"Minor sharpening applied ({features['unnatural_sharpening']:.0%})")
        
        # MINOR indicators
        if not features['has_exif']:
            authenticity_score -= 0.2
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append("No EXIF metadata - stripped, never present, or screenshot")
        elif features['has_software_info']:
            authenticity_score -= 0.1
            manipulation_severity = max(manipulation_severity, 1)
            red_flags.append("Image processed with editing software")
        
        # Determine judgment based on severity, red flags count, and authenticity score
        red_flag_count = len(red_flags)
        critical_flag_count = sum(1 for flag in red_flags if flag.startswith("CRITICAL:"))
        
        # Advanced judgment logic using multiple factors
        if manipulation_severity >= 3 or critical_flag_count >= 2 or authenticity_score < 0.3:
            judgment = "HEAVILY MANIPULATED"  # Changed from "FAKE" - more accurate for images
            reliability_score = max(5, min(40, int(50 + authenticity_score * 15)))
        elif manipulation_severity >= 2 or critical_flag_count >= 1 or (red_flag_count >= 3 and authenticity_score < 0.6):
            judgment = "MANIPULATED"
            reliability_score = max(20, min(60, int(55 + authenticity_score * 25)))
        elif manipulation_severity >= 1 or red_flag_count >= 2 or authenticity_score < 0.9:
            judgment = "LIKELY EDITED"
            reliability_score = max(40, min(80, int(60 + authenticity_score * 20)))
        else:
            judgment = "AUTHENTIC"
            reliability_score = min(98, max(70, int(75 + authenticity_score * 23)))
        
        # Generate comprehensive explanation (no markdown formatting for clean UI)
        severity_text = ["None", "Minor", "Moderate", "Severe"][manipulation_severity]
        
        if judgment == "HEAVILY MANIPULATED":
            explanation = f"HEAVILY MANIPULATED IMAGE - Severity: {severity_text}\n"
            explanation += f"Found {red_flag_count} manipulation indicators"
            if critical_flag_count > 0:
                explanation += f" ({critical_flag_count} critical)"
            explanation += ".\n\n"
            
            # Show critical flags first
            critical_flags = [f for f in red_flags if f.startswith("CRITICAL:")]
            other_flags = [f for f in red_flags if not f.startswith("CRITICAL:")]
            
            if critical_flags:
                explanation += "CRITICAL EVIDENCE:\n" + "\n".join(f"• {flag.replace('CRITICAL: ', '')}" for flag in critical_flags[:3])
                if other_flags:
                    explanation += "\n\nAdditional Evidence:\n" + "\n".join(f"• {flag}" for flag in other_flags[:3])
            else:
                explanation += "MANIPULATION EVIDENCE:\n" + "\n".join(f"• {flag}" for flag in red_flags[:6])
            
            if positive_indicators:
                explanation += "\n\nNote: Some authentic properties remain, but manipulation evidence is overwhelming."
        
        elif judgment == "MANIPULATED":
            explanation = f"MANIPULATED IMAGE - Severity: {severity_text}\n"
            explanation += f"Significant editing detected with {red_flag_count} indicators.\n\n"
            
            critical_flags = [f for f in red_flags if f.startswith("CRITICAL:")]
            other_flags = [f for f in red_flags if not f.startswith("CRITICAL:")]
            
            if critical_flags:
                explanation += "Key Evidence:\n" + "\n".join(f"• {flag.replace('CRITICAL: ', '')}" for flag in critical_flags)
                if other_flags:
                    explanation += "\n\nSupporting Evidence:\n" + "\n".join(f"• {flag}" for flag in other_flags[:4])
            else:
                explanation += "Evidence:\n" + "\n".join(f"• {flag}" for flag in red_flags[:5])
            
            if positive_indicators and len(positive_indicators) >= 2:
                explanation += f"\n\nAuthentic Elements Present: {', '.join(positive_indicators[:3])}"
        
        elif judgment == "LIKELY EDITED":
            explanation = f"LIKELY EDITED - Severity: {severity_text}\n"
            explanation += f"Minor editing detected, but core image appears mostly authentic.\n\n"
            
            if red_flags:
                explanation += "Editing Indicators:\n" + "\n".join(f"• {flag}" for flag in red_flags[:4])
            
            if positive_indicators:
                explanation += f"\n\nAuthentic Properties: {', '.join(positive_indicators[:4])}"
            
            explanation += "\n\nAssessment: Likely basic adjustments (cropping, filters, color correction) rather than content manipulation."
        
        else:  # AUTHENTIC
            explanation = "AUTHENTIC IMAGE - Severity: None\n"
            explanation += "No significant manipulation detected. Image appears genuine.\n\n"
            
            if positive_indicators:
                explanation += "Authenticity Indicators:\n" + "\n".join(f"{ind}" for ind in positive_indicators[:5])
            
            if red_flags:
                explanation += f"\n\nMinor Note: {red_flags[0]}" if len(red_flags) == 1 else ""
            
            explanation += "\n\nAssessment: Image shows characteristics of an authentic, unedited photograph."
        
        # Add technical summary
        explanation += f"\n\nTechnical Analysis:"
        explanation += f"\n• Authenticity Score: {authenticity_score:.2f}/2.0"
        explanation += f"\n• Manipulation Severity: {severity_text}"
        explanation += f"\n• Indicators Found: {red_flag_count}"
        explanation += f"\n• Confidence Level: {reliability_score}%"
        
        return {
            "judgment": judgment,
            "explanation": explanation,
            "confidence": min(0.95, 0.5 + abs(authenticity_score - 0.7)),
            "reliability_score": reliability_score,
            "is_misinformation": judgment in ["HEAVILY MANIPULATED", "MANIPULATED"],
            "authenticity_score": authenticity_score,
            "red_flag_count": red_flag_count,
            "manipulation_score": max(0, 1.0 - (authenticity_score / 2.0)) * 100
        }
    
    def _combine_image_predictions(self, rule_based: Dict[str, Any], nn_prediction: np.ndarray,
                                 features: Dict[str, float]) -> Dict[str, Any]:
        """Combine rule-based and neural network predictions for image analysis"""
        
        result = rule_based.copy()
        result["features"] = features
        result["analysis_method"] = "Rule-based Image Analysis"
        
        if nn_prediction is not None:
            class_names = ["Real", "Half-Truth", "Fake", "No Data"]
            nn_judgment = class_names[np.argmax(nn_prediction)]
            nn_confidence = float(np.max(nn_prediction))
            
            # Weight the predictions
            rule_weight = 0.8
            nn_weight = 0.2
            
            combined_confidence = (rule_based["confidence"] * rule_weight + 
                                 nn_confidence * nn_weight)
            
            if nn_judgment != rule_based["judgment"] and nn_confidence > 0.7:
                result["explanation"] += f" Neural network analysis suggests '{nn_judgment}' with {nn_confidence:.2f} confidence."
            
            result["confidence"] = combined_confidence
            result["nn_prediction"] = {
                "judgment": nn_judgment,
                "confidence": nn_confidence,
                "probabilities": nn_prediction.tolist()
            }
            result["analysis_method"] = "Combined Image Analysis (Rule-based + Neural Network)"
        
        return result
