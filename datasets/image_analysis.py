"""
Image Analysis Features for Vega 2.0 - Phase 1.3 Implementation

Advanced image analysis capabilities including:
- Content-based image retrieval
- Image similarity and clustering
- Automated tagging and categorization
- Reverse image search
- Image quality assessment

This module builds on the computer vision models to provide higher-level analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageStat
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFeatureExtractor:
    """Extract deep features from images for similarity and retrieval."""

    def __init__(self, model_name: str = "resnet50", feature_layer: str = "avgpool"):
        """
        Initialize feature extractor.

        Args:
            model_name: Base model for feature extraction
            feature_layer: Layer to extract features from
        """
        self.model_name = model_name
        self.feature_layer = feature_layer
        self.model = self._load_model()
        self.features = {}  # Store intermediate features
        self._register_hooks()

        # Preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self) -> nn.Module:
        """Load pre-trained model."""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif self.model_name == "resnet101":
            model = models.resnet101(pretrained=True)
        elif self.model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.eval()
        model.to(DEVICE)
        return model

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""

        def hook_fn(module, input, output):
            self.features[self.feature_layer] = output.detach()

        # Find the target layer and register hook
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook_fn)
                break

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

            # Extract features
            with torch.no_grad():
                _ = self.model(input_tensor)
                features = self.features[self.feature_layer]

                # Flatten and normalize features
                features = features.view(features.size(0), -1)
                features = nn.functional.normalize(features, p=2, dim=1)

            return features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return np.array([])


class ImageSimilarityAnalyzer:
    """Analyze image similarity using deep features."""

    def __init__(self, feature_extractor: Optional[ImageFeatureExtractor] = None):
        """
        Initialize similarity analyzer.

        Args:
            feature_extractor: Pre-configured feature extractor
        """
        self.feature_extractor = feature_extractor or ImageFeatureExtractor()
        self.image_database = {}  # Store image features

    def add_image(self, image_path: str, image_id: Optional[str] = None) -> str:
        """
        Add an image to the similarity database.

        Args:
            image_path: Path to image file
            image_id: Optional custom ID for the image

        Returns:
            Image ID in the database
        """
        if image_id is None:
            image_id = hashlib.md5(image_path.encode()).hexdigest()

        features = self.feature_extractor.extract_features(image_path)
        if len(features) > 0:
            self.image_database[image_id] = {
                "path": image_path,
                "features": features,
                "added_at": datetime.now().isoformat(),
            }

        return image_id

    def find_similar_images(
        self, query_image_path: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find images similar to the query image.

        Args:
            query_image_path: Path to query image
            top_k: Number of similar images to return

        Returns:
            List of similar images with similarity scores
        """
        if not self.image_database:
            return []

        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image_path)
        if len(query_features) == 0:
            return []

        # Calculate similarities
        similarities = []
        for image_id, data in self.image_database.items():
            db_features = data["features"]
            similarity = cosine_similarity([query_features], [db_features])[0][0]

            similarities.append(
                {
                    "image_id": image_id,
                    "image_path": data["path"],
                    "similarity": float(similarity),
                    "added_at": data["added_at"],
                }
            )

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def calculate_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        Calculate similarity between two images.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        features1 = self.feature_extractor.extract_features(image1_path)
        features2 = self.feature_extractor.extract_features(image2_path)

        if len(features1) == 0 or len(features2) == 0:
            return 0.0

        return float(cosine_similarity([features1], [features2])[0][0])


class ImageClusteringAnalyzer:
    """Cluster images based on visual similarity."""

    def __init__(self, feature_extractor: Optional[ImageFeatureExtractor] = None):
        """
        Initialize clustering analyzer.

        Args:
            feature_extractor: Pre-configured feature extractor
        """
        self.feature_extractor = feature_extractor or ImageFeatureExtractor()

    def cluster_images(
        self, image_paths: List[str], n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster a collection of images.

        Args:
            image_paths: List of image file paths
            n_clusters: Number of clusters to create

        Returns:
            Clustering results with cluster assignments and centroids
        """
        # Extract features for all images
        features_list = []
        valid_paths = []

        for image_path in image_paths:
            features = self.feature_extractor.extract_features(image_path)
            if len(features) > 0:
                features_list.append(features)
                valid_paths.append(image_path)

        if len(features_list) < n_clusters:
            logger.warning(
                f"Not enough valid images ({len(features_list)}) for {n_clusters} clusters"
            )
            n_clusters = max(1, len(features_list))

        if len(features_list) == 0:
            return {"clusters": [], "error": "No valid features extracted"}

        # Perform clustering
        features_array = np.array(features_list)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_array)

        # Organize results
        clusters = {}
        for i, (path, label) in enumerate(zip(valid_paths, cluster_labels)):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []

            clusters[cluster_id].append(
                {
                    "image_path": path,
                    "distance_to_centroid": float(
                        np.linalg.norm(
                            features_list[i] - kmeans.cluster_centers_[label]
                        )
                    ),
                }
            )

        # Sort images within each cluster by distance to centroid
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: x["distance_to_centroid"])

        return {
            "n_clusters": n_clusters,
            "total_images": len(valid_paths),
            "clusters": clusters,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
        }


class AutoTaggingSystem:
    """Automated image tagging and categorization."""

    def __init__(self):
        """Initialize auto-tagging system."""
        # Predefined categories and associated keywords
        self.categories = {
            "nature": ["tree", "flower", "mountain", "beach", "forest", "sky", "water"],
            "people": ["person", "face", "human", "child", "adult", "group"],
            "animals": ["dog", "cat", "bird", "horse", "wildlife", "pet"],
            "objects": ["car", "building", "furniture", "tool", "device"],
            "food": ["meal", "fruit", "vegetable", "drink", "cooking"],
            "indoor": ["room", "furniture", "decoration", "interior"],
            "outdoor": ["landscape", "street", "garden", "park", "architecture"],
        }

        # Quality thresholds
        self.quality_thresholds = {
            "sharpness": 50.0,
            "brightness": (50, 200),
            "contrast": 30.0,
        }

    def generate_tags(
        self, image_path: str, detection_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate tags for an image based on detected objects and visual features.

        Args:
            image_path: Path to image file
            detection_results: Optional object detection results

        Returns:
            Generated tags and categories
        """
        tags = []
        categories = []
        confidence_scores = {}

        # Extract tags from detection results if provided
        if detection_results:
            for detection in detection_results:
                if "class" in detection:
                    obj_class = detection["class"].lower()
                    confidence = detection.get("confidence", 0.5)

                    tags.append(obj_class)
                    confidence_scores[obj_class] = confidence

                    # Map to categories
                    for category, keywords in self.categories.items():
                        if any(keyword in obj_class for keyword in keywords):
                            if category not in categories:
                                categories.append(category)

        # Analyze image properties for additional tags
        try:
            image = Image.open(image_path)
            width, height = image.size

            # Size-based tags
            if width > 1920 or height > 1080:
                tags.append("high-resolution")

            # Aspect ratio tags
            aspect_ratio = width / height
            if aspect_ratio > 1.5:
                tags.append("landscape-orientation")
            elif aspect_ratio < 0.7:
                tags.append("portrait-orientation")
            else:
                tags.append("square-orientation")

            # Color analysis
            if image.mode == "RGB":
                # Simple color dominance analysis
                pixels = np.array(image.resize((100, 100)))
                avg_color = np.mean(pixels, axis=(0, 1))

                if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                    tags.append("red-dominant")
                elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                    tags.append("green-dominant")
                elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                    tags.append("blue-dominant")

                # Brightness analysis
                brightness = np.mean(avg_color)
                if brightness > 180:
                    tags.append("bright")
                elif brightness < 80:
                    tags.append("dark")
                else:
                    tags.append("normal-brightness")

        except Exception as e:
            logger.error(f"Image analysis failed for tagging: {e}")

        return {
            "tags": list(set(tags)),  # Remove duplicates
            "categories": list(set(categories)),
            "confidence_scores": confidence_scores,
            "total_tags": len(set(tags)),
        }


class ImageQualityAssessment:
    """Assess image quality using various metrics."""

    def __init__(self):
        """Initialize quality assessment."""
        pass

    def assess_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Assess overall image quality.

        Args:
            image_path: Path to image file

        Returns:
            Quality assessment results
        """
        try:
            # Load image
            image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Calculate various quality metrics
            sharpness = self._calculate_sharpness(gray)
            brightness = self._calculate_brightness(image)
            contrast = self._calculate_contrast(gray)
            noise_level = self._estimate_noise(gray)

            # Overall quality score (0-100)
            quality_score = self._calculate_overall_score(
                sharpness, brightness, contrast, noise_level
            )

            return {
                "overall_score": quality_score,
                "metrics": {
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "contrast": contrast,
                    "noise_level": noise_level,
                },
                "assessment": self._get_quality_assessment(quality_score),
                "recommendations": self._get_recommendations(
                    sharpness, brightness, contrast, noise_level
                ),
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}

    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def _calculate_brightness(self, image: Image.Image) -> float:
        """Calculate average brightness."""
        stat = ImageStat.Stat(image)
        if image.mode == "RGB":
            return sum(stat.mean) / 3
        else:
            return stat.mean[0]

    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """Calculate image contrast using standard deviation."""
        return float(np.std(gray_image))

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level using high-frequency content."""
        # Simple noise estimation using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray_image, -1, kernel)
        return float(np.std(filtered))

    def _calculate_overall_score(
        self, sharpness: float, brightness: float, contrast: float, noise_level: float
    ) -> float:
        """Calculate overall quality score."""
        # Normalize metrics to 0-100 scale
        sharpness_score = min(sharpness / 100, 1.0) * 100
        brightness_score = max(0, 100 - abs(brightness - 128) / 128 * 100)
        contrast_score = min(contrast / 64, 1.0) * 100
        noise_score = max(0, 100 - min(noise_level / 50, 1.0) * 100)

        # Weighted average
        overall = (
            sharpness_score * 0.3
            + brightness_score * 0.2
            + contrast_score * 0.3
            + noise_score * 0.2
        )

        return min(100.0, max(0.0, overall))

    def _get_quality_assessment(self, score: float) -> str:
        """Get qualitative assessment."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"

    def _get_recommendations(
        self, sharpness: float, brightness: float, contrast: float, noise_level: float
    ) -> List[str]:
        """Get improvement recommendations."""
        recommendations = []

        if sharpness < 50:
            recommendations.append(
                "Image appears blurry - consider using image sharpening"
            )

        if brightness < 80 or brightness > 180:
            recommendations.append("Adjust brightness levels for better visibility")

        if contrast < 30:
            recommendations.append("Increase contrast to improve image definition")

        if noise_level > 40:
            recommendations.append("Apply noise reduction to clean up the image")

        if not recommendations:
            recommendations.append(
                "Image quality is good - no major improvements needed"
            )

        return recommendations


class ContentBasedImageRetrieval:
    """Complete content-based image retrieval system."""

    def __init__(self):
        """Initialize CBIR system."""
        self.feature_extractor = ImageFeatureExtractor()
        self.similarity_analyzer = ImageSimilarityAnalyzer(self.feature_extractor)
        self.clustering_analyzer = ImageClusteringAnalyzer(self.feature_extractor)
        self.auto_tagger = AutoTaggingSystem()
        self.quality_assessor = ImageQualityAssessment()

    def index_image_collection(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Index a collection of images for retrieval.

        Args:
            image_paths: List of image file paths

        Returns:
            Indexing results and statistics
        """
        indexed_count = 0
        failed_count = 0

        for image_path in image_paths:
            try:
                image_id = self.similarity_analyzer.add_image(image_path)
                if image_id:
                    indexed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to index {image_path}: {e}")
                failed_count += 1

        return {
            "total_images": len(image_paths),
            "indexed_successfully": indexed_count,
            "failed_to_index": failed_count,
            "index_size": len(self.similarity_analyzer.image_database),
        }

    def search_by_image(self, query_image_path: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search for similar images using an image as query.

        Args:
            query_image_path: Path to query image
            top_k: Number of results to return

        Returns:
            Search results with similarity scores
        """
        similar_images = self.similarity_analyzer.find_similar_images(
            query_image_path, top_k
        )

        return {
            "query_image": query_image_path,
            "results_count": len(similar_images),
            "similar_images": similar_images,
        }

    def analyze_image_collection(
        self, image_paths: List[str], n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of an image collection.

        Args:
            image_paths: List of image file paths
            n_clusters: Number of clusters for grouping

        Returns:
            Complete analysis results
        """
        # Cluster images
        clustering_results = self.clustering_analyzer.cluster_images(
            image_paths, n_clusters
        )

        # Quality assessment for each image
        quality_scores = []
        for image_path in image_paths:
            quality = self.quality_assessor.assess_quality(image_path)
            quality_scores.append(
                {
                    "image_path": image_path,
                    "quality_score": quality.get("overall_score", 0),
                    "assessment": quality.get("assessment", "unknown"),
                }
            )

        # Calculate collection statistics
        scores = [q["quality_score"] for q in quality_scores]
        collection_stats = {
            "total_images": len(image_paths),
            "average_quality": np.mean(scores) if scores else 0,
            "quality_std": np.std(scores) if scores else 0,
            "high_quality_count": sum(1 for s in scores if s >= 80),
            "low_quality_count": sum(1 for s in scores if s < 40),
        }

        return {
            "collection_statistics": collection_stats,
            "clustering_results": clustering_results,
            "quality_assessments": quality_scores,
        }


# Convenience functions
def find_similar_images(
    query_image_path: str, database_paths: List[str], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Quick similarity search."""
    analyzer = ImageSimilarityAnalyzer()
    for path in database_paths:
        analyzer.add_image(path)
    return analyzer.find_similar_images(query_image_path, top_k)


def cluster_images(image_paths: List[str], n_clusters: int = 5) -> Dict[str, Any]:
    """Quick image clustering."""
    analyzer = ImageClusteringAnalyzer()
    return analyzer.cluster_images(image_paths, n_clusters)


def assess_image_quality(image_path: str) -> Dict[str, Any]:
    """Quick quality assessment."""
    assessor = ImageQualityAssessment()
    return assessor.assess_quality(image_path)


def auto_tag_image(
    image_path: str, detection_results: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Quick auto-tagging."""
    tagger = AutoTaggingSystem()
    return tagger.generate_tags(image_path, detection_results)
