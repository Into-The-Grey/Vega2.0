"""
Vision-Language Model Integration
================================

Enhanced vision-language models for advanced image-text understanding,
zero-shot classification, and cross-modal reasoning capabilities.

Features:
- Advanced CLIP model integration with multiple model variants
- Zero-shot image classification with confidence scoring
- Image captioning with context-aware generation
- Cross-modal retrieval and sophisticated ranking
- Multi-scale image analysis and feature extraction
- Batch processing for improved performance
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CLIPModelVariant(Enum):
    """Supported CLIP model variants"""

    VIT_B_32 = "ViT-B/32"
    VIT_B_16 = "ViT-B/16"
    VIT_L_14 = "ViT-L/14"
    VIT_L_14_336 = "ViT-L/14@336px"
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"


class ImageProcessingMode(Enum):
    """Image processing approaches"""

    SINGLE_SCALE = "single_scale"
    MULTI_SCALE = "multi_scale"
    CROP_ENSEMBLE = "crop_ensemble"
    ATTENTION_POOLING = "attention_pooling"


@dataclass
class CLIPConfig:
    """Configuration for CLIP model integration"""

    model_variant: CLIPModelVariant = CLIPModelVariant.VIT_B_32
    device: str = "cpu"  # "cuda" if available
    batch_size: int = 32
    image_size: int = 224
    processing_mode: ImageProcessingMode = ImageProcessingMode.SINGLE_SCALE
    use_fp16: bool = False
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    max_cache_size: int = 10000


@dataclass
class ImageAnalysisResult:
    """Results from image analysis"""

    image_id: str
    embeddings: np.ndarray
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    detected_objects: List[str] = field(default_factory=list)
    scene_description: str = ""
    dominant_colors: List[str] = field(default_factory=list)
    text_regions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZeroShotClassificationResult:
    """Results from zero-shot classification"""

    image_id: str
    predictions: List[Tuple[str, float]]  # (class_name, confidence)
    top_prediction: str
    top_confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0


import asyncio
import logging
import numpy as np
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import base64
import io

# Optional imports for advanced functionality
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import requests

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import transformers
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        CLIPTokenizer,
        BlipProcessor,
        BlipForConditionalGeneration,
        AutoProcessor,
        AutoModel,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisionLanguageTask(Enum):
    """Supported vision-language tasks"""

    IMAGE_TEXT_SIMILARITY = "image_text_similarity"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    OBJECT_DETECTION = "object_detection"
    SCENE_UNDERSTANDING = "scene_understanding"
    TEXT_TO_IMAGE_RETRIEVAL = "text_to_image_retrieval"
    IMAGE_TO_TEXT_RETRIEVAL = "image_to_text_retrieval"


class ModelType(Enum):
    """Available vision-language model types"""

    CLIP = "clip"
    BLIP = "blip"
    ALIGN = "align"
    FLAMINGO = "flamingo"
    CUSTOM = "custom"


@dataclass
class VisionLanguageConfig:
    """Configuration for vision-language models"""

    model_type: ModelType = ModelType.CLIP
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"  # auto, cpu, cuda
    cache_dir: Optional[str] = None
    max_length: int = 77  # Max token length for text
    image_size: int = 224  # Standard image size
    batch_size: int = 8
    temperature: float = 0.07  # Temperature for similarity computation
    enable_caching: bool = True


@dataclass
class ImageAnalysis:
    """Results from image analysis"""

    image_path: str
    image_embedding: Optional[np.ndarray] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    scene_description: str = ""
    visual_features: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextAnalysis:
    """Results from text analysis"""

    text: str
    text_embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    language_info: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class CrossModalResult:
    """Results from cross-modal operations"""

    similarity_score: float
    confidence: float
    explanation: str
    detailed_scores: Dict[str, float] = field(default_factory=dict)
    visual_attention: Optional[np.ndarray] = None
    text_attention: Optional[List[float]] = None


class VisionLanguageModel:
    """
    Base class for vision-language models providing unified interface
    for different model types and tasks.
    """

    def __init__(self, config: VisionLanguageConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = self._get_device()

        # Performance tracking
        self.call_count = 0
        self.total_processing_time = 0.0

        logger.info(f"Initializing VisionLanguageModel: {config.model_type.value}")

    def _get_device(self) -> str:
        """Determine the appropriate device"""
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device

    async def initialize(self):
        """Initialize the model asynchronously"""
        try:
            if self.config.model_type == ModelType.CLIP:
                await self._init_clip()
            elif self.config.model_type == ModelType.BLIP:
                await self._init_blip()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            logger.info(
                f"Successfully initialized {self.config.model_type.value} model"
            )

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    async def _init_clip(self):
        """Initialize CLIP model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for CLIP")

        self.model = CLIPModel.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )

        if self.device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()

    async def _init_blip(self):
        """Initialize BLIP model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for BLIP")

        model_name = self.config.model_name or "Salesforce/blip-image-captioning-base"

        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, cache_dir=self.config.cache_dir
        )
        self.processor = BlipProcessor.from_pretrained(
            model_name, cache_dir=self.config.cache_dir
        )

        if self.device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()

    async def analyze_image(self, image_path: str) -> ImageAnalysis:
        """Analyze an image and extract visual features"""
        start_time = datetime.now()

        try:
            # Load and preprocess image
            image = self._load_image(image_path)

            # Extract features based on model type
            if self.config.model_type == ModelType.CLIP:
                embedding = await self._clip_image_embedding(image)
            elif self.config.model_type == ModelType.BLIP:
                embedding = await self._blip_image_embedding(image)
            else:
                embedding = None

            # Basic image analysis
            visual_features = self._extract_visual_features(image)

            # Create analysis result
            processing_time = (datetime.now() - start_time).total_seconds()

            analysis = ImageAnalysis(
                image_path=image_path,
                image_embedding=embedding,
                visual_features=visual_features,
                processing_time=processing_time,
                metadata={"model_type": self.config.model_type.value},
            )

            self._update_stats(processing_time)
            return analysis

        except Exception as e:
            logger.error(f"Image analysis failed for {image_path}: {e}")
            raise

    async def analyze_text(self, text: str) -> TextAnalysis:
        """Analyze text and extract semantic features"""
        start_time = datetime.now()

        try:
            # Extract text embedding
            if self.config.model_type == ModelType.CLIP:
                embedding = await self._clip_text_embedding(text)
            elif self.config.model_type == ModelType.BLIP:
                embedding = await self._blip_text_embedding(text)
            else:
                embedding = None

            # Extract text features
            keywords = self._extract_keywords(text)
            semantic_features = self._extract_semantic_features(text)
            language_info = self._analyze_language(text)

            processing_time = (datetime.now() - start_time).total_seconds()

            analysis = TextAnalysis(
                text=text,
                text_embedding=embedding,
                keywords=keywords,
                semantic_features=semantic_features,
                language_info=language_info,
                processing_time=processing_time,
            )

            self._update_stats(processing_time)
            return analysis

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise

    async def compute_similarity(self, image_path: str, text: str) -> CrossModalResult:
        """Compute similarity between image and text"""
        try:
            # Analyze both modalities
            image_analysis = await self.analyze_image(image_path)
            text_analysis = await self.analyze_text(text)

            # Compute similarity
            similarity_score = 0.0
            detailed_scores = {}

            if (
                image_analysis.image_embedding is not None
                and text_analysis.text_embedding is not None
            ):
                # Normalize embeddings
                img_emb = image_analysis.image_embedding / np.linalg.norm(
                    image_analysis.image_embedding
                )
                txt_emb = text_analysis.text_embedding / np.linalg.norm(
                    text_analysis.text_embedding
                )

                # Compute cosine similarity
                similarity_score = float(np.dot(img_emb, txt_emb))

                # Apply temperature scaling
                similarity_score = similarity_score / self.config.temperature

                detailed_scores["embedding_similarity"] = similarity_score

            # Additional similarity metrics
            if image_analysis.visual_features and text_analysis.keywords:
                # Simple keyword-visual feature matching
                visual_concepts = self._extract_visual_concepts(
                    image_analysis.visual_features
                )
                keyword_match_score = self._compute_keyword_visual_match(
                    text_analysis.keywords, visual_concepts
                )
                detailed_scores["keyword_visual_match"] = keyword_match_score

            # Generate explanation
            explanation = self._generate_similarity_explanation(
                similarity_score, detailed_scores, text_analysis.keywords
            )

            # Compute confidence
            confidence = min(1.0, abs(similarity_score))

            return CrossModalResult(
                similarity_score=similarity_score,
                confidence=confidence,
                explanation=explanation,
                detailed_scores=detailed_scores,
            )

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise

    async def classify_image(
        self, image_path: str, class_labels: List[str]
    ) -> Dict[str, float]:
        """Zero-shot image classification using text labels"""
        try:
            if self.config.model_type != ModelType.CLIP:
                raise ValueError("Zero-shot classification only supported with CLIP")

            # Load image
            image = self._load_image(image_path)

            # Prepare text labels
            text_inputs = [f"a photo of a {label}" for label in class_labels]

            # Process inputs
            inputs = self.processor(
                text=text_inputs, images=image, return_tensors="pt", padding=True
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Compute predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Create results dictionary
            results = {}
            for i, label in enumerate(class_labels):
                results[label] = float(probs[0][i])

            return results

        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            raise

    async def caption_image(self, image_path: str) -> str:
        """Generate caption for an image"""
        try:
            if self.config.model_type == ModelType.BLIP:
                return await self._blip_caption_image(image_path)
            else:
                # Fallback to simple description
                analysis = await self.analyze_image(image_path)
                return self._generate_simple_caption(analysis)

        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return "Failed to generate caption"

    async def _clip_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP image embedding"""
        inputs = self.processor(images=image, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features.cpu().numpy()[0]

    async def _clip_text_embedding(self, text: str) -> np.ndarray:
        """Extract CLIP text embedding"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features.cpu().numpy()[0]

    async def _blip_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract BLIP image embedding"""
        inputs = self.processor(image, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            image_embeds = self.model.vision_model(**inputs).last_hidden_state
            # Use mean pooling
            embedding = image_embeds.mean(dim=1)

        return embedding.cpu().numpy()[0]

    async def _blip_text_embedding(self, text: str) -> np.ndarray:
        """Extract BLIP text embedding"""
        inputs = self.processor(text=text, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            text_embeds = self.model.text_encoder(**inputs).last_hidden_state
            # Use mean pooling
            embedding = text_embeds.mean(dim=1)

        return embedding.cpu().numpy()[0]

    async def _blip_caption_image(self, image_path: str) -> str:
        """Generate image caption using BLIP"""
        image = self._load_image(image_path)
        inputs = self.processor(image, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        if isinstance(image_path, str):
            if image_path.startswith("http"):
                # Download image
                response = requests.get(image_path)
                image = Image.open(io.BytesIO(response.content))
            else:
                # Local file
                image = Image.open(image_path)
        else:
            image = image_path

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic visual features from image"""
        # Convert to numpy array
        img_array = np.array(image)

        features = {}

        # Color statistics
        features["color_stats"] = {
            "mean_r": float(np.mean(img_array[:, :, 0])),
            "mean_g": float(np.mean(img_array[:, :, 1])),
            "mean_b": float(np.mean(img_array[:, :, 2])),
            "std_r": float(np.std(img_array[:, :, 0])),
            "std_g": float(np.std(img_array[:, :, 1])),
            "std_b": float(np.std(img_array[:, :, 2])),
        }

        # Image dimensions
        features["dimensions"] = {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": image.width / image.height,
        }

        # Brightness and contrast
        gray = np.mean(img_array, axis=2)
        features["brightness"] = float(np.mean(gray))
        features["contrast"] = float(np.std(gray))

        return features

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified)"""
        # Simple keyword extraction
        words = text.lower().split()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Return unique keywords
        return list(set(keywords))

    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text"""
        features = {}

        # Text statistics
        features["length"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len([s for s in text.split(".") if s.strip()])

        # Simple sentiment analysis (placeholder)
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "beautiful",
            "nice",
            "love",
            "like",
            "happy",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "ugly",
            "hate",
            "dislike",
            "sad",
            "angry",
            "disappointed",
        ]

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        features["sentiment"] = {
            "positive_score": positive_count,
            "negative_score": negative_count,
            "polarity": positive_count - negative_count,
        }

        return features

    def _analyze_language(self, text: str) -> Dict[str, Any]:
        """Analyze language characteristics"""
        info = {}

        # Character analysis
        info["char_count"] = len(text)
        info["uppercase_ratio"] = (
            sum(1 for c in text if c.isupper()) / len(text) if text else 0
        )
        info["digit_ratio"] = (
            sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        )
        info["punctuation_ratio"] = (
            sum(1 for c in text if c in ".,!?;:") / len(text) if text else 0
        )

        # Simple language detection (placeholder)
        info["language"] = "en"  # Default to English

        return info

    def _extract_visual_concepts(self, visual_features: Dict[str, Any]) -> List[str]:
        """Extract visual concepts from features"""
        concepts = []

        # Color-based concepts
        color_stats = visual_features.get("color_stats", {})
        if color_stats:
            mean_r = color_stats.get("mean_r", 0)
            mean_g = color_stats.get("mean_g", 0)
            mean_b = color_stats.get("mean_b", 0)

            if mean_r > 150 and mean_g < 100 and mean_b < 100:
                concepts.append("red")
            elif mean_g > 150 and mean_r < 100 and mean_b < 100:
                concepts.append("green")
            elif mean_b > 150 and mean_r < 100 and mean_g < 100:
                concepts.append("blue")

        # Brightness-based concepts
        brightness = visual_features.get("brightness", 0)
        if brightness > 200:
            concepts.append("bright")
        elif brightness < 50:
            concepts.append("dark")

        # Contrast-based concepts
        contrast = visual_features.get("contrast", 0)
        if contrast > 50:
            concepts.append("high_contrast")
        elif contrast < 20:
            concepts.append("low_contrast")

        return concepts

    def _compute_keyword_visual_match(
        self, keywords: List[str], visual_concepts: List[str]
    ) -> float:
        """Compute match between keywords and visual concepts"""
        if not keywords or not visual_concepts:
            return 0.0

        # Simple string matching
        matches = 0
        for keyword in keywords:
            for concept in visual_concepts:
                if (
                    keyword.lower() in concept.lower()
                    or concept.lower() in keyword.lower()
                ):
                    matches += 1
                    break

        return matches / len(keywords)

    def _generate_similarity_explanation(
        self,
        similarity_score: float,
        detailed_scores: Dict[str, float],
        keywords: List[str],
    ) -> str:
        """Generate human-readable explanation for similarity"""
        explanations = []

        # Overall similarity
        if similarity_score > 0.7:
            explanations.append("Strong semantic alignment between image and text")
        elif similarity_score > 0.4:
            explanations.append("Moderate semantic similarity")
        elif similarity_score > 0.1:
            explanations.append("Weak semantic connection")
        else:
            explanations.append("Low semantic similarity")

        # Specific matching factors
        if "keyword_visual_match" in detailed_scores:
            match_score = detailed_scores["keyword_visual_match"]
            if match_score > 0.5:
                explanations.append(f"Good keyword-visual concept alignment")
            elif match_score > 0.0:
                explanations.append(f"Some keyword-visual concept matching")

        # Mention key concepts
        if keywords:
            key_concepts = ", ".join(keywords[:3])
            explanations.append(f"Key concepts: {key_concepts}")

        return "; ".join(explanations)

    def _generate_simple_caption(self, analysis: ImageAnalysis) -> str:
        """Generate simple caption from image analysis"""
        caption_parts = []

        # Brightness description
        brightness = analysis.visual_features.get("brightness", 0)
        if brightness > 200:
            caption_parts.append("bright")
        elif brightness < 50:
            caption_parts.append("dark")

        # Color description
        color_stats = analysis.visual_features.get("color_stats", {})
        if color_stats:
            mean_r = color_stats.get("mean_r", 0)
            mean_g = color_stats.get("mean_g", 0)
            mean_b = color_stats.get("mean_b", 0)

            if mean_r > max(mean_g, mean_b) + 50:
                caption_parts.append("reddish")
            elif mean_g > max(mean_r, mean_b) + 50:
                caption_parts.append("greenish")
            elif mean_b > max(mean_r, mean_g) + 50:
                caption_parts.append("bluish")

        # Dimensions
        dims = analysis.visual_features.get("dimensions", {})
        aspect_ratio = dims.get("aspect_ratio", 1.0)
        if aspect_ratio > 1.5:
            caption_parts.append("wide")
        elif aspect_ratio < 0.7:
            caption_parts.append("tall")

        if caption_parts:
            return f"A {' '.join(caption_parts)} image"
        else:
            return "An image"

    def _update_stats(self, processing_time: float):
        """Update performance statistics"""
        self.call_count += 1
        self.total_processing_time += processing_time

    def get_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        avg_time = (
            self.total_processing_time / self.call_count if self.call_count > 0 else 0
        )

        return {
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
            "device": self.device,
            "call_count": self.call_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
        }


class CLIPIntegration:
    """
    Specialized CLIP integration with advanced features for image-text understanding.
    Provides convenient methods for common CLIP-based tasks.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.config = VisionLanguageConfig(
            model_type=ModelType.CLIP, model_name=model_name
        )
        self.model = VisionLanguageModel(self.config)
        self.initialized = False

    async def initialize(self):
        """Initialize CLIP model"""
        await self.model.initialize()
        self.initialized = True
        logger.info("CLIP integration initialized")

    async def find_similar_images(
        self, text_query: str, image_paths: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find images most similar to text query"""
        if not self.initialized:
            await self.initialize()

        results = []

        for image_path in image_paths:
            try:
                result = await self.model.compute_similarity(image_path, text_query)
                results.append((image_path, result.similarity_score))
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_similar_texts(
        self, image_path: str, text_candidates: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find texts most similar to image"""
        if not self.initialized:
            await self.initialize()

        results = []

        for text in text_candidates:
            try:
                result = await self.model.compute_similarity(image_path, text)
                results.append((text, result.similarity_score))
            except Exception as e:
                logger.warning(f"Failed to process text: {e}")

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def classify_images(
        self, image_paths: List[str], class_labels: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Classify multiple images with given labels"""
        if not self.initialized:
            await self.initialize()

        results = {}

        for image_path in image_paths:
            try:
                classifications = await self.model.classify_image(
                    image_path, class_labels
                )
                results[image_path] = classifications
            except Exception as e:
                logger.warning(f"Failed to classify {image_path}: {e}")
                results[image_path] = {}

        return results

    async def batch_similarity(
        self, image_text_pairs: List[Tuple[str, str]]
    ) -> List[CrossModalResult]:
        """Compute similarity for multiple image-text pairs"""
        if not self.initialized:
            await self.initialize()

        results = []

        for image_path, text in image_text_pairs:
            try:
                result = await self.model.compute_similarity(image_path, text)
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Failed to process pair ({image_path}, {text[:50]}...): {e}"
                )
                # Add empty result
                results.append(
                    CrossModalResult(
                        similarity_score=0.0,
                        confidence=0.0,
                        explanation="Processing failed",
                    )
                )

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if self.initialized:
            return self.model.get_stats()
        else:
            return {"status": "not_initialized"}
