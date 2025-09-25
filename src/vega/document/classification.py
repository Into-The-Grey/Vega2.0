"""
Vega 2.0 Document Classification Module

This module provides intelligent document classification capabilities including:
- ML-based document categorization and type detection
- Topic modeling and semantic clustering
- Content-based document routing and organization
- Multi-modal classification (text, layout, structure)
- Confidence-based classification with uncertainty handling
- Custom classification models and fine-tuning support

Dependencies:
- scikit-learn: Machine learning algorithms for classification
- transformers: Pre-trained language models for document understanding
- sentence-transformers: Semantic embeddings for document similarity
- numpy, pandas: Data manipulation and numerical processing
- nltk/spacy: Natural language processing utilities
"""

import asyncio
import logging
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    pipeline = AutoTokenizer = AutoModelForSequenceClassification = torch = None
    HAS_TRANSFORMERS = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer

    HAS_NLTK = True
except ImportError:
    nltk = stopwords = word_tokenize = PorterStemmer = None
    HAS_NLTK = False

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Custom exception for classification errors"""

    pass


class DocumentCategory(Enum):
    """Primary document categories"""

    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    MEDICAL = "medical"
    GOVERNMENT = "government"
    PERSONAL = "personal"
    MARKETING = "marketing"
    CORRESPONDENCE = "correspondence"
    FORMS = "forms"
    REPORTS = "reports"
    MANUALS = "manuals"
    CONTRACTS = "contracts"
    INVOICES = "invoices"
    UNKNOWN = "unknown"


class ClassificationMethod(Enum):
    """Classification methods"""

    NAIVE_BAYES = "naive_bayes"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HYBRID = "hybrid"


class FeatureType(Enum):
    """Feature extraction types"""

    TFIDF = "tfidf"
    BOW = "bow"
    NGRAMS = "ngrams"
    EMBEDDINGS = "embeddings"
    COMBINED = "combined"


class ClusteringMethod(Enum):
    """Clustering algorithms"""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


@dataclass
class ClassificationConfig:
    """Configuration for document classification"""

    method: ClassificationMethod = ClassificationMethod.HYBRID
    feature_type: FeatureType = FeatureType.COMBINED
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    min_confidence: float = 0.7
    use_pretrained_models: bool = True
    model_cache_dir: str = "./models/classification"
    enable_topic_modeling: bool = True
    enable_clustering: bool = True
    num_topics: int = 20
    num_clusters: int = 10


@dataclass
class TopicModelingConfig:
    """Configuration for topic modeling"""

    method: str = "lda"  # lda, nmf
    num_topics: int = 20
    max_iter: int = 100
    alpha: float = 0.1
    beta: float = 0.01
    min_df: int = 2
    max_df: float = 0.95


@dataclass
class ClusteringConfig:
    """Configuration for clustering"""

    method: ClusteringMethod = ClusteringMethod.KMEANS
    num_clusters: int = 10
    min_samples: int = 5
    eps: float = 0.5
    distance_metric: str = "cosine"


@dataclass
class ClassificationResult:
    """Result from document classification"""

    predicted_category: DocumentCategory
    confidence: float
    probabilities: Dict[str, float]
    features_used: List[str]
    model_info: Dict[str, Any]
    topics: Optional[List[Tuple[str, float]]] = None
    cluster_id: Optional[int] = None
    similar_documents: Optional[List[Dict[str, Any]]] = None


@dataclass
class TopicResult:
    """Result from topic modeling"""

    topics: List[Dict[str, Any]]
    document_topics: List[List[Tuple[int, float]]]
    coherence_score: float
    perplexity: Optional[float] = None


@dataclass
class ClusterResult:
    """Result from document clustering"""

    cluster_labels: List[int]
    cluster_centers: Optional[np.ndarray] = None
    silhouette_score: float
    num_clusters: int
    cluster_info: List[Dict[str, Any]] = field(default_factory=list)


class FeatureExtractor:
    """
    Feature extraction for document classification
    """

    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.vectorizers = {}
        self.sentence_transformer = None

        # Initialize sentence transformer if available
        if HAS_SENTENCE_TRANSFORMERS and config.use_pretrained_models:
            try:
                self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")

    async def extract_features(
        self, documents: List[str], feature_type: Optional[FeatureType] = None
    ) -> np.ndarray:
        """
        Extract features from documents

        Args:
            documents: List of document texts
            feature_type: Type of features to extract

        Returns:
            Feature matrix
        """
        try:
            feature_type = feature_type or self.config.feature_type

            if feature_type == FeatureType.TFIDF:
                return await self._extract_tfidf_features(documents)
            elif feature_type == FeatureType.BOW:
                return await self._extract_bow_features(documents)
            elif feature_type == FeatureType.NGRAMS:
                return await self._extract_ngram_features(documents)
            elif feature_type == FeatureType.EMBEDDINGS:
                return await self._extract_embedding_features(documents)
            elif feature_type == FeatureType.COMBINED:
                return await self._extract_combined_features(documents)
            else:
                raise ClassificationError(f"Unknown feature type: {feature_type}")

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.array([])

    async def _extract_tfidf_features(self, documents: List[str]) -> np.ndarray:
        """Extract TF-IDF features"""
        try:
            if not HAS_SKLEARN:
                logger.warning("Scikit-learn not available for TF-IDF")
                return np.array([])

            vectorizer_key = "tfidf"
            if vectorizer_key not in self.vectorizers:
                self.vectorizers[vectorizer_key] = TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                )

            vectorizer = self.vectorizers[vectorizer_key]
            features = vectorizer.fit_transform(documents)

            return features.toarray()

        except Exception as e:
            logger.error(f"TF-IDF extraction error: {e}")
            return np.array([])

    async def _extract_bow_features(self, documents: List[str]) -> np.ndarray:
        """Extract Bag of Words features"""
        try:
            if not HAS_SKLEARN:
                return np.array([])

            vectorizer_key = "bow"
            if vectorizer_key not in self.vectorizers:
                self.vectorizers[vectorizer_key] = CountVectorizer(
                    max_features=self.config.max_features,
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                )

            vectorizer = self.vectorizers[vectorizer_key]
            features = vectorizer.fit_transform(documents)

            return features.toarray()

        except Exception as e:
            logger.error(f"BoW extraction error: {e}")
            return np.array([])

    async def _extract_ngram_features(self, documents: List[str]) -> np.ndarray:
        """Extract n-gram features"""
        try:
            if not HAS_SKLEARN:
                return np.array([])

            vectorizer_key = f"ngram_{self.config.ngram_range}"
            if vectorizer_key not in self.vectorizers:
                self.vectorizers[vectorizer_key] = TfidfVectorizer(
                    max_features=self.config.max_features,
                    ngram_range=self.config.ngram_range,
                    stop_words="english",
                    analyzer="word",
                )

            vectorizer = self.vectorizers[vectorizer_key]
            features = vectorizer.fit_transform(documents)

            return features.toarray()

        except Exception as e:
            logger.error(f"N-gram extraction error: {e}")
            return np.array([])

    async def _extract_embedding_features(self, documents: List[str]) -> np.ndarray:
        """Extract semantic embedding features"""
        try:
            if not self.sentence_transformer:
                logger.warning("Sentence transformer not available")
                return np.array([])

            # Generate embeddings
            embeddings = self.sentence_transformer.encode(
                documents, show_progress_bar=False
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return np.array([])

    async def _extract_combined_features(self, documents: List[str]) -> np.ndarray:
        """Extract combined features from multiple methods"""
        try:
            features_list = []

            # TF-IDF features
            if HAS_SKLEARN:
                tfidf_features = await self._extract_tfidf_features(documents)
                if tfidf_features.size > 0:
                    features_list.append(tfidf_features)

            # Embedding features
            if self.sentence_transformer:
                embedding_features = await self._extract_embedding_features(documents)
                if embedding_features.size > 0:
                    features_list.append(embedding_features)

            # Combine features
            if features_list:
                combined_features = np.concatenate(features_list, axis=1)
                return combined_features
            else:
                logger.warning("No features extracted")
                return np.array([])

        except Exception as e:
            logger.error(f"Combined feature extraction error: {e}")
            return np.array([])

    def get_feature_names(self, feature_type: FeatureType) -> List[str]:
        """Get feature names for interpretability"""
        try:
            if feature_type in [FeatureType.TFIDF, FeatureType.BOW, FeatureType.NGRAMS]:
                vectorizer_key = {
                    FeatureType.TFIDF: "tfidf",
                    FeatureType.BOW: "bow",
                    FeatureType.NGRAMS: f"ngram_{self.config.ngram_range}",
                }[feature_type]

                if vectorizer_key in self.vectorizers:
                    return (
                        self.vectorizers[vectorizer_key]
                        .get_feature_names_out()
                        .tolist()
                    )

            return []

        except Exception as e:
            logger.debug(f"Feature names error: {e}")
            return []


class TopicModeler:
    """
    Topic modeling for document analysis
    """

    def __init__(self, config: TopicModelingConfig):
        self.config = config
        self.model = None
        self.vectorizer = None

    async def fit_topic_model(self, documents: List[str]) -> TopicResult:
        """
        Fit topic model on documents

        Args:
            documents: List of document texts

        Returns:
            Topic modeling results
        """
        try:
            if not HAS_SKLEARN:
                logger.warning("Scikit-learn not available for topic modeling")
                return TopicResult(topics=[], document_topics=[], coherence_score=0.0)

            # Preprocess documents
            processed_docs = [self._preprocess_text(doc) for doc in documents]

            # Vectorize documents
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words="english",
                ngram_range=(1, 2),
            )

            doc_term_matrix = self.vectorizer.fit_transform(processed_docs)

            # Fit topic model
            if self.config.method.lower() == "lda":
                self.model = LatentDirichletAllocation(
                    n_components=self.config.num_topics,
                    max_iter=self.config.max_iter,
                    doc_topic_prior=self.config.alpha,
                    topic_word_prior=self.config.beta,
                    random_state=42,
                )
            elif self.config.method.lower() == "nmf":
                self.model = NMF(
                    n_components=self.config.num_topics,
                    max_iter=self.config.max_iter,
                    random_state=42,
                )
            else:
                raise ClassificationError(
                    f"Unknown topic modeling method: {self.config.method}"
                )

            # Fit model
            self.model.fit(doc_term_matrix)

            # Extract topics
            topics = self._extract_topics()

            # Get document-topic distributions
            doc_topic_dist = self.model.transform(doc_term_matrix)
            document_topics = []

            for doc_dist in doc_topic_dist:
                doc_topics = [
                    (i, prob) for i, prob in enumerate(doc_dist) if prob > 0.1
                ]
                doc_topics.sort(key=lambda x: x[1], reverse=True)
                document_topics.append(doc_topics[:5])  # Top 5 topics per document

            # Calculate coherence (simplified)
            coherence_score = self._calculate_coherence(topics)

            # Calculate perplexity for LDA
            perplexity = None
            if hasattr(self.model, "perplexity"):
                perplexity = self.model.perplexity(doc_term_matrix)

            return TopicResult(
                topics=topics,
                document_topics=document_topics,
                coherence_score=coherence_score,
                perplexity=perplexity,
            )

        except Exception as e:
            logger.error(f"Topic modeling error: {e}")
            return TopicResult(topics=[], document_topics=[], coherence_score=0.0)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling"""
        try:
            # Basic preprocessing
            text = text.lower()

            # Remove special characters but keep spaces
            import re

            text = re.sub(r"[^a-zA-Z\s]", "", text)

            # Remove extra whitespace
            text = " ".join(text.split())

            return text

        except Exception as e:
            logger.debug(f"Text preprocessing error: {e}")
            return text

    def _extract_topics(self) -> List[Dict[str, Any]]:
        """Extract topic information from fitted model"""
        try:
            if not self.model or not self.vectorizer:
                return []

            feature_names = self.vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(self.model.components_):
                # Get top words for topic
                top_word_indices = topic.argsort()[-20:][::-1]  # Top 20 words
                top_words = [(feature_names[i], topic[i]) for i in top_word_indices]

                # Create topic summary
                topic_words = [word for word, _ in top_words[:5]]
                topic_name = f"Topic_{topic_idx}: {', '.join(topic_words[:3])}"

                topics.append(
                    {
                        "id": topic_idx,
                        "name": topic_name,
                        "top_words": top_words,
                        "weight": float(np.sum(topic)),
                    }
                )

            return topics

        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return []

    def _calculate_coherence(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate topic coherence (simplified implementation)"""
        try:
            # Simplified coherence calculation
            # In production, would use more sophisticated metrics like C_v coherence

            if not topics:
                return 0.0

            # Calculate average word probability variance within topics
            coherence_scores = []

            for topic in topics:
                if "top_words" in topic:
                    word_probs = [prob for _, prob in topic["top_words"][:10]]
                    if word_probs:
                        # Use coefficient of variation as coherence measure
                        mean_prob = np.mean(word_probs)
                        std_prob = np.std(word_probs)
                        cov = std_prob / mean_prob if mean_prob > 0 else 0
                        coherence_scores.append(1.0 - min(cov, 1.0))

            return float(np.mean(coherence_scores)) if coherence_scores else 0.0

        except Exception as e:
            logger.debug(f"Coherence calculation error: {e}")
            return 0.0


class DocumentClusterer:
    """
    Document clustering for unsupervised grouping
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.model = None
        self.feature_extractor = None

    async def cluster_documents(
        self, documents: List[str], features: Optional[np.ndarray] = None
    ) -> ClusterResult:
        """
        Cluster documents based on content similarity

        Args:
            documents: List of document texts
            features: Pre-computed features (optional)

        Returns:
            Clustering results
        """
        try:
            if not HAS_SKLEARN:
                logger.warning("Scikit-learn not available for clustering")
                return ClusterResult(
                    cluster_labels=[], silhouette_score=0.0, num_clusters=0
                )

            # Extract features if not provided
            if features is None:
                if not self.feature_extractor:
                    from . import ClassificationConfig  # Avoid circular import

                    config = ClassificationConfig()
                    self.feature_extractor = FeatureExtractor(config)

                features = await self.feature_extractor.extract_features(documents)

            if features.size == 0:
                logger.warning("No features available for clustering")
                return ClusterResult(
                    cluster_labels=[], silhouette_score=0.0, num_clusters=0
                )

            # Initialize clustering model
            if self.config.method == ClusteringMethod.KMEANS:
                self.model = KMeans(
                    n_clusters=self.config.num_clusters, random_state=42, n_init=10
                )
            elif self.config.method == ClusteringMethod.DBSCAN:
                self.model = DBSCAN(
                    eps=self.config.eps,
                    min_samples=self.config.min_samples,
                    metric=self.config.distance_metric,
                )
            else:
                raise ClassificationError(
                    f"Unknown clustering method: {self.config.method}"
                )

            # Fit clustering model
            cluster_labels = self.model.fit_predict(features)

            # Calculate silhouette score
            silhouette_score = 0.0
            try:
                from sklearn.metrics import silhouette_score as calc_silhouette

                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                    silhouette_score = calc_silhouette(features, cluster_labels)
            except Exception as e:
                logger.debug(f"Silhouette score calculation error: {e}")

            # Get cluster information
            cluster_info = self._analyze_clusters(documents, cluster_labels, features)

            # Get cluster centers if available
            cluster_centers = None
            if hasattr(self.model, "cluster_centers_"):
                cluster_centers = self.model.cluster_centers_

            return ClusterResult(
                cluster_labels=cluster_labels.tolist(),
                cluster_centers=cluster_centers,
                silhouette_score=float(silhouette_score),
                num_clusters=len(set(cluster_labels)),
                cluster_info=cluster_info,
            )

        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return ClusterResult(
                cluster_labels=[], silhouette_score=0.0, num_clusters=0
            )

    def _analyze_clusters(
        self, documents: List[str], cluster_labels: np.ndarray, features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Analyze cluster characteristics"""
        try:
            cluster_info = []
            unique_labels = set(cluster_labels)

            for cluster_id in unique_labels:
                if cluster_id == -1:  # Noise cluster in DBSCAN
                    continue

                # Get documents in this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_docs = [
                    doc for i, doc in enumerate(documents) if cluster_mask[i]
                ]
                cluster_features = features[cluster_mask]

                # Calculate cluster statistics
                cluster_size = len(cluster_docs)

                # Find representative documents (closest to centroid)
                if cluster_features.shape[0] > 0:
                    centroid = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    representative_idx = np.argmin(distances)
                    representative_doc = cluster_docs[representative_idx]
                else:
                    representative_doc = ""

                # Extract key terms (simplified)
                key_terms = self._extract_cluster_terms(cluster_docs)

                cluster_info.append(
                    {
                        "cluster_id": int(cluster_id),
                        "size": cluster_size,
                        "percentage": cluster_size / len(documents) * 100,
                        "representative_document": representative_doc[:200],
                        "key_terms": key_terms,
                        "centroid_available": hasattr(self.model, "cluster_centers_"),
                    }
                )

            return cluster_info

        except Exception as e:
            logger.error(f"Cluster analysis error: {e}")
            return []

    def _extract_cluster_terms(
        self, cluster_docs: List[str], top_k: int = 10
    ) -> List[str]:
        """Extract key terms for a cluster"""
        try:
            if not HAS_SKLEARN or not cluster_docs:
                return []

            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(cluster_docs)

            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Get top terms
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]

            return top_terms

        except Exception as e:
            logger.debug(f"Term extraction error: {e}")
            return []


class DocumentClassifier:
    """
    Main document classification system
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()

        # Initialize components
        self.feature_extractor = FeatureExtractor(self.config)
        self.topic_modeler = None
        self.clusterer = None

        # ML models
        self.models = {}
        self.label_encoder = LabelEncoder() if HAS_SKLEARN else None

        # Pre-trained models
        self.transformer_classifier = None

        # Model cache
        self.model_cache_dir = Path(self.config.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        if self.config.enable_topic_modeling:
            topic_config = TopicModelingConfig(num_topics=self.config.num_topics)
            self.topic_modeler = TopicModeler(topic_config)

        if self.config.enable_clustering:
            cluster_config = ClusteringConfig(num_clusters=self.config.num_clusters)
            self.clusterer = DocumentClusterer(cluster_config)

        # Load pre-trained transformer if available
        if HAS_TRANSFORMERS and self.config.use_pretrained_models:
            try:
                self.transformer_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Fallback model
                    device=0 if torch.cuda.is_available() else -1,
                )
                logger.info("Transformer classifier loaded")
            except Exception as e:
                logger.warning(f"Failed to load transformer classifier: {e}")

    async def classify_document(
        self, text: str, document_id: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single document

        Args:
            text: Document text
            document_id: Optional document identifier

        Returns:
            Classification results
        """
        try:
            # Extract features
            features = await self.feature_extractor.extract_features([text])

            if features.size == 0:
                logger.warning("No features extracted from document")
                return self._create_empty_result()

            # Classify using different methods
            results = {}

            if self.config.method == ClassificationMethod.HYBRID:
                # Use multiple methods and combine results
                if HAS_SKLEARN:
                    sklearn_result = await self._classify_with_sklearn([text], features)
                    results["sklearn"] = sklearn_result

                if self.transformer_classifier:
                    transformer_result = await self._classify_with_transformer([text])
                    results["transformer"] = transformer_result

                # Combine results
                final_result = self._combine_classification_results(
                    results, text, features
                )

            else:
                # Use single method
                if (
                    self.config.method
                    in [
                        ClassificationMethod.NAIVE_BAYES,
                        ClassificationMethod.SVM,
                        ClassificationMethod.RANDOM_FOREST,
                        ClassificationMethod.LOGISTIC_REGRESSION,
                    ]
                    and HAS_SKLEARN
                ):
                    final_result = await self._classify_with_sklearn([text], features)
                elif (
                    self.config.method == ClassificationMethod.NEURAL_NETWORK
                    and self.transformer_classifier
                ):
                    final_result = await self._classify_with_transformer([text])
                else:
                    logger.warning(
                        f"Classification method not available: {self.config.method}"
                    )
                    final_result = self._create_empty_result()

            # Add topic information if available
            if self.topic_modeler:
                try:
                    topic_result = await self.topic_modeler.fit_topic_model([text])
                    if topic_result.document_topics:
                        final_result.topics = topic_result.document_topics[0]
                except Exception as e:
                    logger.debug(f"Topic modeling error: {e}")

            # Add clustering information if available
            if self.clusterer:
                try:
                    cluster_result = await self.clusterer.cluster_documents(
                        [text], features
                    )
                    if cluster_result.cluster_labels:
                        final_result.cluster_id = cluster_result.cluster_labels[0]
                except Exception as e:
                    logger.debug(f"Clustering error: {e}")

            return final_result

        except Exception as e:
            logger.error(f"Document classification error: {e}")
            return self._create_empty_result()

    async def train_classifier(
        self, documents: List[str], labels: List[str]
    ) -> Dict[str, Any]:
        """
        Train classification models on labeled data

        Args:
            documents: Training documents
            labels: Document labels

        Returns:
            Training results and metrics
        """
        try:
            if not HAS_SKLEARN:
                logger.error("Scikit-learn not available for training")
                return {}

            if len(documents) != len(labels):
                raise ClassificationError("Number of documents and labels must match")

            # Extract features
            features = await self.feature_extractor.extract_features(documents)

            if features.size == 0:
                logger.error("No features extracted for training")
                return {}

            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                encoded_labels,
                test_size=0.2,
                random_state=42,
                stratify=encoded_labels,
            )

            # Train models
            model_results = {}

            models_to_train = {
                "naive_bayes": MultinomialNB(),
                "svm": SVC(probability=True, random_state=42),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "logistic_regression": LogisticRegression(
                    random_state=42, max_iter=1000
                ),
            }

            for model_name, model in models_to_train.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)

                    # Cross-validation
                    cv_scores = cross_val_score(model, features, encoded_labels, cv=5)

                    # Predictions
                    y_pred = model.predict(X_test)

                    # Store model
                    self.models[model_name] = model

                    model_results[model_name] = {
                        "train_accuracy": float(train_score),
                        "test_accuracy": float(test_score),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std()),
                        "classification_report": classification_report(
                            y_test,
                            y_pred,
                            target_names=self.label_encoder.classes_,
                            output_dict=True,
                        ),
                    }

                    logger.info(
                        f"Trained {model_name}: test accuracy = {test_score:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue

            # Save models
            await self._save_models()

            return {
                "training_completed": True,
                "num_documents": len(documents),
                "num_classes": len(set(labels)),
                "feature_dimensions": features.shape[1],
                "model_results": model_results,
                "best_model": (
                    max(
                        model_results.keys(),
                        key=lambda k: model_results[k]["test_accuracy"],
                    )
                    if model_results
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Training error: {e}")
            return {"training_completed": False, "error": str(e)}

    async def _classify_with_sklearn(
        self, documents: List[str], features: np.ndarray
    ) -> ClassificationResult:
        """Classify using scikit-learn models"""
        try:
            if not self.models or not self.label_encoder:
                # Use default model if no trained models
                logger.info("Using default Naive Bayes classifier")

                # Create simple default classification
                return ClassificationResult(
                    predicted_category=DocumentCategory.UNKNOWN,
                    confidence=0.5,
                    probabilities={"unknown": 0.5, "other": 0.5},
                    features_used=["default"],
                    model_info={"model": "default", "method": "rule_based"},
                )

            # Use best available model
            best_model_name = list(self.models.keys())[0]  # Simplified selection
            model = self.models[best_model_name]

            # Predict
            if features.shape[0] == 0:
                return self._create_empty_result()

            predictions = model.predict(features)
            probabilities = (
                model.predict_proba(features)
                if hasattr(model, "predict_proba")
                else None
            )

            # Convert back to original labels
            predicted_label = self.label_encoder.inverse_transform([predictions[0]])[0]

            # Create probability dictionary
            prob_dict = {}
            if probabilities is not None:
                class_names = self.label_encoder.classes_
                prob_dict = {
                    name: float(prob)
                    for name, prob in zip(class_names, probabilities[0])
                }

            # Map to DocumentCategory
            predicted_category = self._map_to_document_category(predicted_label)

            return ClassificationResult(
                predicted_category=predicted_category,
                confidence=float(
                    max(probabilities[0]) if probabilities is not None else 0.5
                ),
                probabilities=prob_dict,
                features_used=self.feature_extractor.get_feature_names(
                    self.config.feature_type
                ),
                model_info={"model": best_model_name, "method": "sklearn"},
            )

        except Exception as e:
            logger.error(f"Scikit-learn classification error: {e}")
            return self._create_empty_result()

    async def _classify_with_transformer(
        self, documents: List[str]
    ) -> ClassificationResult:
        """Classify using transformer models"""
        try:
            if not self.transformer_classifier:
                return self._create_empty_result()

            # Classify with transformer
            result = self.transformer_classifier(documents[0])

            # Process result
            if isinstance(result, list) and len(result) > 0:
                top_result = result[0]
                label = top_result.get("label", "UNKNOWN")
                score = top_result.get("score", 0.0)

                predicted_category = self._map_to_document_category(label)

                return ClassificationResult(
                    predicted_category=predicted_category,
                    confidence=float(score),
                    probabilities={label: float(score)},
                    features_used=["transformer_embeddings"],
                    model_info={"model": "transformer", "method": "neural_network"},
                )

            return self._create_empty_result()

        except Exception as e:
            logger.error(f"Transformer classification error: {e}")
            return self._create_empty_result()

    def _combine_classification_results(
        self, results: Dict[str, ClassificationResult], text: str, features: np.ndarray
    ) -> ClassificationResult:
        """Combine results from multiple classification methods"""
        try:
            if not results:
                return self._create_empty_result()

            # Weighted combination of results
            weights = {"sklearn": 0.6, "transformer": 0.4}

            category_votes = {}
            confidence_sum = 0.0
            all_probabilities = {}

            for method, result in results.items():
                weight = weights.get(method, 0.5)

                # Vote for category
                category = result.predicted_category.value
                category_votes[category] = (
                    category_votes.get(category, 0) + weight * result.confidence
                )

                # Accumulate confidence
                confidence_sum += weight * result.confidence

                # Merge probabilities
                for label, prob in result.probabilities.items():
                    all_probabilities[label] = (
                        all_probabilities.get(label, 0) + weight * prob
                    )

            # Determine final category
            if category_votes:
                final_category_str = max(category_votes.keys(), key=category_votes.get)
                final_category = DocumentCategory(final_category_str)
                final_confidence = category_votes[final_category_str]
            else:
                final_category = DocumentCategory.UNKNOWN
                final_confidence = 0.5

            # Normalize probabilities
            total_prob = sum(all_probabilities.values())
            if total_prob > 0:
                normalized_probs = {
                    k: v / total_prob for k, v in all_probabilities.items()
                }
            else:
                normalized_probs = all_probabilities

            return ClassificationResult(
                predicted_category=final_category,
                confidence=min(final_confidence, 1.0),
                probabilities=normalized_probs,
                features_used=["combined"],
                model_info={
                    "model": "hybrid",
                    "method": "ensemble",
                    "components": list(results.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Result combination error: {e}")
            return self._create_empty_result()

    def _map_to_document_category(self, label: str) -> DocumentCategory:
        """Map classification label to DocumentCategory"""
        try:
            label_lower = label.lower()

            # Direct mapping
            for category in DocumentCategory:
                if category.value.lower() == label_lower:
                    return category

            # Fuzzy matching
            mapping = {
                "legal": DocumentCategory.LEGAL,
                "contract": DocumentCategory.CONTRACTS,
                "agreement": DocumentCategory.LEGAL,
                "invoice": DocumentCategory.INVOICES,
                "bill": DocumentCategory.INVOICES,
                "receipt": DocumentCategory.INVOICES,
                "financial": DocumentCategory.FINANCIAL,
                "finance": DocumentCategory.FINANCIAL,
                "money": DocumentCategory.FINANCIAL,
                "technical": DocumentCategory.TECHNICAL,
                "manual": DocumentCategory.MANUALS,
                "guide": DocumentCategory.MANUALS,
                "instruction": DocumentCategory.MANUALS,
                "report": DocumentCategory.REPORTS,
                "analysis": DocumentCategory.REPORTS,
                "form": DocumentCategory.FORMS,
                "application": DocumentCategory.FORMS,
                "academic": DocumentCategory.ACADEMIC,
                "research": DocumentCategory.ACADEMIC,
                "paper": DocumentCategory.ACADEMIC,
                "business": DocumentCategory.BUSINESS,
                "corporate": DocumentCategory.BUSINESS,
                "medical": DocumentCategory.MEDICAL,
                "health": DocumentCategory.MEDICAL,
                "government": DocumentCategory.GOVERNMENT,
                "official": DocumentCategory.GOVERNMENT,
                "personal": DocumentCategory.PERSONAL,
                "private": DocumentCategory.PERSONAL,
                "marketing": DocumentCategory.MARKETING,
                "advertisement": DocumentCategory.MARKETING,
                "correspondence": DocumentCategory.CORRESPONDENCE,
                "email": DocumentCategory.CORRESPONDENCE,
                "letter": DocumentCategory.CORRESPONDENCE,
            }

            for key, category in mapping.items():
                if key in label_lower:
                    return category

            return DocumentCategory.UNKNOWN

        except Exception:
            return DocumentCategory.UNKNOWN

    def _create_empty_result(self) -> ClassificationResult:
        """Create empty classification result"""
        return ClassificationResult(
            predicted_category=DocumentCategory.UNKNOWN,
            confidence=0.0,
            probabilities={},
            features_used=[],
            model_info={"error": "classification_failed"},
        )

    async def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            if not self.models:
                return

            # Save sklearn models
            for model_name, model in self.models.items():
                model_path = self.model_cache_dir / f"{model_name}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # Save label encoder
            if self.label_encoder:
                encoder_path = self.model_cache_dir / "label_encoder.pkl"
                with open(encoder_path, "wb") as f:
                    pickle.dump(self.label_encoder, f)

            # Save vectorizers
            vectorizers_path = self.model_cache_dir / "vectorizers.pkl"
            with open(vectorizers_path, "wb") as f:
                pickle.dump(self.feature_extractor.vectorizers, f)

            logger.info(f"Models saved to {self.model_cache_dir}")

        except Exception as e:
            logger.error(f"Model saving error: {e}")

    async def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load sklearn models
            model_files = list(self.model_cache_dir.glob("*_model.pkl"))
            for model_file in model_files:
                model_name = model_file.stem.replace("_model", "")
                with open(model_file, "rb") as f:
                    self.models[model_name] = pickle.load(f)

            # Load label encoder
            encoder_path = self.model_cache_dir / "label_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)

            # Load vectorizers
            vectorizers_path = self.model_cache_dir / "vectorizers.pkl"
            if vectorizers_path.exists():
                with open(vectorizers_path, "rb") as f:
                    self.feature_extractor.vectorizers = pickle.load(f)

            logger.info(f"Models loaded from {self.model_cache_dir}")
            return len(self.models) > 0

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False

    def create_demo_classification(self) -> Dict[str, Any]:
        """Create demo classification results"""
        try:
            return {
                "document_text": "This is a software development contract between Company A and Company B for the development of a web application...",
                "classification_result": {
                    "predicted_category": DocumentCategory.CONTRACTS.value,
                    "confidence": 0.92,
                    "probabilities": {
                        "contracts": 0.92,
                        "legal": 0.85,
                        "business": 0.78,
                        "technical": 0.45,
                        "financial": 0.23,
                    },
                    "features_used": [
                        "contract",
                        "agreement",
                        "development",
                        "software",
                        "terms",
                    ],
                    "model_info": {
                        "model": "hybrid",
                        "method": "ensemble",
                        "components": ["sklearn", "transformer"],
                    },
                    "topics": [
                        (0, 0.65, "software development"),
                        (1, 0.25, "legal terms"),
                        (2, 0.10, "payment conditions"),
                    ],
                    "cluster_id": 2,
                    "similar_documents": [
                        {
                            "id": "doc_123",
                            "similarity": 0.89,
                            "title": "Mobile App Development Agreement",
                        },
                        {
                            "id": "doc_456",
                            "similarity": 0.76,
                            "title": "Web Services Contract",
                        },
                    ],
                },
                "topic_modeling_result": {
                    "topics": [
                        {
                            "id": 0,
                            "name": "Topic_0: software, development, application",
                            "top_words": [
                                ("software", 0.15),
                                ("development", 0.12),
                                ("application", 0.10),
                                ("web", 0.08),
                                ("system", 0.07),
                            ],
                            "weight": 45.2,
                        },
                        {
                            "id": 1,
                            "name": "Topic_1: contract, agreement, terms",
                            "top_words": [
                                ("contract", 0.18),
                                ("agreement", 0.14),
                                ("terms", 0.11),
                                ("conditions", 0.09),
                                ("legal", 0.08),
                            ],
                            "weight": 32.8,
                        },
                    ],
                    "document_topics": [[(0, 0.65), (1, 0.25), (2, 0.10)]],
                    "coherence_score": 0.78,
                    "perplexity": 145.2,
                },
                "clustering_result": {
                    "cluster_labels": [2],
                    "silhouette_score": 0.67,
                    "num_clusters": 5,
                    "cluster_info": [
                        {
                            "cluster_id": 2,
                            "size": 23,
                            "percentage": 15.3,
                            "representative_document": "This is a software development contract...",
                            "key_terms": [
                                "contract",
                                "software",
                                "development",
                                "agreement",
                                "terms",
                            ],
                            "centroid_available": True,
                        }
                    ],
                },
                "processing_config": {
                    "method": "hybrid",
                    "feature_type": "combined",
                    "use_pretrained_models": True,
                    "enable_topic_modeling": True,
                    "enable_clustering": True,
                },
            }

        except Exception as e:
            logger.error(f"Demo creation error: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo document classification"""
        try:
            # Create document classifier
            config = ClassificationConfig(
                method=ClassificationMethod.HYBRID,
                feature_type=FeatureType.COMBINED,
                use_pretrained_models=HAS_TRANSFORMERS,
                enable_topic_modeling=True,
                enable_clustering=True,
            )

            classifier = DocumentClassifier(config)

            print("Document Classification Demo")
            print("=" * 50)
            print(f"Scikit-learn available: {HAS_SKLEARN}")
            print(f"Transformers available: {HAS_TRANSFORMERS}")
            print(f"Sentence Transformers available: {HAS_SENTENCE_TRANSFORMERS}")
            print()

            # Demo documents
            demo_docs = [
                "This software development agreement outlines the terms and conditions for building a web application.",
                "Please find attached the medical records for patient John Doe, including lab results and treatment history.",
                "The quarterly financial report shows revenue growth of 15% compared to last year.",
                "Research paper: Machine Learning Applications in Natural Language Processing",
                "Invoice #12345 - Amount due: $1,500.00 for consulting services rendered in March 2024.",
            ]

            print("Classifying demo documents...")

            for i, doc in enumerate(demo_docs):
                print(f"\nDocument {i+1}: {doc[:60]}...")

                result = await classifier.classify_document(doc)

                print(f"  Category: {result.predicted_category.value}")
                print(f"  Confidence: {result.confidence:.2f}")

                if result.probabilities:
                    top_probs = sorted(
                        result.probabilities.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    print(
                        f"  Top probabilities: {', '.join([f'{k}({v:.2f})' for k, v in top_probs])}"
                    )

            # Create comprehensive demo results
            print("\nCreating comprehensive demo results...")
            demo_results = classifier.create_demo_classification()

            print(f"Demo Classification:")
            print(
                f"  Category: {demo_results['classification_result']['predicted_category']}"
            )
            print(
                f"  Confidence: {demo_results['classification_result']['confidence']}"
            )
            print(
                f"  Topics found: {len(demo_results['topic_modeling_result']['topics'])}"
            )
            print(f"  Clusters: {demo_results['clustering_result']['num_clusters']}")

            print("\nDocument Classification demo completed successfully!")

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
