# Vega 2.0 Multi-modal Processing API Documentation

## Overview

The Vega 2.0 Multi-modal Processing API enables advanced AI processing across multiple data types including audio, video, images, documents, and mixed-media content. This API provides intelligent analysis, feature extraction, content generation, and cross-modal understanding capabilities.

**Base URL**: `http://localhost:8000` (development) | `https://api.vega2.example.com` (production)

**Authentication**: X-API-Key header required for all endpoints

## File Upload & Management

### POST /multimodal/upload

Upload media files for processing

**Request**: multipart/form-data

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('type', 'auto'); // auto, image, audio, video, document
formData.append('metadata', JSON.stringify({
    title: 'Sample Video',
    description: 'Product demonstration video',
    tags: ['demo', 'product']
}));

fetch('/multimodal/upload', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-api-key'
    },
    body: formData
});
```

**Response**:

```json
{
  "file_id": "file_abc123",
  "filename": "demo_video.mp4",
  "type": "video",
  "size_bytes": 52428800,
  "duration_seconds": 180,
  "mime_type": "video/mp4",
  "resolution": {
    "width": 1920,
    "height": 1080
  },
  "upload_url": "https://storage.vega2.example.com/uploads/file_abc123",
  "thumbnail_url": "https://storage.vega2.example.com/thumbnails/file_abc123.jpg",
  "metadata": {
    "title": "Sample Video",
    "description": "Product demonstration video",
    "tags": ["demo", "product"],
    "codec": "h264",
    "bitrate": 2500000,
    "fps": 30
  },
  "processing_status": "pending",
  "created_at": "2025-09-20T10:00:00Z"
}
```

### GET /multimodal/files/{file_id}

Get file details and processing status

**Response**:

```json
{
  "file_id": "file_abc123",
  "filename": "demo_video.mp4",
  "type": "video",
  "processing_status": "completed",
  "analysis_results": {
    "object_detection": {
      "objects": [
        {
          "label": "person",
          "confidence": 0.95,
          "bounding_box": [100, 150, 300, 400],
          "timestamp": 5.2
        }
      ]
    },
    "transcript": {
      "text": "Welcome to our product demonstration...",
      "confidence": 0.88,
      "speakers": ["speaker_1"],
      "language": "en"
    },
    "sentiment": {
      "overall": "positive",
      "score": 0.75,
      "emotions": {
        "joy": 0.45,
        "confidence": 0.30,
        "neutral": 0.25
      }
    }
  },
  "features": {
    "visual_features": [0.1, 0.2, 0.3],
    "audio_features": [0.4, 0.5, 0.6],
    "text_features": [0.7, 0.8, 0.9]
  }
}
```

## Audio Processing

### POST /multimodal/audio/transcribe

Transcribe audio to text with speaker identification

**Request Body**:

```json
{
  "file_id": "audio_xyz789",
  "options": {
    "language": "auto",
    "speaker_separation": true,
    "noise_reduction": true,
    "punctuation": true,
    "confidence_threshold": 0.8
  }
}
```

**Response**:

```json
{
  "transcript_id": "transcript_abc123",
  "status": "completed",
  "duration_seconds": 120,
  "language_detected": "en-US",
  "confidence": 0.92,
  "transcript": {
    "full_text": "Welcome everyone to today's meeting. Let's start with the quarterly review.",
    "segments": [
      {
        "start_time": 0.0,
        "end_time": 3.5,
        "text": "Welcome everyone to today's meeting.",
        "speaker": "speaker_1",
        "confidence": 0.95,
        "words": [
          {
            "word": "Welcome",
            "start": 0.0,
            "end": 0.8,
            "confidence": 0.98
          }
        ]
      }
    ],
    "speakers": [
      {
        "id": "speaker_1",
        "name": null,
        "total_speaking_time": 85.2,
        "confidence": 0.88
      }
    ]
  },
  "analysis": {
    "sentiment": "neutral",
    "topics": ["business", "meeting", "quarterly"],
    "key_phrases": ["quarterly review", "team meeting"],
    "language_quality": 0.92
  }
}
```

### POST /multimodal/audio/analyze

Analyze audio characteristics and content

**Request Body**:

```json
{
  "file_id": "audio_xyz789",
  "analysis_types": [
    "emotion",
    "music_classification",
    "sound_events",
    "speech_quality"
  ]
}
```

**Response**:

```json
{
  "analysis_id": "analysis_def456",
  "status": "completed",
  "results": {
    "emotion_analysis": {
      "overall_emotion": "calm",
      "emotions_timeline": [
        {
          "start_time": 0.0,
          "end_time": 10.0,
          "emotions": {
            "calm": 0.7,
            "happy": 0.2,
            "neutral": 0.1
          }
        }
      ],
      "speaker_emotions": {
        "speaker_1": "calm",
        "speaker_2": "excited"
      }
    },
    "music_classification": {
      "is_music": false,
      "genre": null,
      "instruments": [],
      "tempo": null
    },
    "sound_events": [
      {
        "event": "door_opening",
        "start_time": 15.2,
        "end_time": 16.1,
        "confidence": 0.85
      },
      {
        "event": "phone_ringing",
        "start_time": 45.8,
        "end_time": 50.2,
        "confidence": 0.92
      }
    ],
    "speech_quality": {
      "clarity": 0.88,
      "noise_level": 0.15,
      "reverberation": 0.22,
      "overall_quality": "good"
    }
  }
}
```

### POST /multimodal/audio/synthesis

Generate speech from text

**Request Body**:

```json
{
  "text": "Hello, this is a demonstration of our text-to-speech synthesis.",
  "voice": {
    "name": "neural_voice_1",
    "gender": "female",
    "age": "adult",
    "accent": "american"
  },
  "options": {
    "speed": 1.0,
    "pitch": 0.0,
    "emotion": "neutral",
    "format": "mp3",
    "quality": "high"
  }
}
```

**Response**:

```json
{
  "synthesis_id": "synthesis_ghi789",
  "status": "completed",
  "audio_url": "https://storage.vega2.example.com/synthesis/synthesis_ghi789.mp3",
  "duration_seconds": 12.5,
  "sample_rate": 44100,
  "format": "mp3",
  "size_bytes": 201600,
  "voice_used": {
    "name": "neural_voice_1",
    "model": "wavenet_v2",
    "language": "en-US"
  }
}
```

## Video Processing

### POST /multimodal/video/analyze

Comprehensive video content analysis

**Request Body**:

```json
{
  "file_id": "video_jkl012",
  "analysis_types": [
    "object_detection",
    "face_recognition", 
    "action_recognition",
    "scene_detection",
    "ocr"
  ],
  "options": {
    "frame_sampling_rate": 1.0,
    "confidence_threshold": 0.7,
    "tracking": true
  }
}
```

**Response**:

```json
{
  "analysis_id": "video_analysis_mno345",
  "status": "completed",
  "duration_seconds": 180,
  "frame_count": 5400,
  "fps": 30,
  "results": {
    "object_detection": {
      "objects": [
        {
          "id": "obj_001",
          "label": "person",
          "confidence": 0.95,
          "tracking": {
            "stable": true,
            "duration": 45.2
          },
          "appearances": [
            {
              "start_time": 10.0,
              "end_time": 55.2,
              "bounding_boxes": [
                {
                  "frame": 300,
                  "box": [150, 200, 350, 600],
                  "confidence": 0.95
                }
              ]
            }
          ]
        }
      ],
      "summary": {
        "unique_objects": 15,
        "total_detections": 1250
      }
    },
    "face_recognition": {
      "faces": [
        {
          "face_id": "face_001",
          "person_id": "person_123",
          "appearances": [
            {
              "start_time": 5.0,
              "end_time": 45.0,
              "confidence": 0.92,
              "emotions": {
                "happy": 0.7,
                "neutral": 0.3
              }
            }
          ]
        }
      ]
    },
    "action_recognition": {
      "actions": [
        {
          "action": "presentation",
          "start_time": 0.0,
          "end_time": 120.0,
          "confidence": 0.88,
          "actors": ["person_123"]
        },
        {
          "action": "writing_on_whiteboard",
          "start_time": 45.0,
          "end_time": 75.0,
          "confidence": 0.82,
          "actors": ["person_123"]
        }
      ]
    },
    "scene_detection": {
      "scenes": [
        {
          "scene_id": "scene_001",
          "start_time": 0.0,
          "end_time": 60.0,
          "type": "indoor_office",
          "confidence": 0.91,
          "description": "Modern office conference room with whiteboard"
        }
      ]
    },
    "ocr": {
      "text_regions": [
        {
          "text": "Q3 Sales Report",
          "bounding_box": [500, 100, 800, 150],
          "confidence": 0.95,
          "start_time": 30.0,
          "end_time": 90.0,
          "language": "en"
        }
      ]
    }
  }
}
```

### POST /multimodal/video/extract-frames

Extract specific frames or thumbnails

**Request Body**:

```json
{
  "file_id": "video_jkl012",
  "extraction_type": "keyframes",
  "options": {
    "max_frames": 20,
    "resolution": "720p",
    "format": "jpg",
    "quality": 85,
    "timestamps": [10.0, 30.0, 60.0]
  }
}
```

**Response**:

```json
{
  "extraction_id": "frames_pqr678",
  "status": "completed",
  "frames": [
    {
      "frame_number": 300,
      "timestamp": 10.0,
      "url": "https://storage.vega2.example.com/frames/frames_pqr678_300.jpg",
      "width": 1280,
      "height": 720,
      "size_bytes": 95432,
      "is_keyframe": true
    }
  ],
  "total_frames": 20,
  "processing_time": 15.2
}
```

### POST /multimodal/video/generate

Generate video from images, text, or other media

**Request Body**:

```json
{
  "generation_type": "slideshow",
  "inputs": {
    "images": [
      "file_abc123",
      "file_def456",
      "file_ghi789"
    ],
    "audio": "file_jkl012",
    "text_overlays": [
      {
        "text": "Introduction",
        "start_time": 0.0,
        "duration": 3.0,
        "position": "center",
        "style": "title"
      }
    ]
  },
  "options": {
    "resolution": "1080p",
    "fps": 30,
    "transition_duration": 1.0,
    "background_music": true,
    "format": "mp4"
  }
}
```

**Response**:

```json
{
  "generation_id": "video_gen_stu901",
  "status": "processing",
  "estimated_completion": "2025-09-20T10:15:00Z",
  "progress": {
    "percentage": 25,
    "current_stage": "rendering_transitions",
    "estimated_remaining": 180
  }
}
```

## Image Processing

### POST /multimodal/image/analyze

Comprehensive image analysis

**Request Body**:

```json
{
  "file_id": "image_vwx234",
  "analysis_types": [
    "object_detection",
    "face_detection",
    "ocr",
    "aesthetic_scoring",
    "content_moderation"
  ],
  "options": {
    "confidence_threshold": 0.8,
    "detailed_attributes": true
  }
}
```

**Response**:

```json
{
  "analysis_id": "image_analysis_yza567",
  "status": "completed",
  "image_info": {
    "width": 1920,
    "height": 1080,
    "format": "jpg",
    "color_space": "rgb",
    "has_transparency": false
  },
  "results": {
    "object_detection": {
      "objects": [
        {
          "label": "dog",
          "confidence": 0.95,
          "bounding_box": [200, 300, 800, 900],
          "attributes": {
            "breed": "golden_retriever",
            "size": "large",
            "pose": "sitting"
          }
        }
      ]
    },
    "face_detection": {
      "faces": [
        {
          "bounding_box": [150, 100, 350, 350],
          "confidence": 0.92,
          "attributes": {
            "age": "adult",
            "gender": "female",
            "emotion": "happy",
            "glasses": false,
            "beard": false
          },
          "landmarks": {
            "left_eye": [200, 180],
            "right_eye": [280, 180],
            "nose": [240, 220],
            "mouth": [240, 280]
          }
        }
      ]
    },
    "ocr": {
      "text_blocks": [
        {
          "text": "Welcome to the Park",
          "bounding_box": [100, 50, 500, 100],
          "confidence": 0.98,
          "language": "en"
        }
      ]
    },
    "aesthetic_scoring": {
      "overall_score": 7.8,
      "composition": 8.2,
      "lighting": 7.5,
      "color_harmony": 8.0,
      "sharpness": 9.1,
      "noise": 1.2
    },
    "content_moderation": {
      "safe_for_work": true,
      "categories": {
        "adult": 0.02,
        "violence": 0.01,
        "racy": 0.03
      },
      "flags": []
    }
  }
}
```

### POST /multimodal/image/edit

AI-powered image editing

**Request Body**:

```json
{
  "file_id": "image_vwx234",
  "edits": [
    {
      "type": "background_removal",
      "options": {
        "feather_edges": true,
        "accuracy": "high"
      }
    },
    {
      "type": "object_replacement",
      "target_object": "dog",
      "replacement": "cat",
      "options": {
        "preserve_lighting": true,
        "match_style": true
      }
    },
    {
      "type": "style_transfer",
      "style": "impressionist",
      "intensity": 0.7
    }
  ]
}
```

**Response**:

```json
{
  "edit_id": "image_edit_bcd890",
  "status": "processing",
  "edits_applied": 1,
  "total_edits": 3,
  "progress": {
    "percentage": 33,
    "current_operation": "object_replacement",
    "estimated_completion": "2025-09-20T10:12:00Z"
  },
  "preview_url": "https://storage.vega2.example.com/edits/preview_bcd890.jpg"
}
```

### POST /multimodal/image/generate

Generate images from text descriptions

**Request Body**:

```json
{
  "prompt": "A serene mountain landscape at sunset with a crystal clear lake reflecting the sky",
  "negative_prompt": "blurry, low quality, distorted",
  "options": {
    "style": "photorealistic",
    "resolution": "1024x1024",
    "aspect_ratio": "16:9",
    "guidance_scale": 7.5,
    "steps": 50,
    "seed": 42,
    "batch_size": 4
  }
}
```

**Response**:

```json
{
  "generation_id": "image_gen_efg123",
  "status": "completed",
  "images": [
    {
      "image_id": "gen_img_001",
      "url": "https://storage.vega2.example.com/generated/gen_img_001.png",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "aesthetic_score": 8.5
    }
  ],
  "model_used": "stable_diffusion_xl",
  "generation_time": 15.8
}
```

## Document Processing

### POST /multimodal/document/extract

Extract and analyze document content

**Request Body**:

```json
{
  "file_id": "document_hij456",
  "extraction_types": [
    "text",
    "tables",
    "images",
    "layout",
    "metadata"
  ],
  "options": {
    "preserve_formatting": true,
    "extract_signatures": true,
    "ocr_language": "auto"
  }
}
```

**Response**:

```json
{
  "extraction_id": "doc_extract_klm789",
  "status": "completed",
  "document_info": {
    "pages": 15,
    "format": "pdf",
    "size_bytes": 2048000,
    "creation_date": "2025-09-15T14:30:00Z",
    "author": "John Doe"
  },
  "content": {
    "text": {
      "full_text": "This is the complete document text...",
      "pages": [
        {
          "page_number": 1,
          "text": "Page 1 content...",
          "confidence": 0.95
        }
      ],
      "word_count": 2500,
      "language": "en"
    },
    "tables": [
      {
        "page": 3,
        "bounding_box": [100, 200, 500, 400],
        "rows": 5,
        "columns": 4,
        "data": [
          ["Header 1", "Header 2", "Header 3", "Header 4"],
          ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3", "Row 1 Col 4"]
        ],
        "confidence": 0.92
      }
    ],
    "images": [
      {
        "page": 2,
        "bounding_box": [150, 100, 450, 300],
        "extracted_url": "https://storage.vega2.example.com/extracted/img_001.jpg",
        "type": "chart",
        "description": "Bar chart showing quarterly sales"
      }
    ],
    "layout": {
      "headers": [
        {
          "text": "Chapter 1: Introduction",
          "level": 1,
          "page": 1,
          "position": [100, 50]
        }
      ],
      "paragraphs": [
        {
          "text": "This document provides...",
          "page": 1,
          "bounding_box": [100, 100, 500, 200]
        }
      ],
      "footnotes": [
        {
          "text": "Reference to external source",
          "page": 5,
          "reference_number": 1
        }
      ]
    },
    "metadata": {
      "title": "Annual Report 2025",
      "subject": "Financial Performance",
      "keywords": ["finance", "annual", "report"],
      "security": {
        "encrypted": false,
        "permissions": {
          "print": true,
          "copy": true,
          "edit": false
        }
      }
    }
  }
}
```

### POST /multimodal/document/analyze

Intelligent document analysis and understanding

**Request Body**:

```json
{
  "file_id": "document_hij456",
  "analysis_types": [
    "classification",
    "sentiment",
    "key_entities",
    "summarization",
    "topic_modeling"
  ],
  "options": {
    "language": "auto",
    "confidence_threshold": 0.7
  }
}
```

**Response**:

```json
{
  "analysis_id": "doc_analysis_nop012",
  "status": "completed",
  "results": {
    "classification": {
      "document_type": "financial_report",
      "confidence": 0.95,
      "categories": [
        {
          "category": "business_document",
          "confidence": 0.98
        },
        {
          "category": "financial_document",
          "confidence": 0.95
        }
      ]
    },
    "sentiment": {
      "overall_sentiment": "positive",
      "confidence": 0.82,
      "sentiment_distribution": {
        "positive": 0.65,
        "neutral": 0.30,
        "negative": 0.05
      },
      "page_sentiments": [
        {
          "page": 1,
          "sentiment": "neutral",
          "score": 0.1
        }
      ]
    },
    "key_entities": [
      {
        "text": "Vega Corporation",
        "type": "organization",
        "confidence": 0.98,
        "occurrences": 25,
        "first_mention_page": 1
      },
      {
        "text": "$2.5 million",
        "type": "money",
        "confidence": 0.95,
        "occurrences": 3,
        "context": "revenue"
      }
    ],
    "summarization": {
      "executive_summary": "The annual report shows strong financial performance with 15% revenue growth...",
      "key_points": [
        "Revenue increased by 15% year-over-year",
        "New product launches contributed 8% to growth",
        "Operating margin improved to 22%"
      ],
      "page_summaries": [
        {
          "page": 1,
          "summary": "Introduction and company overview..."
        }
      ]
    },
    "topic_modeling": {
      "topics": [
        {
          "topic_id": "topic_1",
          "label": "financial_performance",
          "weight": 0.35,
          "keywords": ["revenue", "profit", "growth", "margin"]
        },
        {
          "topic_id": "topic_2", 
          "label": "product_development",
          "weight": 0.25,
          "keywords": ["innovation", "product", "research", "development"]
        }
      ]
    }
  }
}
```

## Cross-modal Operations

### POST /multimodal/cross-modal/search

Search across multiple media types using natural language

**Request Body**:

```json
{
  "query": "Find videos of presentations about AI technology with positive sentiment",
  "media_types": ["video", "audio", "image", "document"],
  "filters": {
    "date_range": {
      "start": "2025-09-01T00:00:00Z",
      "end": "2025-09-20T23:59:59Z"
    },
    "sentiment": ["positive"],
    "duration_range": {
      "min_seconds": 60,
      "max_seconds": 3600
    },
    "tags": ["technology", "AI"]
  },
  "options": {
    "semantic_search": true,
    "cross_modal_similarity": true,
    "limit": 20
  }
}
```

**Response**:

```json
{
  "search_id": "search_qrs345",
  "query": "Find videos of presentations about AI technology with positive sentiment",
  "results": [
    {
      "file_id": "video_abc123",
      "type": "video",
      "title": "Future of AI Technology",
      "relevance_score": 0.92,
      "matches": {
        "transcript": "artificial intelligence technology future innovation",
        "visual": "presentation slides, speaker",
        "sentiment": "positive (0.85)"
      },
      "metadata": {
        "duration": 1800,
        "upload_date": "2025-09-15T10:00:00Z",
        "tags": ["AI", "technology", "future"]
      },
      "highlights": [
        {
          "type": "transcript",
          "text": "The future of AI technology looks incredibly promising...",
          "timestamp": 45.2
        }
      ]
    }
  ],
  "total_results": 15,
  "search_time": 2.8,
  "suggestions": [
    "machine learning presentations",
    "technology innovation videos",
    "AI research presentations"
  ]
}
```

### POST /multimodal/cross-modal/generate

Generate content across multiple modalities

**Request Body**:

```json
{
  "prompt": "Create a presentation about renewable energy with slides, narration, and background music",
  "output_types": ["video", "audio", "images", "text"],
  "style": {
    "visual_style": "professional",
    "voice_style": "authoritative",
    "music_style": "ambient"
  },
  "structure": {
    "slides": 8,
    "duration_per_slide": 30,
    "transitions": "fade",
    "include_charts": true
  }
}
```

**Response**:

```json
{
  "generation_id": "multimodal_gen_tuv678",
  "status": "processing", 
  "progress": {
    "percentage": 15,
    "current_stage": "generating_slide_content",
    "stages": [
      "content_planning",
      "generating_slide_content",
      "creating_visuals",
      "generating_narration",
      "adding_background_music",
      "video_compilation"
    ],
    "estimated_completion": "2025-09-20T10:25:00Z"
  },
  "partial_results": {
    "outline": {
      "slides": [
        {
          "slide_number": 1,
          "title": "Introduction to Renewable Energy",
          "content": "Overview of renewable energy sources and their importance"
        }
      ]
    }
  }
}
```

### POST /multimodal/cross-modal/align

Align and synchronize content across multiple media types

**Request Body**:

```json
{
  "files": {
    "video": "video_abc123",
    "audio": "audio_def456", 
    "transcript": "document_ghi789"
  },
  "alignment_type": "temporal",
  "options": {
    "auto_sync": true,
    "confidence_threshold": 0.8,
    "allow_small_adjustments": true
  }
}
```

**Response**:

```json
{
  "alignment_id": "alignment_wxy901",
  "status": "completed",
  "synchronization": {
    "video_audio_offset": 0.15,
    "transcript_alignment": [
      {
        "text": "Welcome to our presentation",
        "video_timestamp": 5.2,
        "audio_timestamp": 5.05,
        "confidence": 0.95
      }
    ],
    "overall_sync_quality": 0.92
  },
  "aligned_files": {
    "synchronized_video": "sync_video_abc123.mp4",
    "timestamped_transcript": "transcript_with_timing.json"
  }
}
```

## Batch Processing

### POST /multimodal/batch/process

Process multiple files in batch with the same operations

**Request Body**:

```json
{
  "files": [
    "file_001",
    "file_002", 
    "file_003"
  ],
  "operations": [
    {
      "type": "transcribe",
      "options": {
        "language": "auto",
        "speaker_separation": true
      }
    },
    {
      "type": "analyze_sentiment"
    },
    {
      "type": "extract_keywords"
    }
  ],
  "output_format": "json",
  "notification_webhook": "https://example.com/batch-complete"
}
```

**Response**:

```json
{
  "batch_id": "batch_zab234", 
  "status": "queued",
  "total_files": 3,
  "estimated_completion": "2025-09-20T10:30:00Z",
  "progress_url": "https://api.vega2.example.com/multimodal/batch/zab234/status"
}
```

### GET /multimodal/batch/{batch_id}/status

Get batch processing status and results

**Response**:

```json
{
  "batch_id": "batch_zab234",
  "status": "completed",
  "started_at": "2025-09-20T10:00:00Z",
  "completed_at": "2025-09-20T10:28:00Z",
  "total_files": 3,
  "processed_files": 3,
  "failed_files": 0,
  "results": [
    {
      "file_id": "file_001",
      "status": "completed",
      "results": {
        "transcription": {
          "text": "Transcript of file 001...",
          "confidence": 0.92
        },
        "sentiment": {
          "overall": "positive",
          "score": 0.75
        },
        "keywords": ["technology", "innovation", "future"]
      }
    }
  ],
  "summary": {
    "avg_processing_time": 560,
    "success_rate": 1.0,
    "total_cost": 2.50
  },
  "download_url": "https://api.vega2.example.com/downloads/batch_zab234_results.zip"
}
```

## Error Responses

### 400 Unsupported File Type

```json
{
  "error": "Unsupported file type",
  "code": 400,
  "details": {
    "provided_type": "application/x-unknown",
    "supported_types": [
      "video/mp4",
      "audio/wav",
      "image/jpeg",
      "application/pdf"
    ]
  }
}
```

### 413 File Too Large

```json
{
  "error": "File size exceeds limit",
  "code": 413,
  "details": {
    "file_size": 104857600,
    "max_size": 52428800,
    "type": "video"
  }
}
```

### 422 Processing Failed

```json
{
  "error": "Processing failed",
  "code": 422,
  "details": {
    "stage": "transcription",
    "reason": "Audio quality too low for reliable transcription",
    "suggestions": [
      "Improve audio quality",
      "Reduce background noise",
      "Lower confidence threshold"
    ]
  }
}
```

## Supported Formats

### Audio Formats

- MP3, WAV, FLAC, AAC, OGG
- Sample rates: 8kHz to 192kHz
- Max duration: 4 hours

### Video Formats  

- MP4, AVI, MOV, MKV, WebM
- Codecs: H.264, H.265, VP9
- Max resolution: 4K (3840x2160)
- Max duration: 2 hours

### Image Formats

- JPEG, PNG, GIF, WebP, TIFF, BMP
- Max resolution: 8K (7680x4320)
- Max file size: 50MB

### Document Formats

- PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX
- TXT, RTF, HTML
- Max pages: 500
- Max file size: 100MB
