# OpenAPI Migration Guide

## Overview

Vega2.0 has been enhanced with a comprehensive OpenAPI-compliant API. This migration introduces standardized schemas, proper documentation, and improved error handling.

## What's Changed

### 1. API Documentation

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI JSON**: Available at `/openapi.json`

### 2. Enhanced Response Models

All endpoints now return structured responses with:

- Consistent `status` and `timestamp` fields
- Proper error handling with error codes
- Comprehensive field documentation
- Type validation

### 3. Request Validation

- Pydantic models for all request bodies
- Query parameter validation
- Path parameter validation
- Request examples in documentation

### 4. Security

- Formalized API key authentication
- Security schemes documented in OpenAPI
- Proper HTTP status codes

## Migration Steps

### For API Consumers

1. **Update API Base URL** (if needed):

   ```
   Old: http://localhost:8000
   New: http://localhost:8001 (for testing)
   ```

2. **Handle New Response Format**:

   ```python
   # Old response
   {"response": "Hello", "session_id": "123"}
   
   # New response
   {
     "status": "success",
     "timestamp": "2024-01-01T12:00:00Z",
     "response": "Hello",
     "session_id": "123",
     "model_used": "gpt-3.5",
     "tokens_used": 15,
     "cost_estimate": 0.001
   }
   ```

3. **Error Handling**:

   ```python
   # Old error
   {"detail": "Error message"}
   
   # New error
   {
     "status": "error", 
     "timestamp": "2024-01-01T12:00:00Z",
     "message": "Error message",
     "error_code": "LLM_UNAVAILABLE",
     "details": {"provider": "openai", "reason": "rate_limit"}
   }
   ```

### For Developers

1. **Use Type Hints**:

   ```python
   from core.openapi_app import ChatRequest, ChatResponse
   
   async def process_chat(request: ChatRequest) -> ChatResponse:
       # Implementation with full type safety
   ```

2. **Leverage Pydantic Validation**:

   ```python
   # Automatic validation of request data
   chat_request = ChatRequest(
       prompt="Hello",
       temperature=0.7,  # Validated: 0.0 <= temperature <= 2.0
       max_tokens=1000   # Validated: 1 <= max_tokens <= 4096
   )
   ```

3. **Use Structured Responses**:

   ```python
   return ChatResponse(
       response=generated_text,
       session_id=session_id,
       model_used="gpt-4",
       tokens_used=token_count,
       cost_estimate=calculated_cost
   )
   ```

## Running the OpenAPI Server

### Development

```bash
# Start the OpenAPI development server
python run_openapi_server.py

# Or using uvicorn directly
uvicorn core.openapi_app:app --host 127.0.0.1 --port 8001 --reload
```

### Production

```bash
uvicorn core.openapi_app:app --host 127.0.0.1 --port 8000 --workers 4
```

## API Reference

### Authentication

All endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8001/health
```

### Core Endpoints

#### Chat

```bash
POST /chat
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "Hello, how are you?",
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

#### History

```bash
GET /history?limit=50&offset=0
X-API-Key: your-api-key
```

#### Proactive Conversations

```bash
POST /proactive/propose
Content-Type: application/json
X-API-Key: your-api-key

{
  "max_per_day": 5,
  "categories": ["technology", "science"]
}
```

## Benefits

1. **Auto-generated Documentation**: Interactive API docs with examples
2. **Type Safety**: Full Pydantic validation and type hints
3. **Better Error Handling**: Structured error responses with codes
4. **API Versioning**: Prepared for future API versions
5. **Standards Compliance**: OpenAPI 3.0 compliant
6. **Testing**: Easy integration testing with generated schemas

## Backward Compatibility

The original API endpoints remain available in `core/app.py`. You can run both simultaneously during migration:

- Original API: Port 8000
- OpenAPI version: Port 8001

## Next Steps

1. Test the OpenAPI endpoints with your existing clients
2. Update client code to use new response formats
3. Switch to the OpenAPI version when ready
4. Remove legacy endpoints after migration is complete
