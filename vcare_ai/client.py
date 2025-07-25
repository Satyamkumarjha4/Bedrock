import boto3
import json
import logging
import time
import atexit
import hashlib
from functools import lru_cache
from typing import Dict, Any, Optional, Union, List, Iterator, Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
from .config import config, ModelProvider, ConfigError
from .utils.token_utils import estimate_tokens


logger = logging.getLogger(__name__)

# Create a module-level registry
registry = CollectorRegistry()

# Register metrics on this registry
REQUEST_COUNTER = Counter('bedrock_requests_total', 'Total number of requests to Bedrock API', ['model', 'status'], registry=registry)
RESPONSE_TIME = Histogram('bedrock_response_time_seconds', 'Response time for Bedrock API calls', ['model'], registry=registry)
TOKEN_COUNTER = Counter('bedrock_tokens_total', 'Total tokens consumed', ['model', 'type'], registry=registry)
CACHE_HITS = Counter('bedrock_cache_hits_total', 'Cache hit count', registry=registry)
CACHE_MISSES = Counter('bedrock_cache_misses_total', 'Cache miss count', registry=registry)
ACTIVE_REQUESTS = Gauge('bedrock_active_requests', 'Number of active requests', registry=registry)

class BedrockClientError(Exception):
    """Base exception for BedrockClient errors"""
    pass

class BedrockRequestError(BedrockClientError):
    """Errors related to request formation"""
    pass

class BedrockResponseError(BedrockClientError):
    """Errors related to response handling"""
    pass

class BedrockRateLimitError(BedrockClientError):
    """Errors related to rate limiting"""
    pass

class ResponseCache:
    """Simple cache for model responses"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        """Get a value from cache"""
        if key in self.cache:
            CACHE_HITS.inc()
            return self.cache[key]
        CACHE_MISSES.inc()
        return None
        
    def set(self, key, value):
        """Add a value to cache"""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove a random item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

class BedrockClient:
    """Client for interacting with AWS Bedrock models"""
    
    # Class-level cache shared between all instances
    _response_cache = ResponseCache(max_size=200)
    _instances = []
    
    def __init__(self, custom_config=None):
        """
        Initialize the Bedrock client
        
        Args:
            custom_config: Optional custom configuration to override defaults
        """
        self.config = custom_config or config
        self.client = boto3.client("bedrock-runtime", region_name=self.config.region)
        self.request_count = 0
        self.active_streams = set()
        logger.info(f"Initialized BedrockClient with model {self.config.model_id}")
        
        # Add to instances list for cleanup
        BedrockClient._instances.append(self)
        
        # Register shutdown handler if this is the first instance
        if len(BedrockClient._instances) == 1:
            atexit.register(BedrockClient._cleanup_all)
    
    @classmethod
    def _cleanup_all(cls):
        """Clean up all client instances during shutdown"""
        logger.info(f"Cleaning up {len(cls._instances)} BedrockClient instances")
        for instance in cls._instances:
            instance.close()
    
    def close(self):
        """Close connections and clean up resources"""
        logger.info(f"Closing BedrockClient connection for model {self.config.model_id}")
        # Close any active streams
        for stream_id in list(self.active_streams):
            logger.warning(f"Closing active stream {stream_id} during shutdown")
            self.active_streams.remove(stream_id)
    
    def _get_cache_key(self, prompt: str, image_url: str = "") -> str:
        """Generate a cache key for a prompt"""
        # Include model and key parameters in the cache key
        cache_input = f"{prompt}|{self.config.model_id}|{self.config.temperature}|{self.config.max_tokens}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _format_prompt(self, prompt: str, image_url: str = "") -> Dict[str, Any]:
        try:
            if self.config.model_provider == ModelProvider.CLAUDE:
                if "claude-3" in self.config.model_id:
                    if image_url:
                        return {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_url}},
                                    {"type": "text", "text": prompt}
                                ]
                            }],
                            "max_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                            "anthropic_version": "bedrock-2023-05-31"
                        }
                    else:
                        return {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt}
                                ]
                            }],
                            "max_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                            "anthropic_version": "bedrock-2023-05-31"
                        }
                else:
                    return {
                        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                        "max_tokens_to_sample": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "stop_sequences": ["\n\nHuman:"]
                    }

            elif self.config.model_provider == ModelProvider.LLAMA:
                if "llama3-2" in self.config.model_id.lower():  # e.g. "llama-3.2-vision"
                    if image_url:
                        return {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}},
                                    {"type": "text", "text": prompt}
                                ]
                            }],
                            "max_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                            "top_p": 0.9  # optional, adjust if needed
                        }
                    else:
                        return {
                            "messages": [{
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}]
                            }],
                            "max_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                            "top_p": 0.9
                        }
                else:
                    return {
                        "prompt": prompt,
                        "max_gen_len": self.config.max_tokens,
                        "temperature": self.config.temperature
                    }


            elif self.config.model_provider == ModelProvider.MISTRAL:
                return {
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }

            else:
                raise BedrockRequestError(f"Unsupported provider: {self.config.model_provider}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise BedrockRequestError(f"Failed to format prompt: {str(e)}")

    def invoke(self, prompt: str, use_cache: bool = True, image_url: str = "") -> Dict[str, Any]:
        """
        Invoke the model with a prompt
        
        Args:
            prompt: The text prompt to send to the model
            use_cache: Whether to use cached responses (default: True)
            
        Returns:
            The model's response as a dictionary
            
        Raises:
            BedrockClientError: If the API call fails
        """
        request_id = f"req_{int(time.time()*1000)}"
        self.request_count += 1
        ACTIVE_REQUESTS.inc()
        
        try:
            # Check cache first if enabled
            if use_cache:
                cache_key = self._get_cache_key(prompt)
                cached_response = BedrockClient._response_cache.get(cache_key)
                if cached_response:
                    logger.debug(f"[{request_id}] Cache hit for prompt")
                    return cached_response
            
            # Proceed with API call
            body = self._format_prompt(prompt, image_url)
            logger.debug(f"[{request_id}] Invoking model with prompt length: {len(prompt)}")
            
            # Track token usage (approximate)
            TOKEN_COUNTER.labels(model=self.config.model_id, type="input").inc(estimate_tokens(prompt))
            
            # Measure response time
            start_time = time.time()
            response = self.client.invoke_model(
                modelId=self.config.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            response_time = time.time() - start_time
            RESPONSE_TIME.labels(model=self.config.model_id).observe(response_time)
            
            # Check response status
            status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            status = "success" if status_code == 200 else "error"
            REQUEST_COUNTER.labels(model=self.config.model_id, status=status).inc()
            
            if status_code != 200:
                logger.warning(f"[{request_id}] Non-200 status code: {status_code}")
            
            # Handle throttling
            if status_code == 429:
                logger.warning(f"[{request_id}] Rate limited by AWS Bedrock")
                raise BedrockRateLimitError("Rate limit exceeded")
            
            response_body = json.loads(response["body"].read())
            logger.debug(f"[{request_id}] Received response of size: {len(str(response_body))}")
            
            # Track output tokens (approximate)
            if self.config.model_provider == ModelProvider.CLAUDE:
                output_text = response_body.get("completion", "")
            elif self.config.model_provider == ModelProvider.LLAMA:
                output_text = response_body.get("generation", "")
            else:
                output_text = str(response_body)
                
            TOKEN_COUNTER.labels(model=self.config.model_id, type="output").inc(estimate_tokens(output_text))
            
            # Parse the response
            parsed_response = self._parse_response(response_body)
            
            # Cache the response if enabled
            if use_cache:
                BedrockClient._response_cache.set(cache_key, parsed_response)
                
            return parsed_response
            
        except BedrockClientError:
            # Re-raise client errors for retry logic
            REQUEST_COUNTER.labels(model=self.config.model_id, status="error").inc()
            raise
        except boto3.exceptions.Boto3Error as e:
            logger.error(f"[{request_id}] Boto3 error: {str(e)}")
            REQUEST_COUNTER.labels(model=self.config.model_id, status="error").inc()
            raise BedrockClientError(f"AWS service error: {str(e)}")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {str(e)}")
            REQUEST_COUNTER.labels(model=self.config.model_id, status="error").inc()
            raise BedrockClientError(f"Failed to invoke model: {str(e)}")
        finally:
            ACTIVE_REQUESTS.dec()
    
    def invoke_stream(self, prompt: str, image_url: str = "") -> Iterator[Dict[str, Any]]:
        """
        Stream responses from the model for a given prompt
        
        Args:
            prompt: The text prompt to send to the model
            
        Yields:
            Chunks of the model's response
            
        Raises:
            BedrockClientError: If the API call fails
        """
        request_id = f"stream_{int(time.time()*1000)}"
        self.active_streams.add(request_id)
        ACTIVE_REQUESTS.inc()
        
        try:
            body = self._format_prompt(prompt)
            
            # Add streaming parameters based on model provider
            if self.config.model_provider == ModelProvider.CLAUDE:
                body["stream"] = True
            
            logger.debug(f"[{request_id}] Invoking model stream with prompt length: {len(prompt)}")
            TOKEN_COUNTER.labels(model=self.config.model_id, type="input").inc(estimate_tokens(prompt))
            
            # Use streaming API
            response = self.client.invoke_model_with_response_stream(
                modelId=self.config.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # Process the streaming response
            accumulated_text = ""
            token_count = 0
            
            for event in response.get("body", []):
                chunk = event.get("chunk", {})
                if not chunk:
                    continue
                    
                chunk_data = json.loads(chunk.get("bytes", b"{}").decode())
                
                # Handle model-specific streaming responses
                if self.config.model_provider == ModelProvider.CLAUDE:
                    chunk_text = chunk_data.get("completion", "")
                    stop_reason = chunk_data.get("stop_reason")
                elif self.config.model_provider == ModelProvider.LLAMA:
                    chunk_text = chunk_data.get("generation", "")
                    stop_reason = chunk_data.get("stop_reason")
                else:
                    chunk_text = str(chunk_data)
                    stop_reason = None
                
                # Track tokens
                token_count += len(chunk_text.split())
                accumulated_text += chunk_text
                
                # Yield the chunk
                yield {"text": chunk_text, "stop_reason": stop_reason}
                
            # Track total output tokens
            TOKEN_COUNTER.labels(model=self.config.model_id, type="output").inc(estimate_tokens(accumulated_text))
            logger.debug(f"[{request_id}] Stream completed, total output size: {len(accumulated_text)}")
            REQUEST_COUNTER.labels(model=self.config.model_id, status="success").inc()
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in streaming response: {str(e)}")
            REQUEST_COUNTER.labels(model=self.config.model_id, status="error").inc()
            raise BedrockClientError(f"Streaming error: {str(e)}")
        finally:
            if request_id in self.active_streams:
                self.active_streams.remove(request_id)
            ACTIVE_REQUESTS.dec()
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response based on model provider
        
        Args:
            response: Raw API response
            
        Returns:
            Standardized response dictionary
        """
        try:
            if self.config.model_provider == ModelProvider.CLAUDE:
                if "claude-3" in self.config.model_id:
                    content = response.get("content", [])
                    if isinstance(content, list):
                        text = "".join([c.get("text", "") for c in content if c.get("type") == "text"])
                        if not text:
                            logger.warning(f"Claude 3 returned empty 'content': {response}")
                        return {"text": text}
                    else:
                        logger.error(f"Unexpected 'content' format: {type(content)} - {content}")
                        return {"text": ""}
                else:
                    return {"text": response.get("completion", "")}
            elif self.config.model_provider == ModelProvider.LLAMA:
                return {"text": response.get("generation", "")}
            elif self.config.model_provider == ModelProvider.MISTRAL:
                return {"text": response.get("outputs", [{}])[0].get("text", "")}
            return response
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise BedrockResponseError(f"Failed to parse response: {str(e)}")
    
    async def invoke_async(self, prompt: str) -> Dict[str, Any]:
        """
        Async version of invoke method
        
        Args:
            prompt: The text prompt to send to the model
            
        Returns:
            The model's response as a dictionary
        """
        # This is a placeholder for async implementation
        # In a real implementation, you would use aiohttp or similar
        # for now, just call the sync version
        return self.invoke(prompt)



