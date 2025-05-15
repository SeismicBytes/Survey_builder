# ai_client.py (Updated for Real Gemini Call)
import os
import logging
import google.generativeai as genai
import time # Keep for potential retries or delays if needed

logger = logging.getLogger(__name__)

_api_key = None
_model = None # Store the initialized model

def configure(api_key: str):
    """Configures the Google Generative AI client."""
    global _api_key, _model
    if not api_key:
        logger.error("AI Client configuration failed: No API key provided.")
        _model = None
        _api_key = None
        return

    _api_key = api_key
    try:
        genai.configure(api_key=_api_key)
        # Initialize the specific model you want to use
        # Make sure the model name is correct for your access level and use case
        # Common model: 'gemini-1.5-flash-latest', 'gemini-1.0-pro', etc.
        _model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info(f"Google Generative AI client configured successfully for model: '{_model.model_name}'")
    except Exception as e:
        logger.critical(f"Failed to configure Google Generative AI or initialize model: {e}", exc_info=True)
        _model = None
        _api_key = None # Reset key if config fails
        # Optionally re-raise the error to halt application startup
        # raise RuntimeError(f"Failed to configure Google AI: {e}")

def is_client_configured() -> bool:
    """Checks if the client (model) is configured and initialized."""
    # Check if the model object was successfully created
    return _model is not None

def generate_with_gemini(prompt: str) -> str | None:
    """
    Generates content using the configured Google Generative AI model.
    Includes basic error handling and safety settings.
    """
    if not is_client_configured():
        logger.error("AI Client not configured. Cannot generate.")
        return None

    logger.info(f"Sending request to Gemini. Prompt length: {len(prompt)} chars. Starts with: {prompt[:150]}...")

    # --- Safety Settings (Adjust as needed) ---
    # Refer to Google AI documentation for details on these settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # --- Generation Configuration (Adjust as needed) ---
    generation_config = {
      "temperature": 0.3, # Lower temperature for more predictable JSON output
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 8192, # Ensure this is sufficient for your expected JSON size
      "response_mime_type": "text/plain", # Request plain text, we'll parse JSON later
    }


    try:
        start_time = time.time()
        # --- Make the API Call ---
        response = _model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            # stream=False # Use stream=True for long responses if needed later
            )
        # ---

        end_time = time.time()
        logger.info(f"Gemini response received in {end_time - start_time:.2f} seconds.")

        # --- Process Response ---
        # Check for safety blocks or other issues before accessing text
        if not response.candidates:
             # This often means the prompt or response was blocked by safety filters
             block_reason = "Unknown"
             try:
                 block_reason = response.prompt_feedback.block_reason
             except Exception:
                 pass # Ignore errors trying to get block reason
             logger.error(f"Gemini response blocked or empty. Block reason: {block_reason}. Full feedback: {response.prompt_feedback}")
             # You might want to return a specific error indicator here
             return json.dumps({"error": f"Gemini response blocked due to safety filters or empty content. Reason: {block_reason}"})


        # Access the text content
        # Ensure you handle potential errors if response.text is not available
        if hasattr(response, 'text'):
            return response.text
        else:
            # If 'text' attribute isn't present, log the response structure for debugging
            logger.error(f"Gemini response did not contain 'text' attribute. Response structure: {response}")
            # Attempt to access content via parts if available (common structure)
            try:
                 if response.candidates[0].content.parts:
                      return response.candidates[0].content.parts[0].text
                 else:
                      raise ValueError("Response candidates[0].content.parts is empty")
            except (AttributeError, IndexError, ValueError) as e:
                 logger.error(f"Could not extract text content using standard attributes or parts: {e}")
                 return json.dumps({"error": "Failed to extract text content from Gemini response."})


    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}", exc_info=True)
        # Return None or an error JSON to indicate failure
        return json.dumps({"error": f"Gemini API call failed: {e}"})