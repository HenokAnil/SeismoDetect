import os
import json

def generate_explanation(structured_data):
    if os.getenv("USE_LLM_EXPLAIN", "false").lower() != "true":
        return template_based_explanation(structured_data)
        
    # Example pseudo-LLM call
    try:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            return template_based_explanation(structured_data) + "\n\n(LLM API key not set, using template)"
            
        # Call LLM here
        return template_based_explanation(structured_data) + "\n\n(LLM explanation would be here)"
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def template_based_explanation(data):
    det_prob = data.get("detection_prob", 0)
    mag = data.get("magnitude_proxy", 0)
    
    explanation = f"We detected a seismic event with {det_prob*100:.1f}% confidence. "
    explanation += f"The estimated magnitude is {mag}. "
    
    picks = data.get("picks", [])
    if picks:
        explanation += f"We found {len(picks)} phase picks. "
        
    return explanation
