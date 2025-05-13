from langflow.custom import Component
from langflow.inputs import StrInput, FloatInput
from langflow.template import Output
from langflow.field_typing import Text
import json
import re
from typing import Optional, Dict, Any

class LLMJsonExtractorComponent(Component):
    display_name = "LLM JSON Extractor"

    inputs = [
        MessageTextInput(
            name="llm_output",
            display_name="LLM Output",
            required=True,
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            name="parsed_json",
            display_name="Parsed LLM JSON",
            method="extract_data"
        ),
    ]

    def extract_data(self, *args, **kwargs) -> Message:
        try:
            # Extract the dictionary portion from the response
            match = re.search(r"\{[\s\S]*?\}", self.llm_output)
            if match:
                raw_dict_str = match.group()
                parsed_dict = ast.literal_eval(raw_dict_str)  # safer than eval()
                return json.dumps(parsed_dict)
            else:
                return json.dumps({"error": "No dictionary found in response."})
        except Exception as e:
            return json.dumps({"error": f"Parsing failed: {str(e)}"})
