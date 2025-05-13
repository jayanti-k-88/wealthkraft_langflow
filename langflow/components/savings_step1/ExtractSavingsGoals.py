import ast
import json
import re
from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.template import Output
from langflow.field_typing import Text

class ExtractSavingsGoals(Component):
    display_name = "Extract Savings Goals"

    inputs = [
        MessageTextInput(name="response_text", display_name="LLM Response"),
    ]

    outputs = [
        Output(name="savings_data", display_name="Savings Data", method="extract_data"),
    ]

    def extract_data(self, *args, **kwargs) -> Message:
        try:
            # Extract the dictionary portion from the response
            match = re.search(r"\{[\s\S]*?\}", self.response_text)
            if match:
                raw_dict_str = match.group()
                parsed_dict = ast.literal_eval(raw_dict_str)  # safer than eval()
                return json.dumps(parsed_dict)
            else:
                return json.dumps({"error": "No dictionary found in response."})
        except Exception as e:
            return json.dumps({"error": f"Parsing failed: {str(e)}"})
