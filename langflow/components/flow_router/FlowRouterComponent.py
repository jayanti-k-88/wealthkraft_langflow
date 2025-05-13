from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.template import Output
from langflow.field_typing import Text
import json
import requests

class FlowRouterComponent(Component):
    display_name = "Flow Router"

    inputs = [
        MessageTextInput(
            name="parsed_json",
            display_name="Parsed JSON (as string)",
            required=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="user_input",
            display_name="User Input",
            required=True,
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            name="flow_response",
            display_name="Flow Output",
            method="route_flow"
        ),
    ]

    def route_flow(self, *args, **kwargs) -> Message:
        
        base_url = "http://127.0.0.1:7860"
        
        try:
            parsed = json.loads(self.parsed_json)
        except Exception as e:
            return json.dumps({
                "error": f"Invalid JSON format: {str(e)}",
                "raw_input": self.parsed_json
            })

        intent = parsed.get("intent", "qna").lower()

        # Define your flow ID map here
        flow_ids = {
            "savings": "2d942562-d81a-4549-a837-f3fd379f46fd",
            "investment": "503b1cfd-fd5a-48ce-8b1c-ab980af750d0",
            "qna": "b7bf5351-0bd3-4a2c-b485-4f232acb26a6"
        }

        selected_flow_id = flow_ids.get(intent, flow_ids["qna"])
        
        payload = {
            "input_value": self.user_input,
            "output_type": "chat",
            "input_type": "chat"
        }
        
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Assuming the API expects just the 'message' key for chat input
            response = requests.post(
                f"{base_url}/api/v1/run/{selected_flow_id}",
                json=payload,  # Directly passing the message
                headers=headers
            )
            response_json = response.json()
    
            # Extract the most recent message (you might need to tweak based on actual structure)
            message_text = ""
            try:
                message_text = response_json["outputs"][0]["outputs"][0]["results"]["message"]["data"]["text"]
            except Exception as inner_e:
                message_text = f"[Router Warning] Failed to parse message: {str(inner_e)}"
    
            return message_text
    
        except Exception as e:
            return f"[Router Error] Failed to forward to flow: {str(e)}"
