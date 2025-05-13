from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.template import Output
from langflow.schema import Data
import json

class InsuranceJSONToDataFrameComponent(Component):
    display_name = "Insurance JSON to DataFrame"
    description = "Parses JSON input containing a 'insurance' dictionary and returns a Data object."

    inputs = [
        MessageTextInput(
            name="json_input",
            display_name="JSON Input",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="data_object",
            display_name="Data Object",
            method="to_data_object",
        ),
    ]

    def to_data_object(self) -> Data:
        try:
            # Parse the input JSON
            full_data = json.loads(self.json_input)

            # Extract the nested 'savings' dictionary
            insurance_data = full_data.get("insurance")
            if not insurance_data:
                raise ValueError("Missing 'savings' key in input JSON.")

            # Pass the extracted savings data to the Data object
            data_object = Data(**insurance_data)
            return data_object

        except Exception as e:
            raise ValueError(f"Invalid JSON input: {e}")
