from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.template import Output
from langflow.field_typing import Text
import json

class MonthlySavingsComponent(Component):
    display_name = "Monthly Savings Calculator"
    description = "Calculates the required monthly savings based on the total savings amount, duration, and inflation factor."

    inputs = [
        MessageTextInput(name="savings_data", display_name="Savings Data (JSON)", required=True),
    ]

    outputs = [
        Output(name="monthly_savings", display_name="Desired Monthly Savings", method="execute"),
    ]
    
    # Set the inflation rate (can be parameterized if needed)
    inflation_rate = 0.03  # 3% inflation rate, for example

    def execute(self) -> Message:
        # Parse the input JSON (savings_data)
        savings_data = json.loads(self.savings_data)
        
        savings_amount = float(savings_data.get('savings_amount', 0))
        savings_duration = int(savings_data.get('savings_duration', 0))
        
        if savings_amount <= 0 or savings_duration <= 0:
            return "Invalid input data. Ensure savings_amount and savings_duration are positive values."

        # Adjust the savings amount for inflation
        adjusted_savings_amount = savings_amount * (1 + self.inflation_rate) ** savings_duration
        
        # Calculate the number of months
        number_of_months = savings_duration * 12
        
        # Calculate the desired monthly savings
        monthly_savings = adjusted_savings_amount / number_of_months
        
        spending_details_msg = f"Provide your demographics and spending details by clicking on the button **🔍 Analyze My Spending Patterns**"
        
        desired_savings_msg = f"Your goal is to save Rs. {savings_amount} in the next {savings_duration} years.\nTaking into account inflation rate of 3%, your desired monthly savings amount is Rs. {monthly_savings:.2f}.\n{spending_details_msg}, so we can analyze your data and predict your potential savings."
        
        return desired_savings_msg

