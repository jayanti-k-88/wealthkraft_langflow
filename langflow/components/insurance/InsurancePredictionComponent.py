from langflow.custom import Component
from langflow.io import DataFrameInput, Output
from langflow.schema import Message
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InsurancePredictionComponent(Component):
    display_name = "Insurance Prediction"
    description = "Predicts medical insurance charges using a trained pipeline."
    icon = "activity"
    name = "InsurancePredictionComponent"

    inputs = [
        DataFrameInput(
            name="input_df",
            display_name="Input DataFrame",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            name="insurance_prediction",
            display_name="Predicted Charges",
            method="predict",
        )
    ]

    def predict(self) -> Message:
        
        input_df = self.input_df

        try:
            input_df['region'] = 'southeast'
            # Load your trained model
            model = joblib.load('./model/insurance_prediction_pipeline.pkl')

            # Run prediction
            prediction = model.predict(input_df)
            predicted_charges = prediction[0]
            coverage_level = str(input_df.iloc[0]['coverage_level'])

            # Classify risk
            if predicted_charges <= 10000:
                risk_class = "Low Risk"
            elif predicted_charges <= 20000:
                risk_class = "Moderate Risk"
            else:
                risk_class = "High Risk"
    
            # Determine insurance product
            product_key = f"{coverage_level} - {risk_class}"
            product_info = self.insurance_products().get(product_key, "No details available for this product.")
    
            # Compose response
            response = (
                f"💸 **Predicted Medical Charges** : Rs. {predicted_charges:,.2f}\n\n"
                f"🛡️ **Risk Category** : {risk_class}\n\n"
                f"📦 **Recommended Insurance Product** : {product_key}\n\n"
                f"📘 **Product Info** : {product_info}"
            )
    
            return Message(text=response)

        except Exception as e:
            print(e)
            return Message(text=f"❌ Error in prediction: {str(e)}")
         
           
    def insurance_products(self) -> dict: 
        
        return {
            "Basic - Low Risk": (
                "You’ve been categorized as a low-risk individual based on your health data and predicted expenses. "
                "Our *Basic - Low Risk* plan is ideal for someone like you — offering essential medical coverage at a very low premium. "
                "It includes doctor consultations, basic diagnostics, and emergency care, making it a smart and affordable choice."
            ),
            "Basic - Moderate Risk": (
                "With moderate predicted health expenses, the *Basic - Moderate Risk* plan offers an affordable yet slightly expanded coverage. "
                "It provides support for minor procedures, prescribed medications, and a wider network of hospitals — perfect for managing costs without overpaying."
            ),
            "Basic - High Risk": (
                "Although you prefer basic coverage, your predicted medical costs suggest a higher health risk. "
                "The *Basic - High Risk* plan is designed to ensure you’re still protected from the most essential treatments, "
                "while keeping premiums manageable. It covers emergency treatments and limited inpatient care."
            ),
            "Standard - Low Risk": (
                "You’ve chosen our standard coverage while being a low-risk individual — excellent balance. "
                "The *Standard - Low Risk* plan includes not just hospitalization, but also preventive checkups, outpatient benefits, "
                "and wellness programs that help you stay healthy affordably."
            ),
            "Standard - Moderate Risk": (
                "Your health profile shows moderate risk, and our *Standard - Moderate Risk* plan is built just for that. "
                "It offers a broader range of outpatient and specialist services, diagnostic tests, and semi-private room eligibility — "
                "giving you peace of mind without high premiums."
            ),
            "Standard - High Risk": (
                "As someone with higher predicted expenses, the *Standard - High Risk* plan offers comprehensive support. "
                "It includes coverage for chronic condition management, frequent hospital visits, and lab tests — "
                "balanced against reasonable premiums to give you dependable care."
            ),
            "Premium - Low Risk": (
                "You’ve opted for a premium plan with a low-risk profile — a great way to future-proof your health. "
                "The *Premium - Low Risk* plan includes annual executive checkups, dental/vision benefits, wellness coaching, "
                "and a concierge claim process. It’s designed for proactive, long-term health optimization."
            ),
            "Premium - Moderate Risk": (
                "Your profile suggests moderate risk, and the *Premium - Moderate Risk* plan meets that with extensive coverage. "
                "It includes multiple specialty consultations, access to super-specialty hospitals, advanced diagnostics, "
                "and priority claim settlements — ensuring you're always ahead in care."
            ),
            "Premium - High Risk": (
                "With higher predicted charges, our *Premium - High Risk* plan delivers unmatched coverage. "
                "It includes critical illness insurance, major surgery coverage, post-op care, mental health support, "
                "and 24x7 health assistance — ideal for those who need full-spectrum protection."
            )
        }
