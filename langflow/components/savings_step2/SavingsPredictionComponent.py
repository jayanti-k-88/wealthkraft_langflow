# from langflow.field_typing import Data
from langflow.custom import Component
from langflow.io import MessageTextInput, Output, DataFrameInput
from langflow.schema import Data
import joblib
import pickle


class SavingsPredictionComponent(Component):
    display_name = "Savings Prediction Component"
    description = "Use as a template to create your own component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "SavingsPredictionComponent"

    inputs = [
        DataFrameInput(
            name="input_df",
            display_name="Input DataFrame",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(name="savings_prediction", display_name="Predicted Savings Info", method="execute"),
    ]

    def execute(self) -> Message:
        input_df = self.input_df
        # Convert and clean input values
        row = input_df.iloc[0]
        income = float(row['Income'])
        rent = float(row['Rent'])
        loan_repayment = float(row['Loan_Repayment'])
        insurance = float(row['Insurance'])
        other_expenditure = float(row['total_expenditure_others'])
        desired_savings = float(row['Desired_Savings'])
        
        disposable_income = income - other_expenditure - rent - loan_repayment - insurance
    
        # Check conditions before prediction
        if disposable_income < desired_savings:
            return Message(text="⚠️ Your disposable income is less than your desired savings. Please reduce expenses or adjust your goal.")
        
        elif disposable_income == desired_savings:
            return Message(text="✅ Your disposable income exactly matches your desired savings. You are on track!")
    
        # Only proceed with prediction if above two conditions are not met
        input_df['disposable_income_calc'] = disposable_income
    
        # Load model and predict
        model = joblib.load('./model/savings_model_pipeline.pkl')
        prediction = model.predict(input_df)
    
        predicted_savings = prediction[0]
        
        total_potential_savings = desired_savings + predicted_savings
        
        output_df = input_df
        output_df['total_potential_savings'] = total_potential_savings
        output_df['disposable_income'] = disposable_income
        
        indivdual_potential_savings_df = self.predict_and_split_saving(output_df)
        indivdual_potential_savings_table = indivdual_potential_savings_df.to_markdown(index=False)
    
        # Remaining checks
        if disposable_income < (desired_savings + predicted_savings):
            message = "✅ You can potentially achieve your desired savings with a bit more financial discipline."
        elif disposable_income == (desired_savings + predicted_savings):
            message = "🎯 Your predicted savings aligns perfectly with your financial goals."
        else:
            message = "💰 You have a healthy surplus beyond your savings goal. Consider exploring investment opportunities by clickling on the Investments menu."
            

    
        return Message(text=f"Predicted Additional Savings: Rs. {predicted_savings:.2f}\n\n🐷As per your demographics and spending patterns, you can potentially save Rs. {desired_savings:.2f} + Rs. {predicted_savings: .2f} = Rs. {total_potential_savings: .2f}.\n\n{message}\n\nHere is a breakdown of individual potential savings for Rs. {total_potential_savings: .2f} as per your data: \n\n{indivdual_potential_savings_table}")
        
        
    def predict_and_split_saving(self, df_input):
        """
        Function to Predict Clusters and Split Total Saving into Categories
        based on pretrained models (Scaler, Encoder, PCA, KMeans) and Ratio DataFrame.
        """
    
        with open('./model/Kmeans_Req_Model.pkl', 'rb') as file:
            models = pickle.load(file)
            #print(models.keys())
            scaler = models['scaler']
            encoder = models['One_hot_encoding']
            pca = models['pca']
            kmeans = models['kmeans']
            ratio_df = models['dataframe']
    
        #print(ratio_df.columns)
    
        # Preprocessing
        # One-hot encode categorical features
        encoded_data = encoder.transform(df_input[['Occupation', 'City_Tier']])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Occupation', 'City_Tier']))
        encoded_df.index = df_input.index
    
        # Standard scale numerical features
        column_req =  ['Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Desired_Savings', 'disposable_income', 'total_expenditure_others', 'total_potential_savings']
    
        scaled_data = scaler.transform(df_input[column_req])
    
        scaled_df = pd.DataFrame(scaled_data, columns = column_req, index=df_input.index)
    
        # Combine encoded and scaled features
        X_processed = pd.concat([scaled_df, encoded_df], axis=1)
    
        # PCA Transformation
        X_pca = pca.transform(X_processed)
    
        #  Predict clusters
        cluster_labels = kmeans.predict(X_pca)
        df_input['Predicted_Cluster'] = cluster_labels
    
        #  Map ratio based on predicted cluster
        df_cluster_ratio = ratio_df.set_index('Cluster')
        ratios = df_cluster_ratio.loc[cluster_labels].reset_index(drop=True)
    
    
        #  Multiply and get category savings
        Potential_Saving_Category = ['Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 'Potential_Savings_Education', 'Potential_Savings_Miscellaneous']
    
        for category in Potential_Saving_Category :
            df_input[category] = df_input['total_potential_savings'] * ratios[category]
    
        #Create Dictionary for the all the saving columns
        #saving_cols = (for col in Potential_Saving_Category)
        saving_cols = [col for col in Potential_Saving_Category]
        result_dicts = df_input[saving_cols].to_dict(orient='records')
    
        # Extract the only dictionary
        data = result_dicts[0]
        
        # Filter and process keys/values
        prefix = "Potential_Savings_"
        cleaned_data = {
            key.replace(prefix, "").replace("_", " "): round(value, 2)
            for key, value in data.items() if key.startswith(prefix)
        }
        
        # Format as a table
        df = pd.DataFrame([cleaned_data])

        return df
            