# from langflow.field_typing import Data
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling


class QnAComponent(Component):
    display_name = "QnA Component"
    description = "Use as a template to create your own component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "QnAComponent"

    inputs = [
        MessageTextInput(
            name="prompt",
            display_name="Prompt",
            info="This is a Prompt Input",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="model_load_inference"),
    ]

        
    def generate_response(self, model, tokenizer):

        prompt_text = f"Answer this question in about 50 words: {self.prompt}"
        
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")      # 'pt' for returning pytorch tensor
    
        # Create the attention mask and pad token id
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            do_sample=False,
            attention_mask=attention_mask,
            eos_token_id=pad_token_id
        )
    
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    
    def model_load_inference(self) -> Message: 
        # Load the fine-tuned model and tokenizer
        model_path = "./model/gpt2-qna-finetuned/checkpoint-42000/"
        my_model = GPT2LMHeadModel.from_pretrained(model_path)
        my_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        response = self.generate_response(my_model, my_tokenizer)
        
        trimmed = response[(len(self.prompt) + 49):].strip()
        last_punct_idx = max(trimmed.rfind('.'), trimmed.rfind('!'), trimmed.rfind('?'))
        
        if last_punct_idx != -1: 
            return trimmed[:last_punct_idx + 1].strip()
        
        return trimmed
        
