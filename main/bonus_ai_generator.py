import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
logging.basicConfig(level=logging.INFO)
class AITweetGenerator:
    def __init__(self, model_name="gpt2"):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        try:
            logging.info(f"Loading tokenizer for model: {model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info(f"Loading model: {model_name}")
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            ).to(self.device)
            logging.info("Applying int8 dynamic quantization to the model...")
            self.model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logging.info(f"Successfully loaded and quantized model '{model_name}'!")
        except Exception as e:
            logging.error(f"Model initialization failed: {e}", exc_info=True)
    def generate_tweet(self, company, tweet_type, message, topic):
        if not self.model:
            return "AI model is not available due to an error."
        try:
            prompt = f"Create a {tweet_type} tweet for the company {company} about {topic}: {message}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=100,
                truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            response = generated_text[len(prompt):].strip()
            return response if response else "Generated an empty response, please try again."
        except Exception as e:
            logging.error(f"Error during tweet generation: {e}", exc_info=True)
            return "An error occurred while generating the AI tweet."