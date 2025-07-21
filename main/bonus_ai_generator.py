import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging to see errors and information
logging.basicConfig(level=logging.INFO)

# --- Other Potential Small Models to Try ---
# "sshleifer/tiny-gpt2"
# "microsoft/phi-2" (Might be too large for 512MB RAM)
# "google/gemma-2b" (Likely too large for 512MB RAM)

class AITweetGenerator:
    def __init__(self, model_name="distilgpt2"):
        """
        Initializes the tweet generator. "distilgpt2" is the default as it provides
        the best balance of performance and low memory usage for your server.
        """
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")

        try:
            logging.info(f"Loading tokenizer for model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            logging.info(f"Loading model: {model_name}")
            # Using AutoModelForCausalLM is more flexible for different model types
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True # Crucial for low-memory environments
            ).to(self.device)

            # Apply int8 dynamic quantization
            logging.info("Applying int8 dynamic quantization to the model...")
            self.model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            # ---

            logging.info(f"Successfully loaded and quantized model '{model_name}'!")

        except Exception as e:
            logging.error(f"Model initialization failed: {e}", exc_info=True)
            self.model = None

    def generate_tweet(self, company, tweet_type, message, topic):
        if not self.model:
            return "AI model is not available due to an initialization error."

        try:
            prompt = f"A {tweet_type} tweet for the company '{company}' about '{topic}': {message}"
            
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