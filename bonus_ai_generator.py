from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AITweetGenerator:
    def __init__(self, model_name="gpt2-medium"):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Loaded {model_name} successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None

    def generate_tweet(self, company, tweet_type, message, topic):
        if not self.model:
            return "AI model not available. Using simple generator as fallback."
            
        try:
            prompt = f"Create a {tweet_type} tweet for {company} about {topic}: {message}"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=50,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            return generated_text[len(prompt):].strip()[:280]
        except Exception as e:
            return(f"Generation error: {e}")
            print(f"Error generating tweet: {e}")