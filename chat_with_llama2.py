import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3

def main():
    # Initialize TTS engine
    engine = pyttsx3.init()
    
    # Prompt for Hugging Face access token
    token = input("Please enter your Hugging Face access token: ").strip()
    
    # Replace deprecated parameter 'use_auth_token' with 'token'
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    
    # Determine appropriate torch_dtype
    if torch.cuda.is_available():
        torch_dtype = torch.float16  # Use float16 on GPU
    else:
        torch_dtype = torch.float32  # Use float32 on CPU
    
    try:
        print("Loading the model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=token
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return  # Exit the script if model loading fails
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("You can start chatting with the model now (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Prepare the input
        inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Generate response
        try:
            print("Generating response...")
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,  # Reduced for faster generation
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Calculate the length of the input
            input_length = inputs['input_ids'].shape[1]
            
            # Extract the generated tokens (excluding the input)
            generated_tokens = outputs[0, input_length:]
            
            # Decode the generated tokens
            assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            continue  # Skip to the next iteration
        
        # Print and speak the response
        print("Assistant:", assistant_reply, "\n")
        engine.say(assistant_reply)
        engine.runAndWait()

if __name__ == '__main__':
    main()
