import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3

def main():
    # Initialize TTS engine
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"An error occurred while initializing TTS: {e}")
        return  # Exit the script if TTS initialization fails

    # Set TTS voice to Italian if available
    try:
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'it' in voice.languages or 'italian' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        else:
            print("Italian voice not found. Using default voice.")
    except Exception as e:
        print(f"An error occurred while setting TTS voice: {e}")
        return  # Exit the script if voice setup fails

    # Prompt for Hugging Face access token
    token = input("Please enter your Hugging Face access token: ").strip()

    # Load the model and tokenizer
    model_name = 'GroNLP/gpt2-small-italian'  # Use the correct Italian GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    try:
        print("Loading the model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        model.resize_token_embeddings(len(tokenizer))  # Adjust the model's embedding layer if necessary
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit the script if model loading fails
    
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("You can start chatting with the model now (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Prepare the input
        inputs = tokenizer(
            user_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,            # Limit input length
            add_special_tokens=True,   # Include special tokens if necessary
            return_attention_mask=True # Generate attention mask
        ).to(device)
        
        # Generate response
        try:
            print("Generating response...")
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=150,   # Adjust as needed
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated tokens
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the assistant's reply by removing the input
            assistant_reply = generated_text[len(user_input):].strip()
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip to the next iteration
        
        # Print and speak the response
        print("Assistant:", assistant_reply, "\n")
        engine.say(assistant_reply)
        engine.runAndWait()

if __name__ == '__main__':
    main()
