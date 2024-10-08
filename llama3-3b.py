import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Enable TensorFloat32 (TF32) for better performance on your 4070 GPU
torch.backends.cuda.matmul.allow_tf32 = True

def speak_text(text, lang='it'):
    """Convert the text to speech."""
    tts = gTTS(text=text, lang=lang)
    tts.save("response_full.mp3")
    
    os.system("start response_full.mp3")  # For Windows
    # Uncomment below line for macOS
    # os.system("afplay response_full.mp3")

def main():
    # Prompt for Hugging Face access token
    token = input("Please enter your Hugging Face access token: ").strip()

    # Model name for Llama-3.2-3B
    model_name = 'meta-llama/Llama-3.2-3B'

    # Load the tokenizer using the Hugging Face token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model on the device, with bfloat16 for efficient GPU usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,  # Use bfloat16 for GPU, float32 for CPU
        device_map="auto",  # Auto device allocation
        use_auth_token=token  # Use Hugging Face token for authentication
    )

    print("You can start chatting with the model now (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Tokenize the user input
        inputs = tokenizer(user_input, return_tensors='pt').to(device)

        # Generate a response using the model
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Reduced length
            do_sample=True,
            top_p=0.9, 
            temperature=0.85, 
            repetition_penalty=1.2, 
            no_repeat_ngram_size=2, # Penalize repetition
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the response
        assistant_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Print the assistant's response
        print("Assistant:", assistant_reply, "\n")

        # Convert the assistant's reply to speech
        speak_text(assistant_reply)

if __name__ == '__main__':
    main()
