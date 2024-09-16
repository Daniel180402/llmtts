import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Enable TF32 in PyTorch
torch.backends.cuda.matmul.allow_tf32 = True

def speak_text(text, lang='it'):
    """Convert the entire text to speech."""
    tts = gTTS(text=text, lang=lang)
    tts.save("response_full.mp3")
    
    # Use "start" for Windows to play the audio
    os.system("start response_full.mp3")  
    
    # For macOS, use "afplay"
    # os.system("afplay response_full.mp3")

def main():
    # Prompt for Hugging Face access token
    token = input("Please enter your Hugging Face access token: ").strip()

    # Set the model name
    model_name = 'google/gemma-2-9b-it'

    # Load the tokenizer with token authentication
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Determine the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model with GPU or CPU support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically allocate the model across available devices
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,  # Use float16 for GPU, float32 for CPU
        token=token  # Use the correct token argument
    )

    print("You can start chatting with the model now (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Prepare the conversation as per the chat template
        messages = [
            {"role": "user", "content": user_input},
        ]

        # Apply the chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            return_dict=True,
            add_special_tokens=False,
        ).to(device)

        # Generate the response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Increase to generate a longer response
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's reply
        input_length = inputs['input_ids'].shape[1]
        assistant_reply_ids = outputs[0][input_length:]
        assistant_reply = tokenizer.decode(assistant_reply_ids, skip_special_tokens=True).strip()

        # Print the complete response
        print("Assistant:", assistant_reply, "\n")

        # Convert the complete response to speech using gTTS
        speak_text(assistant_reply)

if __name__ == '__main__':
    main()
