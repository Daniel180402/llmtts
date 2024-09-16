import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3

def main():
    # Initialize TTS engine
    engine = pyttsx3.init()

    # Prompt for Hugging Face access token
    token = input("Please enter your Hugging Face access token: ").strip()

    # Set the model name
    model_name = 'google/gemma-2-9b-it'

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPU
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",          # Automatically allocate the model across available devices
    #     torch_dtype=torch.bfloat16, # Use bfloat16 precision as per the model's requirements
    # )
    # CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,  # Change to float32 if bfloat16 is unsupported
    )

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the TTS voice to Italian if available
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'it' in voice.languages or 'italian' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    else:
        print("Italian voice not found. Using default voice.")

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
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's reply
        # Remove the input prompt to isolate the model's response
        input_length = inputs['input_ids'].shape[1]
        assistant_reply_ids = outputs[0][input_length:]
        assistant_reply = tokenizer.decode(assistant_reply_ids, skip_special_tokens=True).strip()

        # Print and speak the response
        print("Assistant:", assistant_reply, "\n")
        engine.say(assistant_reply)
        engine.runAndWait()

if __name__ == '__main__':
    main()
