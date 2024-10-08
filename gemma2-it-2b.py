import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

torch.backends.cuda.matmul.allow_tf32 = True

def speak_text(text, lang='it'):
    """Convert the text to speech."""
    tts = gTTS(text=text, lang=lang)
    tts.save("response_full.mp3")
    
    os.system("start response_full.mp3")

def main():
    token = input("Please enter your Hugging Face access token: ").strip()

    model_name = 'google/gemma-2-2b-it'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,  
        device_map="auto",  
        use_auth_token=token 
    )

    print("You can start chatting with the model now (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        messages = [
            {"role": "user", "content": user_input},
        ]
        
        
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors='pt', return_dict=True
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  
            num_beams=5,         
            top_p=0.8,           
            temperature=0.7,     
            repetition_penalty=1.1,  
            no_repeat_ngram_size=3,  
            pad_token_id=tokenizer.eos_token_id,
        )

        assistant_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("Assistant:", assistant_reply, "\n")

        speak_text(assistant_reply)

if __name__ == '__main__':
    main()
