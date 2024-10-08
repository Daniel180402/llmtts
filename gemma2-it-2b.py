import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

torch.backends.cuda.matmul.allow_tf32 = True

def speak_text(text, lang='it'):
    if not text:
        print("Error: No text to speak")
        return
    tts = gTTS(text=text, lang=lang)
    tts.save("response_full.mp3")
    os.system("start response_full.mp3")

def filter_response(response):
    lines = response.splitlines()
    filtered_lines = [line for line in lines if not (line.strip().startswith('-') or line.strip().startswith('*'))]
    return ' '.join(filtered_lines)

def ensure_complete_sentence(text):
    last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punctuation != -1:
        return text[:last_punctuation+1]
    return text

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

        final_input = f"{user_input}. Rispondi direttamente alla domanda senza elencare esempi o informazioni generali. Massimo 100 parole."

        messages = [
            {"role": "user", "content": final_input},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors='pt', return_dict=True
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150, 
            num_beams=3,
            top_p=0.5,
            temperature=0.4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        assistant_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        input_length = inputs['input_ids'].shape[1]
        assistant_reply_ids = outputs[0][input_length:]
        assistant_reply = tokenizer.decode(assistant_reply_ids, skip_special_tokens=True).strip()

        filtered_reply = filter_response(assistant_reply)
        complete_reply = ensure_complete_sentence(filtered_reply)

        if complete_reply:
            print("Assistant:", complete_reply, "\n")
            speak_text(complete_reply)
        else:
            print("Assistant did not provide a response.")

if __name__ == '__main__':
    main()
