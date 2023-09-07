import torch 
from transformers import AutoTokenizer, AutoModelWithLMHead
import warnings

def main():
    warnings.filterwarnings("ignore")

    with open('data/text_bot.txt', 'r', encoding='utf-8') as file:
        file_contents = file.read()

    print(file_contents)
    
    tokenizer = AutoTokenizer.from_pretrained('model/epoch1_ruDialoGPT_dvach')
    model = AutoModelWithLMHead.from_pretrained('model/epoch1_ruDialoGPT_dvach')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    chat_history = ""

    while(True):
        message = input("Вы: ")

        if message == 'exit':
            break

        if len(chat_history) == 0:
            encoded_message = tokenizer(f'@@ПЕРВЫЙ@@ {message} @@ВТОРОЙ@@', return_tensors='pt')
        else:
            encoded_message = tokenizer(f'{chat_history} {message} @@ВТОРОЙ@@', return_tensors='pt')

        if encoded_message['input_ids'].shape[1] > 1024:
            encoded_message = tokenizer(f'@@ПЕРВЫЙ@@ {message} @@ВТОРОЙ@@', return_tensors='pt')

        encoded_message.to(device)

        generated_token_ids = model.generate(
            **encoded_message,
            top_k=10,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )

        context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
        response = context_with_response[0].split('@@ВТОРОЙ@@')[-1].replace('@@ПЕРВЫЙ@@', '') 
        
        chat_history = context_with_response[0]

        print(f'RuDialoGPT_dvach: {response}')

        if chat_history.count('@@ВТОРОЙ@@') == 2:
            chat_history = ""
        
if __name__ == "__main__":
    main()