import telebot
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch 

chat_history_dict = {}

def generate_response(message, chat_history, chat_id):
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

    if chat_history.count('@@ВТОРОЙ@@') == 2:
        chat_history_dict[chat_id] = ""
    else:
        chat_history_dict[chat_id] = chat_history

    return response

def get_chat_history(chat_id):
    if chat_id not in chat_history_dict:
        chat_history_dict[chat_id] = ""

    return chat_history_dict[chat_id]


def main():    
    model_dir = "model\epoch1_ruDialoGPT_dvach"
    model = AutoModelWithLMHead.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TOKEN = input("Введите BOT-TOKEN: ")
    
    bot = telebot.TeleBot(TOKEN)

    @bot.message_handler(commands=['start'])
    def handle_start(message):
        with open('data/text_bot.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()  

        hello_text = lines[:2]
        bot.send_message(message.chat.id, hello_text)

    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        user_input = message.text
        chat_history = get_chat_history(message.chat.id)
        response = generate_response(user_input, chat_history, message.chat.id)
        bot.reply_to(message, response)

    try:
        bot.polling()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Bot stopped.")

    



if __name__ == "__main__":
    main()
