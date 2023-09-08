# FineTune_ruDialoGPT

## Description:
Fine-tune ruDialoGPT-medium using chat history from any Telegram conversation and implement an application interface using the Telegram Bot API to interact with the fine-tuned model. I've decided to use conversational data from the popular chat platform 'Двач' because it contains a vast number of messages, and it frequently includes highly toxic content. These reasons should significantly impact the final results.
The model was trained on data consisting of three messages of context and respons."

## Data processing
I collected all of the data using a Python script called prepare_messages.py. This script scrapes the data into a pandas DataFrame with columns: context_3, context_2, context_1, and response. Since I had a sufficient amount of data, I removed all rows that contained any empty columns. After it I comple the next ponts:
* remove urls
* remove special telegram symbols for text-style
* remove all english characters

## Data tokenization
The main idea was concate all dialogue turns in one row with special separation tokens. The concated turns looks like:  
<div style="border: 2px solid black; padding: 10px;">
  **\<sp1\>context_3\<sp2\>context_2\<sp1\>context_1\<sp2\>response\<sp1\>**
</div>



