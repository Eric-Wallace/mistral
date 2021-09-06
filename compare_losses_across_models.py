#wget https://storage.googleapis.com/mistral-models/gpt2-small/alias-gpt2-small-x21/alias-x21-checkpoint-400000.zip
#wget https://storage.googleapis.com/mistral-models/gpt2-small/battlestar-gpt2-small-x49/battlestar-x49-checkpoint-400000.zip
#wget https://storage.googleapis.com/mistral-models/gpt2-small/caprica-gpt2-small-x81/caprica-x81-checkpoint-400000.zip
#wget https://storage.googleapis.com/mistral-models/gpt2-small/darkmatter-gpt2-small-x343/darkmatter-x343-checkpoint-400000.zip
#wget https://storage.googleapis.com/mistral-models/gpt2-small/expanse-gpt2-small-x777/expanse-x777-checkpoint-400000.zip

model1name = 'alias-x21-checkpoint-400000'   
model2name = 'caprica-x81-checkpoint-400000'    
model3name = 'darkmatter-x343-checkpoint-400000'
model4name = 'battlestar-x49-checkpoint-400000'
model5name = 'expanse-x777-checkpoint-400000'

from src.models.mistral_gpt2 import MistralGPT2LMHeadModel
import transformers
import torch
from datasets import load_dataset

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

model1 = MistralGPT2LMHeadModel.from_pretrained(model1name)
model2 = MistralGPT2LMHeadModel.from_pretrained(model2name)
model3 = MistralGPT2LMHeadModel.from_pretrained(model3name)
model4 = MistralGPT2LMHeadModel.from_pretrained(model4name)
model5 = MistralGPT2LMHeadModel.from_pretrained(model5name)
model1.eval().cuda();
model2.eval().cuda();
model3.eval().cuda();
model4.eval().cuda();
model5.eval().cuda();


dataset = load_dataset("stas/openwebtext-10k") # first 10k examples
#print(dataset['train'][0]['text'])

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss).item()

results = []
results_normalized = []
for idx, sample in enumerate(dataset['train']): # TODO batch me
    if idx > 1000:
        break
    model1ppl = calculatePerplexity(sample['text'][0:2000], model1, tokenizer)
    model2ppl = calculatePerplexity(sample['text'][0:2000], model2, tokenizer)
    model3ppl = calculatePerplexity(sample['text'][0:2000], model3, tokenizer)
    model4ppl = calculatePerplexity(sample['text'][0:2000], model4, tokenizer)
    model5ppl = calculatePerplexity(sample['text'][0:2000], model5, tokenizer)
    highest_ppl = max([model1ppl, model2ppl, model3ppl, model4ppl, model5ppl])
    smallest_ppl = min([model1ppl, model2ppl, model3ppl, model4ppl, model5ppl])
    results.append(highest_ppl - smallest_ppl)
    results_normalized.append((highest_ppl - smallest_ppl) / (smallest_ppl))

import pickle
with open('results.pkl', "wb") as output_file:
    pickle.dump(results, output_file)
with open('results_normalized.pkl', "wb") as output_file:
    pickle.dump(results_normalized, output_file)
