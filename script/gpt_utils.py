import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm.notebook import tqdm

class GPT_Dataset(Dataset):
    ''' processing data samples, so DataLoader wraps an iterable around this Dataset to enable easy access to the samples. '''
    def __init__(self, text , tokenizer , max_len):
      super(GPT_Dataset).__init__()
      self.text = text
     
      self.tokenizer = tokenizer
      self.max_len = max_len

      

    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())
      input_ids = self.tokenizer.encode(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          truncation='longest_first'
      )     
    
      attention_mask = [1] * len(input_ids)

      # pad up to the sequence length.
      padding_length = self.max_len - len(input_ids)
      input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
      attention_mask = attention_mask + ([0] * padding_length)    
      
      return {
        'input_ids' : torch.tensor(input_ids, dtype=torch.long),
        'attention_mask' : torch.tensor(attention_mask, dtype=torch.long)      
    }


class GPT(nn.Module):
    ''' define the GPT model '''

    def __init__(self, model ):
      
      super(GPT, self).__init__(  )

      self.model = model
    
      

    #define the forward pass
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        doc_id = None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

      #pass the inputs to the model  
     outputs = self.model(
            input_ids,
            attention_mask=attention_mask)
     last_hidden_state = outputs['last_hidden_state']
     weights = (
      torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
      .unsqueeze(0)
      .unsqueeze(-1)
      .expand(last_hidden_state.size())
      .float().to(last_hidden_state.device)
     )

      # Get attn mask of shape [bs, seq_len, hid_dim]
     input_mask_expanded = (
          attention_mask
          .unsqueeze(-1)
          .expand(last_hidden_state.size())
          .float()
      )

     # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
     sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
     sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
     pooled_output = sum_embeddings / sum_mask
     #print(pooled_output.shape)
     return pooled_output

def emb_fn(data_loader, model, device ):
    ''' iterates over the data loader, passes the samples to the model , returns embedded data '''
    model.eval()
    with torch.no_grad():

      embs= None
    
      for data in tqdm(data_loader, total=len(data_loader)):
    
        for k, v in data.items():
            data[k] = v.to(device)
        logits = model(**data)
       
        if logits is not None:        
          embs = logits if embs is None else torch.cat((embs, logits), dim=0)
        
        

      embs = embs.detach().cpu().numpy()
         
    return embs

def init_gpt(sents):
  '''main flow'''
  #defults 
  if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    

  # If not...
  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu") 

  gpt_tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
  gpt_model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
  BATCH_SIZE = 16

  gpt = GPT(gpt_model)
  gpt.to(device)

  to_dataset = GPT_Dataset(
     text =sents  , tokenizer = gpt_tokenizer , max_len = 20)
  

  dataloader = torch.utils.data.DataLoader(
    dataset=to_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
  )

  return  emb_fn(dataloader,gpt, device )


