import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer 
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # Linear layer 
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # step 1 : Creating embedded word vectors for each token in a batch of captions
        embeds = self.word_embeddings(captions[:,:-1]) # batch_size,cap_length -> batch_size,cap_length-1,embed_size

        # step 2 : Concatenating the input features and caption inputs
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1) # batch_size, caption (sequence) length, embed_size

        # step 3 : Feeding into LSTM layer
        lstm_out, _ = self.lstm(inputs) # lstm_out.shape : batch_size, caplength, hidden_size

        # step 4 : Converting LSTM outputs to word prediction
        outputs = self.linear(lstm_out) # outputs.shape : batch_size, caplength, vocab_size
        
        return outputs  #[:,:-1,:] : Discarding the last output of each sample in the batch.

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        caption = []

        # Initializing the hidden state & Sending it to the same device as the inputs
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

        # Feeding the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)              # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1)                 # outputs shape : (1, vocab_size)
            wordid  = outputs.argmax(dim=1)
            caption.append(wordid.item())
            
            # Preparing input for next iteration
            inputs = self.word_embeddings(wordid.unsqueeze(0)) # inputs shape : (1, 1, embed_size)
          
        return caption