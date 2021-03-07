#
# Encoder and Decoder Models
#

#Import pytorch modules.
import torch
import torch.nn as nn
import torchvision.models as models


#Encoder and decoder class.

class cnn_encoder_resnet50(nn.Module):
    def __init__(self,embed_size):
        super(cnn_encoder_resnet50,self).__init__()
        resnet50=models.resnet50(pretrained=true)
        for param in resnet50.parameters():
            param.requires_grad_(False)
        
        modules=list(resnet50.children())[:-1]
        self.resnet50=nn.Sequential(*modules)
        self.embed=nn.Linear(resnet50.fc.in_features,embed_size)
    
    def forward(self,images):
        features=self.resnet50(images)
        features=features.view(features.size(0),-1)
        features=self.embed(features)
        return features

class rnn_decoder(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1):
        super(rnn_decoder,self).__init__()

        self.word_embedding=nn.Embedding(vocab_size,embed_size)

        self.lstm_layer=nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)

        self.fc_layer=nn.Linear(hidden_size,vocab_size)

    def forward(self,features,captions):

        captions=captions[:,:-1]
        word_embedding=self.word_embedding(captions)

        features=features.unsqueeze(1)

        word_embedding=torch.cat((features,word_embedding),1)

        lstm_out,hidden_state=self.lstm_layer(word_embedding)

        output=self.fc_layer(lstm_out)

        return output

