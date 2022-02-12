import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from glob import glob
from librosa.display import specshow
import torch
from torch import nn
from torchvision import models, transforms
import cv2

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

model = models.vgg16(pretrained=True)
# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

#Import labels for the dataset.
#It is accepted that there are no missing data in the dataset.
data_df = pd.read_csv('ff1010bird_metadata.csv')
#Import audio files.
audio_files=glob('ff1010bird_wav/wav/*.wav')

for i in range(len(audio_files)):
    #print(i)
    audio, sfreq =lr.load(audio_files[i])
    
    #Constant-Q chromagram
    Chroma_CQT = lr.feature.chroma_cqt(audio,sfreq)
    #fig = plt.subplots()
    img = lr.display.specshow(Chroma_CQT)
    plt.savefig('Chroma_CQT.jpg',bbox_inches='tight',pad_inches = 0)

    # Will contain the feature
    features = []
    img = cv2.imread('Chroma_CQT.jpg')
    img = transform(img)
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    with torch.no_grad():
        feature = new_model(img)
        features.append(feature.cpu().detach().numpy().reshape(-1))
    # Convert to NumPy Array
    features = np.array(features)
    features = np.column_stack([int(audio_files[i].split('\\')[1].split('.')[0]), features])
    if i == 0:
        Chroma_CQT_df = pd.DataFrame(features, index = [str(i)])
    else:
        Chroma_CQT_df=Chroma_CQT_df.append(pd.DataFrame(features,index = [str(i)]))  
        
colname = []
for i in range(len(Chroma_CQT_df.columns)):
    if i == 0:
        colname.append('itemid')
    else:
        colname.append('x' + str(i))

Chroma_CQT_df.columns = colname
Data_Chroma_CQT=pd.merge(data_df, Chroma_CQT_df, on='itemid', how='left')
Data_Chroma_CQT.to_pickle("Data_Chroma_CQT.pkl")

for i in range(len(audio_files)):
    #print(i)
    audio, sfreq =lr.load(audio_files[i])
    
    #Short-time Fourier transform
    Chroma_STFT = lr.feature.chroma_stft(audio,sfreq)
    #fig = plt.subplots()
    img1 = lr.display.specshow(Chroma_STFT)
    plt.savefig('Chroma_STFT.jpg',bbox_inches='tight',pad_inches = 0)

    # Will contain the feature
    features1 = []
    img1 = cv2.imread('Chroma_STFT.jpg')
    img1 = transform(img1)
    img1 = img1.reshape(1, 3, 448, 448)
    img1 = img1.to(device)
    with torch.no_grad():
        feature = new_model(img1)
        features1.append(feature.cpu().detach().numpy().reshape(-1))
    # Convert to NumPy Array
    features1 = np.array(features1)
    features1 = np.column_stack([int(audio_files[i].split('\\')[1].split('.')[0]), features1])
    if i == 0:
        Chroma_STFT_df = pd.DataFrame(features1, index = [str(i)])
    else:
        Chroma_STFT_df=Chroma_STFT_df.append(pd.DataFrame(features1,index = [str(i)])) 
    
colname = []
for i in range(len(Chroma_STFT_df.columns)):
    if i == 0:
        colname.append('itemid')
    else:
        colname.append('x' + str(i))

Chroma_STFT_df.columns = colname
Data_Chroma_STFT=pd.merge(data_df, Chroma_STFT_df, on='itemid', how='left')
Data_Chroma_STFT.to_pickle("Data_Chroma_STFT.pkl")