import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
from torchinfo import summary
import ipdb
from tqdm import tqdm
from torch_lr_finder import LRFinder

class Encoder(nn.Module):
    def __init__(self, embed_size=768, num_heads=3, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attention(x, x, x)[0]
        x = x + self.ff(self.ln2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, num_encoders=6, embed_size=768, img_size=(324, 324), patch_size=16, num_classes=10, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_tokens = (img_size[0]*img_size[1])//(patch_size**2)
        self.class_token = nn.Parameter(torch.randn((embed_size,)), requires_grad=True)
        self.patch_embedding = nn.Linear(in_channels*patch_size**2,embed_size)
        self.pos_embedding = nn.Parameter(torch.randn((num_tokens+1, embed_size)), requires_grad=True)
        self.encoders = nn.ModuleList([
            Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )
    def forward(self, x):
        ipdb.set_trace()
        batch_size, channel_size = x.shape[:2]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), -1, channel_size*self.patch_size*self.patch_size)
        x = self.patch_embedding(patches)
        class_token = self.class_token.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat([class_token, x], dim=1)  
        x = x + self.pos_embedding.unsqueeze(0)
        for encoder in self.encoders:
            x = encoder(x)
        x = x[:,0, :].squeeze()
        x = self.mlp_head(x)
        return x
    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer(in_channels=1, img_size=(28, 28), patch_size=7, embed_size=64, num_heads=4, num_encoders=3).to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()


    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    train_data = datasets.MNIST(root = './data/02/',
                                train=True,
                                download=True,
                                transform=data_transforms)
    test_data = datasets.MNIST(root = './data/02/',
                                train=False,
                                download=True,
                                transform=data_transforms)
    

    # train_dl = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    # optimizer = torch.optim.Adam(lr=5e-4, params=model.parameters())
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        cnt, correct_cnt = 0, 0
        for image, label in test_dl:
            image, label = image.to(device), label.to(device)
            pred = model(image).argmax(dim=1)
            cnt += label.shape[0]
            correct_cnt += (pred==label).sum().item()
        print("accuracy: ", correct_cnt / cnt)





    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # lr_finder.reset() # to reset the model and optimizer to their initial state

    # epochs = 10
    # for epoch in range(epochs):
    #     losses = []
    #     print(f"Epoch {epoch+1} / {epochs}", end=" ")
    #     for image, label in tqdm(train_dl):
    #         image, label = image.to(device), label.to(device)
    #         pred = model(image)
    #         loss = criterion(pred, label)
    #         loss.backward()
    #         losses.append(loss.item())
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     print(f"loss: {sum(losses) / len(losses)}", end=" ")
    #     with torch.no_grad():
    #         cnt, correct_cnt = 0, 0
    #         for image, label in test_dl:
    #             image, label = image.to(device), label.to(device)
    #             pred = model(image).argmax(dim=1)
    #             cnt += label.shape[0]
    #             correct_cnt += (pred==label).sum().item()
    #         print("accuracy: ", correct_cnt / cnt)
    # torch.save(model.state_dict(), './model.pt')