import torch.nn.functional as F
from torch import nn
import timm


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, config, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            config['model_name'], pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class SpotEncoder(nn.Module):
    """
    Encode spot to a fixed size vector
    """

    def __init__(
        self, config, trainable=True, 
    ):
        super().__init__()
        self.model = nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim'])
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
            self,
            config,
            embedding_dim=3467,
            projection_dim=256
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(config['dropout'])
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class HECLIPModel(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(config)
        self.spot_encoder = SpotEncoder(config)
        self.image_projection = ProjectionHead(config,embedding_dim=2048)
        self.spot_projection = ProjectionHead(config,embedding_dim=config['projection_dim'])
        self.temperature = config['temperature']

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        # spot_features = self.spot_encoder(batch["reduced_expression"])
        spot_features = batch["reduced_expression"]

        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the image centric Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        targets = F.softmax(
            (images_similarity) / self.temperature, dim=-1
        )
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = images_loss


        return loss.mean()




def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


