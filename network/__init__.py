import torchvision.models as models
import torch.nn as nn

def get_model(args):
    # BUILD MODEL
    if args.model == 'resnet18_backbone':
        print('Building resnet18 without BN layers')
        backbone = models.resnet18(args.pretrained)
        backbone = ResNet18_No_BN(backbone)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential()
        model = ResNetBackbone(backbone=backbone, feature_dim=in_features, num_classes=args.num_classes)

    elif args.model == 'resnet18':
        print('Building resnet18 without BN layers')
        model = models.resnet18(args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        model = ResNet18_No_BN(model)

    else:
        exit('Error: unrecognized model')
    return model

def ResNet18_No_BN(model):
    model.bn1 = nn.Sequential()
    model.layer1[0].bn1 = nn.Sequential()
    model.layer1[0].bn2 = nn.Sequential()

    model.layer1[1].bn1 = nn.Sequential()
    model.layer1[1].bn2 = nn.Sequential()

    model.layer2[0].bn1 = nn.Sequential()
    model.layer2[0].bn2 = nn.Sequential()
    model.layer2[0].downsample[1] = nn.Sequential()

    model.layer2[1].bn1 = nn.Sequential()
    model.layer2[1].bn2 = nn.Sequential()

    model.layer3[0].bn1 = nn.Sequential()
    model.layer3[0].bn2 = nn.Sequential()
    model.layer3[0].downsample[1] = nn.Sequential()

    model.layer3[1].bn1 = nn.Sequential()
    model.layer3[1].bn2 = nn.Sequential()

    model.layer4[0].bn1 = nn.Sequential()
    model.layer4[0].bn2 = nn.Sequential()
    model.layer4[0].downsample[1] = nn.Sequential()

    model.layer4[1].bn1 = nn.Sequential()
    model.layer4[1].bn2 = nn.Sequential()

    return model


class ResNetBackbone(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(ResNetBackbone, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        out = self.linear(feature)

        return out

    def update_encoder(self, backbone):
        self.backbone = backbone

if __name__ == '__main__':
    backbone = models.resnet18(False)
    backbone = ResNet18_No_BN(backbone)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential()
    model = ResNetBackbone(backbone=backbone, feature_dim=in_features, num_classes=10)
    for i in model.parameters():
        print(i)


