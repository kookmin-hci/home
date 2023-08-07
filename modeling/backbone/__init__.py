from modeling.backbone import resnet, xception, drn, mobilenet, efficientnet

def build_backbone(backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained)
    elif backbone == 'efficientnet':
        return efficientnet.EfficientNet_b6(stage_idxs=(11, 18, 38, 55), out_channels=(3, 64, 48, 80, 224, 640), model_name="efficientnet-b7", pretrained=pretrained)
    else:
        raise NotImplementedError
