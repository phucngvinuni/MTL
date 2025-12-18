import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class FPNHead(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPNHead, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, features):
        laterals = [
            conv(features[i]) for i, conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)

        outputs = [
            self.output_convs[i](laterals[i]) for i in range(len(laterals))
        ]

        h, w = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(h, w), mode='bilinear', align_corners=False)

        return torch.cat(outputs, dim=1)


class MultiTaskSwinTransformer(nn.Module):
    def __init__(self, num_species, num_traits, num_seg_classes, model_name='swin_base_patch4_window7_224.ms_in22k'):
        super(MultiTaskSwinTransformer, self).__init__()
        
        # timm Swin model trả về định dạng channel-last (B, H, W, C)
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        
        backbone_out_channels = self.backbone.feature_info.channels()
        feature_dim = backbone_out_channels[-1]

        self.species_head = nn.Linear(feature_dim, num_species)
        self.trait_head = nn.Linear(feature_dim, num_traits)

        fpn_channels = 256
        self.seg_fpn_head = FPNHead(in_channels_list=backbone_out_channels, out_channels=fpn_channels)
        self.seg_classifier = nn.Conv2d(len(backbone_out_channels) * fpn_channels, num_seg_classes, 1)


    def forward(self, x):
        # features là một list các tensor (B, H, W, C)
        features = self.backbone(x)
        
        # --- LOGIC ĐÚNG: Chuyển đổi tất cả về (B, C, H, W) ---
        features_2d = [feat.permute(0, 3, 1, 2) for feat in features]

        # --- Xử lý cho Classification và Identification ---
        # Lấy feature map cuối cùng đã được chuyển đổi
        last_feature_map = features_2d[-1] # Shape (B, 1024, 7, 7)
        
        # Áp dụng Global Average Pooling
        pooled_features = F.adaptive_avg_pool2d(last_feature_map, (1, 1)).flatten(1) # Shape (B, 1024)
            
        species_logits = self.species_head(pooled_features)
        trait_logits = self.trait_head(pooled_features)

        # --- Xử lý cho Segmentation ---
        # `features_2d` đã ở đúng định dạng cho FPNHead
        fpn_features = self.seg_fpn_head(features_2d)
        
        fpn_features_upsampled = F.interpolate(fpn_features, scale_factor=4, mode='bilinear', align_corners=False)
        
        seg_output = self.seg_classifier(fpn_features_upsampled)
        
        seg_logits = F.interpolate(seg_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return {
            'species': species_logits,
            'traits': trait_logits,
            'segmentation': seg_logits
        }