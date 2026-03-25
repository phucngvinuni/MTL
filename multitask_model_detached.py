import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PrototypeHead(nn.Module):
    def __init__(self, in_dim, num_prototypes, dim=256, temp=0.07):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temp = temp 
        self.projector = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False) 
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x):
        embeddings = self.projector(x)
        feat_norm = F.normalize(embeddings, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        logits = F.conv2d(feat_norm, proto_norm.unsqueeze(2).unsqueeze(3))
        return logits / self.temp

class AttributeAwareProtoSwin_Detached(nn.Module):
    """
    Phiên bản DETACHED: Cắt đứt dòng gradient giữa các Head.
    """
    def __init__(self, num_species, num_traits, num_seg_classes, model_name='swin_base_patch4_window7_224.ms_in22k'):
        super(AttributeAwareProtoSwin_Detached, self).__init__()

        # Backbone
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        backbone_channels = self.backbone.feature_info.channels()
        
        # FPN & Seg Head
        fpn_dim = 256
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in backbone_channels])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for c in backbone_channels])
        self.proto_head = PrototypeHead(in_dim=fpn_dim, num_prototypes=num_seg_classes, dim=fpn_dim)

        # Context Dim
        base_dim = backbone_channels[-1] + num_seg_classes
        
        # Trait Branch
        self.trait_embedder = nn.Sequential(
            nn.Linear(base_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.trait_classifier = nn.Linear(512, num_traits)

        # Species Branch
        species_in_dim = base_dim + 512
        self.species_head = nn.Sequential(
            nn.Linear(species_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_species)
        )

    def forward(self, x):
        # 1. Backbone & FPN
        features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2) for f in features]
        last_inner = self.lateral_convs[-1](features[-1])
        fpn_results = [self.fpn_convs[-1](last_inner)]
        for i in range(len(features) - 2, -1, -1):
            lat = self.lateral_convs[i](features[i])
            top_down = F.interpolate(last_inner, size=lat.shape[2:], mode='bilinear', align_corners=False)
            last_inner = lat + top_down
            fpn_results.append(self.fpn_convs[i](last_inner))
        
        # 2. Segmentation
        fine_grained_feature = fpn_results[-1] 
        seg_proto_logits = self.proto_head(fine_grained_feature)
        seg_logits_final = F.interpolate(seg_proto_logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 3. Feature Aggregation
        global_feat = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        proto_presence = F.adaptive_max_pool2d(seg_proto_logits, 1).flatten(1)
        
        # --- CẮT GRADIENT 1: Seg -> Trait ---
        # Nhánh Trait sẽ dùng kết quả của Seg để đoán, nhưng KHÔNG được update Seg
        proto_presence_detached = proto_presence.detach()
        
        base_context = torch.cat([global_feat, proto_presence_detached], dim=1)

        # 4. Trait Prediction
        trait_emb = self.trait_embedder(base_context)
        trait_logits = self.trait_classifier(trait_emb)
        
        # --- CẮT GRADIENT 2: Trait -> Species ---
        # Nhánh Species sẽ dùng kết quả Trait để đoán, nhưng KHÔNG được update Trait
        trait_emb_detached = trait_emb.detach()
        
        # Lưu ý: Chúng ta vẫn dùng base_context (có global_feat) để update backbone
        # Nhưng trait_emb_detached chặn dòng chảy về Trait Head
        species_input = torch.cat([base_context, trait_emb_detached], dim=1)
        species_logits = self.species_head(species_input)

        return {
            'species': species_logits,
            'traits': trait_logits,
            'segmentation': seg_logits_final
        }