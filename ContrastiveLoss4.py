import torch

import torch.nn.functional as F

def ContrastiveLossx(features, temp=0.5):
    # features are assumed to be L2 normalized
    # features shape: [batch_size, 2, feature_dim]
    # where the second dimension represents the two augmented views


   # print('++++++features.shape ===', features.shape)  ##tensor type [(batchSize*2) xFeatures No.]

    # Reshape to [batch_size * 2, feature_dim]
    #features_flat = features.view(-1, features.shape[-1])
    features = F.normalize(features, dim=-1)  # Normalize features along the last dimension
    features_flat = features


    # Compute cosine similarity matrix
    similarity_matrix = F.cosine_similarity(features_flat.unsqueeze(1), features_flat.unsqueeze(0), dim=2)
   # print('++++++similarity_matrix ===', similarity_matrix.shape)  ##tensor type [(batchSize*2) x (batchSize*2)]

    # Create masks for positive and negative pairs
    labels = torch.arange(features.shape[0]//2).repeat(2).long().to(features.device)
    #print('++++++labels ===', labels.shape)  ##tensor type [0,1,2,...,(batchSize-1),0,1,2,...,(batchSize-1)]
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
   # print('++++++mask ===', mask.shape)  ##tensor type [(batchSize*2) x (batchSize*2)]
    mask_negatives = 1 - mask
    #print('++++++mask.shape ===', mask.shape)  ##tensor type [(batchSize*2) x (batchSize*2)]
    mask_positives = mask - torch.eye(mask.shape[0]).to(features.device) # remove self-similarity
    #print('++++++mask_positives ===', mask_positives)  ##tensor type [(batchSize*2) x (batchSize*2)]

    # Compute logit scaling
    logits = similarity_matrix / temp

    # Apply masks
    exp_logits = torch.exp(logits) * mask_negatives
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True)+.001)  # Adding a small value to avoid log(0)

    # Sum over positive pairs and average over batch
    loss = - (mask_positives * log_prob).sum(dim=1) / mask_positives.sum(dim=1)
    Lss=loss.mean()
    #print('++++++Lss ===', Lss)  ##tensor type [1] 
    return Lss, labels, logits