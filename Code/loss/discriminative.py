from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F

class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=False, size_average=True):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        # _assert_no_grad(target)
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, prediction, seg_gt):

        loss_batch = torch.zeros(prediction.size()[0],dtype = torch.float32).cuda()
        l_var_batch= torch.zeros(prediction.size()[0],dtype = torch.float32).cuda()
        l_dist_batch = torch.zeros(prediction.size()[0],dtype = torch.float32).cuda()
        l_reg_batch = torch.zeros(prediction.size()[0],dtype = torch.float32).cuda()

        for i in range(prediction.size()[0]):
            loss_batch[i],l_var_batch[i],l_dist_batch[i],l_reg_batch[i] = self._discriminative_loss_single(prediction[i], seg_gt[i])

        loss_batch = torch.mean(loss_batch)
        l_var_batch = torch.mean(l_var_batch)
        l_dist_batch = torch.mean(l_dist_batch)
        l_reg_batch = torch.mean(l_reg_batch)

        return loss_batch, l_var_batch, l_dist_batch, l_reg_batch

    def _discriminative_loss_single(self, prediction, seg_gt):
        feature_dim = 3
        seg_gt = seg_gt.view(1,-1)
        prediction = prediction.view(feature_dim, -1)
        unique_labels,unique_ids = torch.unique(seg_gt,sorted=True,return_inverse=True)
        unique_labels = unique_labels.cuda().type(torch.cuda.LongTensor)
        unique_ids = unique_ids.cuda().type(torch.cuda.LongTensor)

        num_instances = unique_labels.size()[0]
        segment_mean = torch.zeros((feature_dim,num_instances),dtype=torch.float32).cuda()

        for i,lb in enumerate(unique_labels):
            mask = seg_gt.eq(lb).repeat(feature_dim,1)
            segment_embedding = torch.masked_select(prediction,mask).view(feature_dim,-1)
            segment_mean[:,i] = torch.mean(segment_embedding,dim=1)

        unique_ids = unique_ids.view(-1)
        mu_expand = segment_mean.index_select(1,unique_ids)
        distance = mu_expand-prediction
        distance = distance.norm(2,0,keepdim=True)
        distance = distance - self.delta_var
        distance = F.relu(distance)
        distance = distance**2

        l_var = torch.empty(num_instances,dtype=torch.float32).cuda()
        for i,lb in enumerate(unique_labels):
            mask = seg_gt.eq(lb)
            var_sum = torch.masked_select(distance,mask)
            l_var[i] = torch.mean(var_sum)

        l_var = torch.mean(l_var)

        seg_interleave = segment_mean.permute(1,0).repeat(num_instances,1)
        seg_band = segment_mean.permute(1,0).repeat(1,num_instances).view(-1,feature_dim)
        dist_diff = seg_interleave - seg_band
        mask = (1-torch.eye(num_instances,dtype = torch.int8)).view(-1,1).repeat(1,feature_dim).cuda()
        mask = mask == 1

        dist_diff = torch.masked_select(dist_diff,mask).view(-1,feature_dim)
        dist_norm = dist_diff.norm(2,1)
        dist_norm = 2*self.delta_dist - dist_norm
        dist_norm = F.relu(dist_norm)
        dist_norm = dist_norm**2
        l_dist = torch.mean(dist_norm)

        l_reg = torch.mean(torch.norm(segment_mean,2,0))

        l_var = self.alpha * l_var
        l_dist = self.beta * l_dist
        l_reg = self.gamma*l_reg
        loss = l_var + l_dist + l_reg

        return loss, l_var, l_dist, l_reg
