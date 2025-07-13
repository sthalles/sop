import torch.nn as nn
import torch
import utils

# class CrossEntropyWithLogits(nn.Module):
#     def __init__(self):
#         super().__init__()

#     # def cross_entropy(self, p, q):
#     #     assert p.shape == q.shape
#     #     assert p.requires_grad == True
#     #     assert q.requires_grad == False

#     #     p = torch.log_softmax(p, dim=-1)
#     #     q = torch.softmax(q, dim=-1)

#     #     loss = torch.sum(-q * p, dim=-1).mean()
#     #     return loss

#     def cross_entropy(self, p, q):
#         assert p.shape == q.shape
#         assert p.requires_grad == True
#         assert q.requires_grad == False

#         p = torch.softmax(p, dim=-1)
#         q = torch.softmax(q, dim=-1)

#         EPS = torch.finfo(p.dtype).eps
#         loss = torch.einsum("nc,nc->n", [p, q])
#         loss = torch.clamp(loss, EPS, 1.0 - EPS)
#         loss = -torch.log(loss).mean()
#         return loss

#     def forward(self, student_output, teacher_output):
#         # EPS = torch.finfo(student_output[0].dtype).eps
#         consistency = 0
#         count = 0
#         for i in range(len(student_output)):
#             for j in range(len(teacher_output)):
#                 if i == j:
#                     continue
#                 consistency += self.cross_entropy(
#                     student_output[i], teacher_output[j])
#                 count += 1

#         consistency /= count
#         return consistency


class PatchCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy(self, p, q, mask):
        assert p.shape == q.shape
        assert p.requires_grad == True
        assert q.requires_grad == False
        assert len(p.shape) == 3
        assert len(q.shape) == 3

        loss = torch.sum(torch.log(p**(-q)), dim=-1)
        loss = torch.sum(loss * mask.float(), dim=-1) / \
            mask.sum(dim=-1).clamp(min=1.0)
        return loss.mean()

    def forward(self, student_output, teacher_output, student_mask):
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    consistency += self.cross_entropy(
                        student_output[i], teacher_output[j], student_mask[i].flatten(-2, -1))
                    count += 1

        consistency /= count
        return consistency


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy(self, p, q):
        assert p.shape == q.shape
        assert p.requires_grad == True
        assert q.requires_grad == False
        assert len(p.shape) == 2
        assert len(q.shape) == 2
        # assert torch.all(p.sum(-1) == 1), f"{p.sum(-1)}"
        loss = torch.mean(torch.sum(torch.log(p**(-q)), dim=-1))
        return loss


    def forward(self, student_output, teacher_output):
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.cross_entropy(student_output[i], teacher_output[j])
                count += 1

        consistency /= count
        return consistency


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
