import torch
import torch.nn.functional as F
import vllm
import numpy as np

torch.manual_seed(0)

a = torch.tensor(
    [[0,1,2,3], [4,5,6,7]], dtype=torch.float32
)
a = torch.random.randn((3, 4096), dtype=torch.float32)
# a = torch.from_numpy(a, dtype=torch.float32)
print(F.softmax(a, dim=1))