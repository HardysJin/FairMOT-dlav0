import torch

from utils import _init_paths

from models.pose_dla_dcn import get_pose_net

print(torch.__version__)
name='mot'

heads = {'hm': 1,
            'wh': 4,
            'reg': 2,
            'id': 128,}
model = get_pose_net(34, heads, 256)
model.load_state_dict(torch.load("../pretrained/final_mot.pth"))
model.cuda()
dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')


output_names = [ 'hm' , 'wh', 'reg', 'id']
torch.onnx.export(model, dummy_input, "../pretrained/final_mot.onnx", verbose=True, output_names=output_names)
print(model)