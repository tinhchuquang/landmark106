import os
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./checkpoints/pfld/best.pth")
parser.add_argument('--onnx_model', default="./checkpoints/pfld/pfld.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./checkpoints/pfld/pfld_sim.onnx")
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dummy input')
args = parser.parse_args()

# print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
# print("PFLD bachbone:", pfld_backbone)

# print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(args.batch_size, 3, 112, 112))
input_names = ["input"]
output_names = ["output", "landmarks"]
torch.onnx.export(
    pfld_backbone,
    dummy_input,
    args.onnx_model,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
        'landmarks': {0: 'batch_size'},
    }
)
# print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(
    args.onnx_model,
    overwrite_input_shapes={"input": [args.batch_size, 3, 112, 112]}
)
assert check, "Simplified ONNX model could not be validated"
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")