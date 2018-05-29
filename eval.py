import torch
from generateDataBatch import somethingBatch
from model import Net
from utils import load_checkpoint
from config import *
import argparse

parser = argparse.ArgumentParser(description="parse argument for evaluation")
parser.add_argument("--batch-size", default=4, type=int, help="batch size")
parser.add_argument("--multigpu", default=False, type=bool, help="using multigpu or not")
parser.add_argument("--checkpoint", default="None", type=str, help="which checkpoint to use")
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


testSomething = Net(dataset_name="Something")
if args.checkpoint == "None":
    raise Exception("Must give checkpoint path")
else:
    load_checkpoint(testSomething,check_points_path+args.checkpoint)

if (torch.cuda.device_count() > 1) and args.multigpu:
    print("using mutigpu")
    testSomething = torch.nn.DataParallel(testSomething)


testSomething.to(device)
testSomething.eval()

something_validation_loader = torch.utils.data.DataLoader(
    somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingValidation"], datasets_path["SomethingData"]),
    batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)


top1 = 0
top5 = 0
total = 0
with torch.set_grad_enabled(False):
    for batch_number,batch_data in enumerate(something_validation_loader):
        print("----{}----".format(batch_number))
        input_data, target_data = batch_data["images"], batch_data["labels"]
        target_data = target_data.view(-1)
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        total += target_data.size()[0]
        low_out, space_out = testSomething(input_data)
        index = torch.sort(space_out, 1, descending=True)[1]
        for i in range(5):
            if i == 0:
                top1 += torch.sum(index[:,i] == target_data).item()
                top5 += torch.sum(index[:,i] == target_data).item()
            else:
                top5 += torch.sum(index[:,i] == target_data).item()
print(top1)
print(top5)
print(total)
print("-----top1-----{}".format(top1/total))
print("-----top5-----{}".format(top5/total))