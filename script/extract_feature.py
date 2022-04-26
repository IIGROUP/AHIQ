import torch

def get_resnet_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0],
            save_output.outputs[1],
            save_output.outputs[2]
        ),
        dim=1
    )
    return feat

def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0][:,1:,:],
            save_output.outputs[1][:,1:,:],
            save_output.outputs[2][:,1:,:],
            save_output.outputs[3][:,1:,:],
            save_output.outputs[4][:,1:,:],
        ),
        dim=2
    )
    return feat

def get_inception_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0],
            save_output.outputs[2],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[8],
            save_output.outputs[10]
        ),
        dim=1
    )
    return feat

def get_resnet152_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[3],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[7],
            save_output.outputs[8],
            save_output.outputs[10]
        ),
        dim=1
    )
    return feat