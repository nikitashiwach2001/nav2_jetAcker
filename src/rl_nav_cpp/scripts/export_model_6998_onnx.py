#!/usr/bin/env python3
"""Export model_6998.pt (actor-only) to ONNX.

Architecture reconstructed by inspecting carrot_l6.onnx's graph (exported earlier
from the same CnnGruActorCritic family, confirmed identical actor-path weight
shapes to model_6998.pt / model_189933.pt): a lidar CNN branch with circular
(wrap-around) padding + stride-2 convs, an ELU proprio branch, concatenated and
fed through a single-layer GRU, then an ELU MLP head producing a raw
(unbounded) 2-dim action. This mirrors carrot_l6.onnx node-for-node so the new
export goes through the same torch.onnx GRU gate-reordering path.

obs [1, 1098] = lidar_raw [1080] (6 frames x 180 rays) ++ proprio [18]
h_in [1, 1, 256] GRU hidden state
-> action [1, 2] (raw, unbounded), h_out [1, 1, 256]
"""
import torch
import torch.nn as nn

N_RAYS = 180
N_FRAMES = 6
LIDAR_LEN = N_FRAMES * N_RAYS  # 1080
PROPRIO_LEN = 18
OBS_DIM = LIDAR_LEN + PROPRIO_LEN  # 1098
HIDDEN = 256


class CircularConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=0)

    def forward(self, x):
        left = x[..., -self.pad:]
        right = x[..., :self.pad]
        x = torch.cat([left, x, right], dim=-1)
        return self.conv(x)


class LidarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CircularConv1d(N_FRAMES, 32, 5, stride=2)
        self.conv2 = CircularConv1d(32, 64, 3, stride=2)
        self.proj = nn.Linear(2880, 256)

    def forward(self, x):
        x = x.view(-1, N_FRAMES, N_RAYS)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.proj(x))
        return x


class ActorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar = LidarEncoder()
        self.proprio = nn.Sequential(nn.Linear(PROPRIO_LEN, 64), nn.ELU())
        self.rnn = nn.GRU(input_size=320, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 2),
        )

    def forward(self, obs, h_in):
        lidar_feat = self.lidar(obs[:, :LIDAR_LEN])
        proprio_feat = self.proprio(obs[:, LIDAR_LEN:])
        feat = torch.cat([lidar_feat, proprio_feat], dim=-1).unsqueeze(1)
        out, h_out = self.rnn(feat, h_in)
        action = self.actor(out.squeeze(1))
        return action, h_out


def build_and_load(pt_path):
    ckpt = torch.load(pt_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    model = ActorPolicy()
    mapped = {
        'lidar.conv1.conv.weight': sd['conv_a.lidar.conv1.weight'],
        'lidar.conv1.conv.bias': sd['conv_a.lidar.conv1.bias'],
        'lidar.conv2.conv.weight': sd['conv_a.lidar.conv2.weight'],
        'lidar.conv2.conv.bias': sd['conv_a.lidar.conv2.bias'],
        'lidar.proj.weight': sd['conv_a.lidar.proj.weight'],
        'lidar.proj.bias': sd['conv_a.lidar.proj.bias'],
        'proprio.0.weight': sd['conv_a.proprio.0.weight'],
        'proprio.0.bias': sd['conv_a.proprio.0.bias'],
        'rnn.weight_ih_l0': sd['memory_a.rnn.weight_ih_l0'],
        'rnn.weight_hh_l0': sd['memory_a.rnn.weight_hh_l0'],
        'rnn.bias_ih_l0': sd['memory_a.rnn.bias_ih_l0'],
        'rnn.bias_hh_l0': sd['memory_a.rnn.bias_hh_l0'],
        'actor.0.weight': sd['actor.0.weight'],
        'actor.0.bias': sd['actor.0.bias'],
        'actor.2.weight': sd['actor.2.weight'],
        'actor.2.bias': sd['actor.2.bias'],
        'actor.4.weight': sd['actor.4.weight'],
        'actor.4.bias': sd['actor.4.bias'],
    }
    missing, unexpected = model.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    model.eval()
    return model


def main():
    pt_path = '/home/ubuntu/ros2_ws/src/rl_nav_cpp/model_6998.pt'
    out_path = '/home/ubuntu/ros2_ws/src/rl_nav_cpp/policies/policy_v4_model6998.onnx'

    model = build_and_load(pt_path)

    obs = torch.zeros(1, OBS_DIM)
    h_in = torch.zeros(1, 1, HIDDEN)

    torch.onnx.export(
        model, (obs, h_in), out_path,
        input_names=['obs', 'h_in'],
        output_names=['action', 'h_out'],
        opset_version=17,
        dynamic_axes=None,
    )
    print('exported ->', out_path)


if __name__ == '__main__':
    main()
