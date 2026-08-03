#!/usr/bin/env python3
"""Export model_10496.pt (actor-only) to ONNX.

Architecture confirmed identical to v1's policy_full_new.onnx: verified by (1) comparing
model_10496.pt's state_dict shapes directly against policy_full_new.onnx's initializers,
and (2) inspecting policy_full_new.onnx's own ONNX graph node-for-node (conv strides/
padding, activation placement, GRU wiring) so this export mirrors it exactly rather than
guessing. Only the trained weights differ from v1's original model.

obs [1, 1102] = lidar_raw [1080] (6 frames x 180 rays) ++ proprio_raw [22] (no dense layer)
h_in [1, 1, 256] GRU hidden state
-> actions [1, 2] (raw, unbounded), h_out [1, 1, 256]
"""
import torch
import torch.nn as nn

N_RAYS = 180
N_FRAMES = 6
LIDAR_LEN = N_FRAMES * N_RAYS  # 1080
PROPRIO_LEN = 22
OBS_DIM = LIDAR_LEN + PROPRIO_LEN  # 1102
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
    # 3 conv layers (v1's architecture), strides 1/2/2 -- NOT v3/v4's 2-conv/stride-2/2
    def __init__(self):
        super().__init__()
        self.conv1 = CircularConv1d(N_FRAMES, 32, 5, stride=1)
        self.conv2 = CircularConv1d(32, 32, 5, stride=2)
        self.conv3 = CircularConv1d(32, 64, 5, stride=2)
        self.proj = nn.Linear(2880, 128)

    def forward(self, x):
        x = x.view(-1, N_FRAMES, N_RAYS)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.proj(x))
        return x


class ActorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar = LidarEncoder()
        # v1 has NO proprio dense layer -- raw 22 values concat straight into the GRU input
        self.rnn = nn.GRU(input_size=128 + PROPRIO_LEN, hidden_size=HIDDEN,
                           num_layers=1, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 2),
        )

    def forward(self, obs, h_in):
        lidar_feat = self.lidar(obs[:, :LIDAR_LEN])
        proprio_raw = obs[:, LIDAR_LEN:]
        feat = torch.cat([lidar_feat, proprio_raw], dim=-1).unsqueeze(1)
        out, h_out = self.rnn(feat, h_in)
        action = self.actor(out.squeeze(1))
        return action, h_out


def build_and_load(pt_path):
    ckpt = torch.load(pt_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    model = ActorPolicy()
    mapped = {
        'lidar.conv1.conv.weight': sd['conv_a.lidar_encoder.conv1.weight'],
        'lidar.conv1.conv.bias': sd['conv_a.lidar_encoder.conv1.bias'],
        'lidar.conv2.conv.weight': sd['conv_a.lidar_encoder.conv2.weight'],
        'lidar.conv2.conv.bias': sd['conv_a.lidar_encoder.conv2.bias'],
        'lidar.conv3.conv.weight': sd['conv_a.lidar_encoder.conv3.weight'],
        'lidar.conv3.conv.bias': sd['conv_a.lidar_encoder.conv3.bias'],
        'lidar.proj.weight': sd['conv_a.lidar_encoder.proj.weight'],
        'lidar.proj.bias': sd['conv_a.lidar_encoder.proj.bias'],
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
    pt_path = '/home/ubuntu/ros2_ws/src/rl_nav_cpp/model_10496.pt'
    out_path = '/home/ubuntu/ros2_ws/src/rl_nav_cpp/policies/policy_v5_model10496.onnx'

    model = build_and_load(pt_path)

    obs = torch.zeros(1, OBS_DIM)
    h_in = torch.zeros(1, 1, HIDDEN)

    torch.onnx.export(
        model, (obs, h_in), out_path,
        input_names=['obs', 'h_in'],
        output_names=['actions', 'h_out'],
        opset_version=17,
        dynamic_axes=None,
    )
    print('exported ->', out_path)


if __name__ == '__main__':
    main()
