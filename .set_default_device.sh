#!/bin/zsh
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"

USB_SINK=$(pactl list sinks | awk '
  BEGIN { sink_name = ""; description = "" }
  /^Sink #[0-9]+/ {
    if (sink_name != "" && tolower(description) ~ /usb.*audio device/) {
      exit  # 找到第一个后立即退出
    }
    sink_name = ""; description = ""
  }
  /Name: alsa_output/ { sink_name = $2 }
  { description = description " " tolower($0) }
  END { if (sink_name != "" && tolower(description) ~ /usb.*audio device/) print sink_name }
  ')

if [ -n "$USB_SINK" ]; then
    echo "set output: $USB_SINK"
    pactl set-default-sink "$USB_SINK"
    pactl set-sink-volume "$USB_SINK" 80%
    echo "set volume: 80%"
else
    echo "cannot find output device"
fi

USB_SOURCE=$(pactl list short sources | grep -v "\.monitor" | awk '
  BEGIN {
    xfm_dp = ""
    usb_audio = ""
  }
  {
    # 优先匹配XFM-DP设备
    if ($2 ~ /XFM-DP/ && xfm_dp == "") {
      xfm_dp = $2
    }
    # 其次匹配GeneralPlus USB设备
    else if ($2 ~ /usb.*audio device/i && usb_audio == "") {
      usb_audio = $2
    }
  }
  END {
    if (xfm_dp != "") {
      print xfm_dp
    } else if (usb_audio != "") {
      print usb_audio
    }
  }
')

if [ -n "$USB_SOURCE" ]; then
    echo "set input: $USB_SOURCE"
    pactl set-default-source "$USB_SOURCE"
else
    echo "cannot find input device"
fi
