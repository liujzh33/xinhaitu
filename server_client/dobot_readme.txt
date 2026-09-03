已新建两份独立 WebSocket 代码，没有 import 或调用你列出的旧文件函数：

c:\Users\34149\Desktop\server-client\dobot_websocket_vlm_server.py
c:\Users\34149\Desktop\server-client\dobot_websocket_vlm_client.py
服务端是独立版：包含模型加载、WAN/VLM 配置、图片拼接、VLM 输入构造、T5 embedding 加载、inference_step() 调用和 WebSocket 收发。客户端也是独立版：用 WebSocket 发送 observation，图片直接以 numpy uint8 数组通过 MessagePack 二进制传输，不再走 HTTP JSON/Base64。

启动服务端示例：

python dobot_websocket_vlm_server.py \
  --model_config /path/to/config.yaml \
  --ckpt_dir /path/to/checkpoint \
  --wan_path /path/to/pretrained_models \
  --vlm_path /path/to/vlm \
  --device cuda:0 \
  --t5_embeddings_dir /path/to/t5_embeddings \
  --host 0.0.0.0 \
  --port 6790
客户端推理示例：

python dobot_websocket_vlm_client.py \
  --host 服务器IP \
  --port 6790 \
  --instruction "cook vegetable" \
  --top_image top.png \
  --left_wrist_image left.png \
  --right_wrist_image right.png \
  --state_csv "0.1,0.2,..."
我也做了检查：python -m py_compile 通过，两个新文件当前没有 IDE linter 报错。