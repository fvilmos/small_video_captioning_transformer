####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import cv2
import collections
from torchvision import transforms
from PIL import Image

from utils.video_captioning_transformer import VideoCaptioner, generate_caption
from utils.vocabulary import Vocabulary
from utils.video_encoder import CNNEncoder, ViTEncoder, MobileNetV2Encoder

def main():
    num_frames = 8
    img_size = 224
    max_len = 100
    cam_index = 0
    caption = [""]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './vct_model.pth'

    voc = Vocabulary()
    # load your vocabulary
    voc.import_vocabulary('voc.json') 
    
    vision_encoder_class = MobileNetV2Encoder

    #create the model
    model = VideoCaptioner(vocab_size=len(voc),
                           dim=256,
                           num_heads=4,
                           num_layers=2,
                           vis_out_dimension=1280,
                           vis_hxw_out=49,
                           num_frames=num_frames,
                           max_len=max_len,
                           VisionEncoder=vision_encoder_class).to(device)

    # load the model
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # get camera stream
    cap = cv2.VideoCapture(cam_index)
    buffer = collections.deque(maxlen=num_frames)
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        # convert image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # add frame to buffer
        buffer.append(pil_img)

        if len(buffer) == num_frames:

            # preprocess frames
            frames_tensor = torch.stack([transform(f) for f in buffer]).to(device)

            # generate caption
            with torch.no_grad():
                caption = generate_caption(model,frames_tensor, voc, max_len=max_len)
                print("Caption:", " ".join(caption[0]))


        # frame with caption
        cv2.putText(frame, " ".join(caption[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Captioning', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
