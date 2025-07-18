import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class BiRefNet:
    def __init__(self, model_path, target_size_h, target_size_w, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.image_size = (target_size_h, target_size_w)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        return model

    def extract(self, image_path, pil_image=None):
        # 1. 读取和预处理图片
        if pil_image is None:
            image = Image.open(image_path).convert('RGB')
            image = self.center_crop(image).resize(self.image_size)
        else:
            image = pil_image.convert('RGB')
        input_tensor = self.transform_image(image).unsqueeze(0).to(self.device)

        # 2. 推理
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask) / 255.0  # [0, 1]

        # 3. 原图与mask融合
        image_np = np.array(image)
        result_np = image_np * mask_np[:, :, np.newaxis]
        masked_image = Image.fromarray(np.uint8(result_np))

        return masked_image, mask

    @staticmethod
    def center_crop(img):
        # 简单中心裁剪为正方形
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        return img.crop((left, top, right, bottom))
