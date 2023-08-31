import os
import torch
import PIL
#from pathlib import Path
#from PIL import UnidentifiedImageError
from PIL import Image, ImageFile
from utils.zero123_utils import predict_stage1_gradio, init_model
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave

class MultiView:
    def __init__(self):
        self._GPU_INDEX = 0
        self._HALF_PRECISION = True
        self.device = f"cuda:{self._GPU_INDEX}" if torch.cuda.is_available() else "cpu"

        self.models = init_model(self.device, 'zero123-xl.ckpt', half_precision=self._HALF_PRECISION)
        self.model_zero123 = self.models["turncam"]
        self.predictor = sam_init(self._GPU_INDEX)
    
    def preprocess(self, predictor, raw_im, lower_contrast=False):
        raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
        image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
        input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
        torch.cuda.empty_cache()
        return input_256

    def stage_run(self, model, device, exp_dir,
                   input_im, scale, ddim_steps):
        stage1_dir = os.path.join(exp_dir, "out")
        os.makedirs(stage1_dir, exist_ok=True)

        output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)

        return output_ims

    def multi_view(self, image_name):
        
        shape_id = image_name
        example_input_path = f"{shape_id}"
        example_dir = f"./exp/{shape_id}"

        os.makedirs(example_dir, exist_ok=True)
        input_raw = Image.open(example_input_path)
        input_256 = self.preprocess(self.predictor, input_raw)
        stage_imgs = self.stage_run(self.model_zero123, self.device, example_dir, input_256, scale=3, ddim_steps=75)

    def check_pngs(self, images_path):
        path = Path(images_path).rglob("*.png")
        for img_p in path:
            try:
                img = PIL.Image.open(img_p)
            except PIL.UnidentifiedImageError:
                print(img_p)

if __name__ == "__main__":
    multiview = MultiView()
    multiview.multi_view("image_name")
            

