import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Model names: RealESRGAN_x4plus | RealESRNet_x4plus |
# 						 RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus |
# 						 realesr-animevideov3 | realesr-general-x4v3


def main(
	inputImage: str = "",
	modelName: str = "RealESRGAN_x4plus",
	outPutFolder: str = "results",
	denoiseStrength: float = 0.5,  # 0 weak, 1 strong. Only used for the realesr-general-x4v3 model
	outScale: float = 4,  # final upsampling of the image
	modelPath: str = None,  # shouldn't need to specify
	suffix: str = "out",  # suffix of the restored image
	tile: int = 0,  # tile size, 0 for no tile during testing
	tilePad: int = 10,  # tile padding
	prePad: int = 0,  # pre padding size at each border
	faceEnhance: bool = False,  # Use GFPGAN to enhance face
	fp32: bool = False,  # Use fp32 precision during inference. Default: fp16 (half precision).
	alphaUpsampler: str = "realesrgan",  # The upsampler for the alpha channels. Options: realesrgan | bicubic'
	ext: str = "auto",  # Image extension. Options: auto | jpg | png, auto means using the same extension as inputs
	gpuID: int = None  # gpu device to use (default=None) can be 0,1,2 for multi-gpu
):
	modelName = modelName.split('.')[0]
	if modelName == 'RealESRGAN_x4plus':  # x4 RRDBNet model
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
		netscale = 4
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
	elif modelName == 'RealESRNet_x4plus':  # x4 RRDBNet model
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
		netscale = 4
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
	elif modelName == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
		netscale = 4
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
	elif modelName == 'RealESRGAN_x2plus':  # x2 RRDBNet model
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
		netscale = 2
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
	elif modelName == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
		model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
		netscale = 4
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
	elif modelName == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
		model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
		netscale = 4
		file_url = [
			'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
			'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
		]
	# determine model paths
	if modelPath is not None:
		model_path = modelPath
	else:
		model_path = os.path.join('weights', modelName + '.pth')
		if not os.path.isfile(model_path):
			ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
			for url in file_url:
				# model_path will be updated
				model_path = load_file_from_url(
					url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

	# use dni to control the denoise strength
	dni_weight = None
	if modelName == 'realesr-general-x4v3' and denoiseStrength != 1:
		wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
		model_path = [model_path, wdn_model_path]
		dni_weight = [denoiseStrength, 1 - denoiseStrength]

	# restorer
	upsampler = RealESRGANer(
		scale=netscale,
		model_path=model_path,
		dni_weight=dni_weight,
		model=model,
		tile=tile,
		tilePad=tilePad,
		pre_pad=prePad,
		half=not fp32,
		gpu_id=gpuID
	)
	if faceEnhance:  # Use GFPGAN for face enhancement
		from gfpgan import GFPGANer
		face_enhancer = GFPGANer(
			model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
			upscale=outScale,
			arch='clean',
			channel_multiplier=2,
			bg_upsampler=upsampler)

	os.makedirs(outPutFolder, exist_ok=True)

	if os.path.isfile(inputImage):
		paths = [inputImage]
	else:
		paths = sorted(glob.glob(os.path.join(inputImage, '*')))

		for idx, path in enumerate(paths):
			imgname, extension = os.path.splitext(os.path.basename(path))
			print('Testing', idx, imgname)

			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			if len(img.shape) == 3 and img.shape[2] == 4:
				img_mode = 'RGBA'
			else:
				img_mode = None

			try:
				if faceEnhance:
					_, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
				else:
					output, _ = upsampler.enhance(img, outscale=outScale)
			except RuntimeError as error:
				print('Error', error)
				print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
			else:
				if ext == 'auto':
					extension = extension[1:]
				else:
					extension = ext
				if img_mode == 'RGBA':  # RGBA images should be saved in png format
					extension = 'png'
				if suffix == '':
					save_path = os.path.join(outPutFolder, f'{imgname}.{extension}')
				else:
					save_path = os.path.join(outPutFolder, f'{imgname}_{suffix}.{extension}')
				cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
