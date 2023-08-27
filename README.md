# 2D Semantic Segmentation with PyTorch U-Net

1. Download the dataset
2. Clone this [fork](https://github.com/cppxor2arr/Pytorch-UNet) of Pytorch-UNet

   ```sh
   git clone https://github.com/cppxor2arr/Pytorch-UNet
   ```
3. Remove similar images using image hashing

   ```sh
   python -m pip install imagehash numpy Pillow tqdm
   python remove_similar_images.py
   ```
4. Create image mask files from the coordinates in the label files

   ```sh
   python preprocess.py
   ```
5. Archive the code and data to upload to Google Colab

   ```sh
   cp -R '2D Semantic Segmentation/training/images' data/imgs
   cp -R data Pytorch-UNet
   tar -cf upload.tar Pytorch-UNet
   ```
6. Upload `upload.tar` to Google Drive
7. Run `self_driving_ai_contest.ipynb` in Google Colab
8. Download the trained model in the `checkpoints` directory
9. (Optional) Test the model
   ```sh
   python -m pip install -r Pytorch-UNet/requirements.txt
   python -m pip install torch torchvision
   python Pytorch-UNet/predict.py --model <downloaded_model> --classes 27 -i <input_image> -o output.jpg
   ```
