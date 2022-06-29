images_dir = "images"
extensions = ("jpg", "jpeg", "png", "bmp")
batch_size = 128
embedding_size = 256
image_resize = 256
epsilon = 0.001
# Warning: Setting the following config to True will result in removing the detected duplicate images from disk.
# If you are not sure to do this, you can check and then delete the detected images manually one by one.
delete_files = True
