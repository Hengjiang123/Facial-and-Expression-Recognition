from PIL import Image
import torchvision.transforms as transforms
import os
import glob


def augment_image(image, augmentations, save_dir, base_name, num_images=10):
    # augmentation and saving for each picture
    for i in range(num_images):
        augmented_image = augmentations(image)
        augmented_image.save(f'{save_dir}/{base_name}_augmented_{i}.jpg')


def augment_and_save_images(dataset_dir, save_dir, num_images=10):
    # the data augmentation transform
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])

    # make sure the addr is ture
    os.makedirs(save_dir, exist_ok=True)

    # go through all the pictures in addr
    for image_path in glob.glob(f'{dataset_dir}/*'):  # Adjust wildcards to match all image files
        image = Image.open(image_path)
        base_name = os.path.basename(image_path).split('.')[0]  # get the file name

        # call the augmentation function
        augment_image(image, augmentations, save_dir, base_name, num_images)

    print("Data augmentation completed successfully.")


# addr for augmented data and save addr
dataset_dir = 'C:/Users/LIU/Desktop/archive1/test/disgust'
save_dir = 'C:/Users/LIU/Desktop/disgust'

augment_and_save_images(dataset_dir, save_dir)
