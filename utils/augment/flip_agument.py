def main():
    import os
    import torchvision
    from pathlib import Path
    from PIL import Image


    path = "../../data/train_flip/"
    images_path = "../../data/train1"

    horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)

    if os.path.isdir(path) == False:
        os.mkdir(path)
        print('\nCreated:', path)        

    images = Path(images_path).glob("*.png")
    for image in images:
        save_path = path + str(image)[-13:]
        img=Image.open(image)
        img.save(save_path)
        save_path = path + str(image)[-13:-4] + "_" + "horizontal_flip.png"
        img=Image.open(image)
        img = horizontal_flip(img)
        img.save(save_path)
        save_path = path + str(image)[-13:-4] + "_" + "vertical_flip.png"
        img=Image.open(image)
        img = vertical_flip(img)
        img.save(save_path)
        save_path = path + str(image)[-13:-4] + "_" + "double_flip.png"
        img=Image.open(image)
        img = vertical_flip(img)
        img = horizontal_flip(img)
        img.save(save_path)
    print("done")
if __name__ == "__main__":
    main()    