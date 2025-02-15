from PIL import Image
import os
import matplotlib.pyplot as plt
import random

def load_triplets(save_dir="results"):
    triplets = []

    # Get list of indices from one folder (they are the same in all three)
    indices = sorted([int(f.split('.')[0]) for f in os.listdir(f"{save_dir}/original")])

    # Load images by matching indices
    for index in indices:
        original = Image.open(f"{save_dir}/original/{index}.jpg")
        damaged = Image.open(f"{save_dir}/damaged/{index}.jpg")
        reconstructed = Image.open(f"{save_dir}/reconstructed/{index}.jpg")

        # Store triplet
        triplets.append((original, damaged, reconstructed))

    return triplets


def plot_triplet(idx=None, save_dir="results"):
    # Load all triplets
    triplets = load_triplets(save_dir)

    # Choose a random index if none is provided
    if idx is None:
        idx = random.randint(0, len(triplets) - 1)

    # Extract the triplet
    original, damaged, reconstructed = triplets[idx]

    # Plot the images side by side
    titles = ['Original', 'Damaged', 'Reconstructed']
    images = [original, damaged, reconstructed]

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.suptitle(f'Triplet Index: {idx}')
    plt.show()


if __name__ == '__main__':
    path = ""
    plot_triplet(save_dir=path)