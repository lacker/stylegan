import PIL.Image

def main():
    infile = "/home/lacker/Desktop/AJ.jpg"
    image = PIL.Image.open(infile)
    resized = image.resize((512, 512), PIL.Image.ANTIALIAS)
    resized.save("results/aj.png")

if __name__ == "__main__":
    main()
