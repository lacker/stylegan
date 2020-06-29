import PIL.Image

def main():
    infile = "/home/lacker/AJ.jpg"
    image = PIL.Image.open(infile)
    resized = image.resize((64, 64), PIL.Image.ANTIALIAS)
    resized.save("results/aj.png")

if __name__ == "__main__":
    main()
