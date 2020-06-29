import PIL.Image

def main():
    infile = "/home/lacker/Desktop/AJ.jpg"
    image = PIL.Image.open(infile)
    print(image)

if __name__ == "__main__":
    main()
