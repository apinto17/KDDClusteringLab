import pandas as pd
import sys



def main():
    filename = None
    thresh = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    elif(len(sys.argv) > 2):
        thresh = sys.argv[2]
        filename = sys.argv[1]
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)
    print(thresh)
    print(data)



if(__name__ == "__main__"):
    main()
