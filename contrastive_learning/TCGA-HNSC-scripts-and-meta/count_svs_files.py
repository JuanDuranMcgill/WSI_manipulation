import os

def count_svs_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".svs"):
                count += 1
                print(count)
    return count

# Input path
path = input("Enter the directory path: ")
number_of_svs = count_svs_images(path)
print(f"The number of .svs files in the directory is: {number_of_svs}")
