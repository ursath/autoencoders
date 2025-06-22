import os

class Statistics:
    folder = "stats/data/"
    def __init__(self, filename: str):
        self.filename = filename
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.filepath = os.path.join(self.folder, filename) + ".csv"
        self.file = open(self.filepath, "w")

        self.file.write("epoch,network_error\n") 
    
    def write(self, text: str):
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
        print(f"Statistics saved to {self.filepath}")

