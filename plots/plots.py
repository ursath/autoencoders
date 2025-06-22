import matplotlib.pyplot as plt
import pandas as pd

def plot_epoch_network_error(statistics, output_path):

    df = pd.read_csv(statistics.filepath)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['network_error'], marker='o', markersize=4)
    plt.title('Network Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Network Error')
    plt.grid(True)
    
    plt.savefig(f"{output_path}")
    plt.close()
    print(f"Plot saved to {output_path}")