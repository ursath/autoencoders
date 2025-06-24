import numpy as np
from fonts.utils import plot_font_single
from autoencoder.neural_network import NeuralNetwork

def generate_new_character_and_plot(neural_network: NeuralNetwork):
    valid_option_selected = False
    while not valid_option_selected:
        print("Select an option:")
        print("1. Generate a new character")
        print("2. Continue without generating a new character")
        option = input("Enter your choice (1 or 2): ").strip()

        if option == "1":
            valid_option_selected = True
            keep_generating = True
            while keep_generating:
                valid_input = False
                while not valid_input:
                    generate_latent_vector = input("Enter the latent vector to generate a new character (e.g: 0.2,-1.1): ")
                    try:
                        latent_vector = np.array([float(x.strip()) for x in generate_latent_vector.split(",")])
                        if latent_vector.shape[0] != 2:
                            print("❌ The latent vector must have exactly 2 values.")
                        else:
                            valid_input = True
                    except ValueError:
                        print("❌ Invalid input.")

                new_character = neural_network.decode_from_latent(latent_vector)
                plot_font_single(new_character, f"generated_char_{latent_vector[0]:.2f}_{latent_vector[1]:.2f}")

                while True:
                    generate_more = input("Do you want to generate another character? (yes/no): ").strip().lower()
                    if generate_more in {"yes", "no"}:
                        keep_generating = (generate_more == "yes")
                        break
                    else:
                        print("❌ Please type 'yes' or 'no'.")

        elif option == "2":
            valid_option_selected = True
        else:
            print("❌ Invalid option. Please enter 1 or 2.")
