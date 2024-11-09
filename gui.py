import PySimpleGUI as sg
from graphcut import imageSegmentation  # Import the image segmentation function
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov
from dinics import dinic_algorithm

# Dictionary of available algorithms for the user to select
ALGO_MAPPING = {
    "Augmenting Path": "ap",
    "Push Relabel": "pr",
    "Boykov-Kolmogorov": "bk",
    "Dinic Algorithm": "dn"
}

def create_gui():
    # Define the window layout
    layout = [
        [sg.Text("Select Image File:")],
        [sg.InputText(key="image_path", size=(40, 1)), sg.FileBrowse(file_types=(("Image Files", "*"),))],
        
        [sg.Text("Select Algorithm:")],
        [sg.Combo(list(ALGO_MAPPING.keys()), default_value="Augmenting Path", key="algo", size=(20, 1))],
        
        [sg.Text("Resize Image (px):")],
        [sg.InputText("30", key="size", size=(10, 1))],
        
        [sg.Button("Run Segmentation"), sg.Button("Exit")],
        [sg.Text("", size=(40, 1), key="output", text_color="green")]
    ]
    
    # Create the window
    window = sg.Window("Image Segmentation GUI", layout)
    
    # Event loop
    while True:
        event, values = window.read()
        
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        
        if event == "Run Segmentation":
            # Retrieve values from the input fields
            image_path = values["image_path"]
            algo = values["algo"]
            algo = ALGO_MAPPING[algo]
        
            size = values["size"]
            
            print(algo)
            # Validation of input
            if not image_path:
                window["output"].update("Please select an image file.", text_color="red")
                continue
            
            if not size.isdigit():
                window["output"].update("Please enter a valid number for image size.", text_color="red")
                continue
            
            size = int(size)
            
            try:
                # Run the segmentation process
                imageSegmentation(image_path, (size, size), algo)
                window["output"].update(f"Segmentation complete! Image saved as {image_path}_cut.jpg", text_color="green")
            except Exception as e:
                window["output"].update(f"An error occurred: {str(e)}", text_color="red")
    
    window.close()

# Start the GUI when this file is run
if __name__ == "__main__":
    create_gui()
