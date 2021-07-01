import tkinter as tk
from PIL import ImageTk, Image
from detect_drowsiness import *

window = tk.Tk()
window.title("Drowsiness Detection")
window.configure(background='Black')


def Detect():
    Detection()

path = "Drowsy-Driving.jpg"

img = ImageTk.PhotoImage(Image.open(path))

panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

b2=tk.Button(panel,text="Start Detection", command=Detect)
b2.pack(side="left")
b2.place(x=250,y=150)

window.mainloop()
