from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
import h5py

h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')
X, y, attr = get_train_data(h5_file, ['ADC'])
data = np.array(X[0])
img = plt.imshow(data)

points = []

cursor = mplcursors.cursor(img, hover=False)
@cursor.connect("add")
def cursor_clicked(sel):
    # sel.annotation.set_visible(False)
    sel.annotation.set_text(
        f'Clicked on\nx: {sel.target[0]:.2f} y: {sel.target[1]:.2f}\nindex: {sel.target.index}')
    points.append(sel.target.index)
    print("Current list of points:", points)

plt.show()
print("Selected points:", points)
