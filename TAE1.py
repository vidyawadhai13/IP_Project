import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import webbrowser
import math

# ---------- Select Image ----------
root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
root.destroy()

img = cv2.imread(path)
if img is None:
    print("Image not found!")
    exit()

# ---------- QR Detection ----------
qr = cv2.QRCodeDetector()
data, points, _ = qr.detectAndDecode(img)

print("\n=========================================")
if points is not None:
    print("QR CODE DETECTED: YES")
    if data:
        print("METHOD USED: Original Image")
        print("DECODED INFORMATION:", data)
        if data.startswith(("http://", "https://")):
            print("QR TYPE: WEBPAGE / URL")
            print("Opening website...")
            webbrowser.open(data)
    else:
        print("DECODED INFORMATION: NONE")
        print("QR STATUS: DAMAGED / UNREADABLE")
else:
    print("QR CODE DETECTED: NO")
print("=========================================")

# ---------- Image Processing ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
enhanced = cv2.equalizeHist(gaussian)

sharp = cv2.filter2D(enhanced, -1, np.array([
    [0,-1,0], [-1,5,-1], [0,-1,0]]))

sx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

otsu_value, otsu = cv2.threshold(
    enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

canny = cv2.Canny(enhanced, 100, 200)

# ---------- Statistics ----------
hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
p = hist / hist.sum()
entropy = -sum(x * math.log2(x) for x in p if x > 0)
edge_density = np.count_nonzero(canny) / canny.size * 100

print("\n------------- IMAGE INFORMATION -------------")
print("Image Size          :", img.shape[1], "x", img.shape[0])
print("Mean                :", round(np.mean(gray), 2))
print("Standard Deviation  :", round(np.std(gray), 2))
print("Contrast            :", int(gray.max() - gray.min()))
print("Entropy             :", round(entropy, 2))
print("Otsu Threshold      :", round(otsu_value, 2))
print("Gaussian Kernel     : 5 x 5")
print("Sharpening Kernel   : 3 x 3")
print("Sobel Kernel        : 3 x 3")
print("Edge Density        :", round(edge_density, 2), "%")

# ---------- Binary Matrix ----------
matrix = (cv2.resize(otsu, (25,25)) > 127).astype(int)

print("\n------------- 25 x 25 BINARY MATRIX -------------")
for row in matrix:
    print(" ".join(map(str, row)))

# ---------- QR Boundary ----------
result = img.copy()
if points is not None:
    pts = np.int32(points.reshape(-1,2))
    cv2.polylines(result, [pts], True, (0,0,255), 3)

# ---------- 8 Main Results ----------
images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, gaussian, enhanced,
    sharp, sobel, otsu, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)]

titles = [
    "1. Original Image", "2. Grayscale",
    "3. Gaussian Filter (5x5)", "4. Contrast Enhanced",
    "5. Mask 1 - Sharpening", "6. Mask 2 - Sobel Edge",
    "7. Otsu Threshold", "8. QR Detected + Boundary"]

plt.figure(figsize=(14,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

# ---------- Histogram ----------
plt.figure(figsize=(7,4))
plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()