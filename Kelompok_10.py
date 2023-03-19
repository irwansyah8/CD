#import package
import cv2
import numpy as np
from matplotlib import pyplot as plt

#read file image
image = cv2.imread('burung.jpg')

##################################### Sharpening Image ##################################### 
#mendefinisikan ukuran kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
sharpening = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

#################################### Contrast Stretching #################################### 
#mengolah efek contrast gambar
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]

#mengembalikan nilai dengan jarak yang sama dalam interval tertentu
x = np.arange(256)

#mengembalikan interpolan linier satu dimensi ke fungsi dengan titik data diskrit yang diberikan (xp, fp), dievaluasi pada x
table = np.interp(x, xp, fp).astype('uint8')

#fungsi LUT mengisi larik keluaran dengan nilai dari tabel pencarian, Indeks entri diambil dari array input
contrast = cv2.LUT(sharpening, table)

#################################### Image Brightening ####################################
#mendefinisikan alpha % beta
alpha = 1.5 # Contrast control
beta = 10 # Brightness control

# memanggil convertScaleAbs function
brightening = cv2.convertScaleAbs(contrast, alpha=alpha, beta=beta)

#################################### Segmentasi Citra K-Means ####################################
#ubah image menjadi vektor 2D
image2D = brightening.reshape((-1,3))

#ubah uint8 menjadi tipe data float (persyaratan metode k-means dari OpenCV)
image2D = np.float32(image2D)

#criteria : penghentian iterasi atau perulangan
#parameter : (type, max_iter, epsilon)
#cv2.TERM_CRITERIA_EPS : menghentikan iterasi algoritma jika akurasi yang ditentukan, epsilon, tercapai
#cv2.TERM_CRITERIA_MAX_ITER : hentikan algoritma setelah jumlah iterasi yang ditentukan, max_iter
#cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : hentikan iterasi ketika salah satu kondisi di atas terpenuhi
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3

# attempts: Tandai untuk menentukan berapa kali algoritme dijalankan menggunakan pelabelan awal yang berbeda
attempts=10

#ret : jumlah kuadrat jarak dari setiap titik ke pusat masing-masing
#label : array label di mana setiap elemen ditandai '0', '1'
#center : array pusat cluster
#bestLabel : label kategori default > jika tidak ada (None)
#flag : digunakan untuk menentukan bagaimana pusat awal diambil
#KMEANS_RANDOM_CENTERS : memilih pusat awal acak dalam setiap upaya
#parameter kmeans :(img_src, jumlah kluster, bestLabel, criteria, attempts, flag)
ret,label,center=cv2.kmeans(image2D,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

#ubah lagi kedalam uint8
center = np.uint8(center)

#(labeling) mengakses label untuk membuat ulang gambar berkerumun
res = center[label.flatten()]
result_image = res.reshape((brightening.shape))

#################################### Matplotlib ####################################
#convert image to RGB
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
RGB_sharcon = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
RGB_bright = cv2.cvtColor(brightening, cv2.COLOR_BGR2RGB)
RGB_seg = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

#untuk melihat dan membandingkan nilai matriks
plt.subplot(221),plt.imshow(RGB_img),plt.title("Original Image") # Tata Letak Image baris 1 kolom 1 plot 1
plt.xticks([]), plt.yticks([]) # Akses nilai-nilai dari sumbu x dan y
plt.subplot(222),plt.imshow(RGB_sharcon),plt.title("Sharpening & Contrast") # Tata Letak Image baris 1 kolom 2 plot 2
plt.xticks([]), plt.yticks([]) # Akses nilai-nilai dari sumbu x dan y
plt.subplot(223),plt.imshow(RGB_bright),plt.title("Image Brightening") # Tata Letak Image baris 2 kolom 1 plot 3
plt.xticks([]), plt.yticks([]) # Akses nilai-nilai dari sumbu x dan y
plt.subplot(224),plt.imshow(RGB_seg),plt.title("Segmentasi K-Means") # Tata Letak Image baris 2 kolom 2 plot 4
plt.xticks([]), plt.yticks([]) # Akses nilai-nilai dari sumbu x dan y

#menampilkan plot
plt.show()

# Menunggu User menekan Sembarang Tombol untuk mengclose frame
cv2.waitKey(0)
