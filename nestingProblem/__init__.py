import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from glob import glob
from PIL import Image
from numpy import asarray
from datetime import datetime

#defaut parameters
page_r    = [297, 210]
threshold = 127               # binary image
kernel    = np.ones((9,9), np.uint8)
distance  = 10

# Create the page
def paperA4(pag_size = page_r):
  pag_h = pag_size[0]
  pag_w = pag_size[1]
  return (np.zeros(shape=(pag_h,pag_w), dtype=np.uint8))

# Inverse binary image (region of interest - white pixels)
def figureBinary(fig, threshold):
  fig = cv2.threshold(fig, threshold, 255, cv2.THRESH_BINARY_INV)
  return fig[1]

# Read and binarize image
def call_Image(path):
  img = cv2.imread(path)                          # Read image
  img = figureBinary(img, threshold)              # Binarize image
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 3D to 1D
  return img

# Function to calculate white pixel density
def densi_image(img):
  white_p = cv2.countNonZero(img) 
  area = img.shape[0]*img.shape[1]
  dens = white_p/area
  return dens
  
# Function to calculate the distance
def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

# Function that creates the background image (in case if it increase height or width)
def new_dst(img1, a2, l2, a):
  caso = 0
  dst = []
  if(a[0] - (2/3)*l2> 0):            # Check if it is possible to fit 1/3 of image 2 to the left of image 1
    sum_w = 0
    for ii in range(int((2/3)*l2)):
      sum_w = sum_w + img1[a[1], a[0]-ii]
    if sum_w == 0:                              # Check if the left pixels are empty
      if l2 > a[0]:                             # Check if it needs to increase the width of the image or not
        if a[1] + a2 > img1.shape[0]:
          dst = Image.new('L', (img1.shape[1] + l2-a[0], a[1]+a2), color=0)
        else:
          dst = Image.new('L', (img1.shape[1] + l2-a[0], img1.shape[0]), color=0)
      else:
        if a[1] + a2 > img1.shape[0]:
          dst = Image.new('L', (img1.shape[1], a[1]+a2), color=0)
        else:
          dst = Image.new('L', (img1.shape[1], img1.shape[0]), color=0)
      caso = 1
      return dst, caso

    if(a[1] - (2/3)*a2> 0):          # Check if it is possible to fit 1/3 of image 2 above image 1
      sum_h = 0
      for ii in range(int((2/3)*a2)):
        sum_h = sum_h + img1[a[1]-ii, a[0]]
      if (sum_h == 0) and (a[0]+l2 < img1.shape[1]):                            # Check if the pixels above are empty
        if a2 > a[1]:                           # Check if it needs to increase the height of the image or not
          dst = Image.new('L', (img1.shape[1], img1.shape[0] + a2-a[1]), color=0)
        else:
          dst = Image.new('L', (img1.shape[1], img1.shape[0]), color=0)
        caso = 2
        return dst, caso

    if (a[0] + ((1/3)*l2)) < img1.shape[1]: # Check if it is possible to fit 1/3 of image 2 to the right of image 1
      sum_w = 0
      for ii in range(int((1/3)*l2)):
        sum_w = sum_w + img1[a[1], a[0]+ii]
      if sum_w == 0:                                 # Checks if the pixels on the right are empty
        if a[0]+l2 > img1.shape[1]:
          if a[1]+a2 > img1.shape[0]:
            dst = Image.new('L', (a[0]+l2, a[1]+a2), color = 0)
          else:
            dst = Image.new('L', (a[0]+l2, img1.shape[0]), color = 0)
        else:
          if a[1]+a2 > img1.shape[0]:
            dst = Image.new('L', (img1.shape[1], a[1]+a2), color = 0)
          else:
            dst = Image.new('L', (img1.shape[1], img1.shape[0]), color=0)                       # in case the image size doesn't change
        caso = 3
        return dst, caso
  return dst, caso

# Function to create and add the images in separate backgrounds
def two_dst(dst, img1, img2, caso, a):
  dst1 = dst.copy()
  dst2 = dst.copy()

  match caso:
    case 1:
      if img2.shape[1] > a[0]:
        dst1.paste(Image.fromarray(img1), (img2.shape[1]-a[0], 0))
        dst2.paste(Image.fromarray(img2), (0, a[1]))
      else:
        dst1.paste(Image.fromarray(img1), (0, 0))
        dst2.paste(Image.fromarray(img2), (a[0]-img2.shape[1], a[1]))
    case 2:
      if img2.shape[0] > a[1]:
        dst1.paste(Image.fromarray(img1), (0, img2.shape[0]-a[1]))
        dst2.paste(Image.fromarray(img2), (a[0], 0))
      else:
        dst1.paste(Image.fromarray(img1), (0, 0))
        dst2.paste(Image.fromarray(img2), (a[0], a[1]-img2.shape[0]))
    case 3:
      dst1.paste(Image.fromarray(img1), (0, 0))
      dst2.paste(Image.fromarray(img2), (a[0], a[1]))
  return dst1, dst2

# Function to check collision and if not, paste one image over the other
def check_dst(dst, img1, img2, caso, a):
  bandeira = True
  if caso == 0:
    bandeira = False
  else:
    dst1, dst2 = two_dst(dst, img1, img2, caso, a)
    for i in range((np.array(dst1)).shape[0]):
      for j in range((np.array(dst1).shape[1])):
        if (np.array(dst1))[i,j] == 255 and (np.array(dst2))[i,j] == 255:                 # if there is object in the same position
          bandeira = False
          break

  if bandeira == True:                                                               
    dst = np.array(dst)
    Y1,X1 = np.where((np.array(dst1))==255)
    dst[Y1,X1] = 255
    Y2,X2 = np.where((np.array(dst2))==255)
    dst[Y2,X2] = 255
  return dst, bandeira

# Function to search for the best fitting position
def best_fit(img1, img2, cnt, pag_size = page_r, d = distance):
  id = 0
  best_img = []
  flag = True
  pos_act = cnt[0,:]
  
  if len(cnt) < 200:
    dist = d
  elif len(cnt) >= 200 and len(cnt) < 500:
    dist = d+15
  else:
    dist = d+35
  
  while(id < len(cnt)):
    if distanceCalculate(cnt[id], pos_act) >= dist:
      dst, caso = new_dst(img1, img2.shape[0], img2.shape[1], cnt[id,:])
      dst, bandeira = check_dst(dst, img1, img2, caso, cnt[id,:])
      if bandeira == True:
        if len(best_img) > 0:
          if flag == False:
            if (dst.shape[0] <= pag_size[0]) and (dst.shape[1] <= pag_size[1]):
              best_img = dst
              flag = True
          else:
            if densi_image(dst) > densi_image(best_img):
              if (dst.shape[0] <= pag_size[0]) and (dst.shape[1] <= pag_size[1]):
                best_img = dst
                flag = True
        else:
          best_img = dst
          if (best_img.shape[0] > pag_size[0]) or (best_img.shape[1] > pag_size[1]):
            flag = False
      pos_act = cnt[id,:]
    id = id+1
  return best_img, flag

# Function to find the best rotation and fitting position
def best_fit_rotation(img1, img2, cnt, pag_size = page_r, dist = distance):
  flag = True
  best_img, flag = best_fit(img1, img2, cnt, pag_size, dist)
  angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
  for a in angles:
    new_img, flag1 = best_fit(img1, cv2.rotate(img2, a), cnt, pag_size, dist)
    if len(new_img) > 0:
      if len(best_img) > 0:
        if flag1 == True:
          if (flag == False) or (densi_image(new_img) > densi_image(best_img)):
            best_img = new_img
            flag = True
      else:
        best_img = new_img
        flag = flag1
  return best_img, flag

# Function to check if an image fits in another
def it_Fit(img1, img2, pag_size = page_r, dist = distance, mask = kernel):
  flag = True
  best_img = []
  img_dilation = cv2.dilate(img1, mask, iterations=1)
  img_dilation = figureBinary(img_dilation, threshold)
  contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  areas = []
  for i in range(len(contours)):
    areas.append(cv2.contourArea(contours[i]))
  if len(areas) > 0:
    if(max(areas) >= (1/3)*(img2.shape[0]*img2.shape[1])):
      idx = areas.index(max(areas))
      cnt = contours[idx][:,0,:]
      cnt = cnt[~(cnt[:,0] == 0), :]
      cnt = cnt[~(cnt[:,1] == 0), :]
      best_img, flag = best_fit_rotation(img1, img2, cnt, pag_size, dist)
  return best_img, flag

# Function to check if it fits at least one image of the pair (only for the 'couple_first' heuristic)
def it_Fit2(img1, list_path, pag_size, dist):
  kernel2 = np.ones((5,5), np.uint8)
  back_up = []
  flag = False
  
  for id in range(len(list_path)):
    best_img, flag1 = it_Fit(img1, call_Image(list_path[id]), pag_size, dist, kernel2)
    if flag1 == False:
      best_img, flag1 = best_match(img1, call_Image(list_path[id]), pag_size)
    if (flag1 == True) and (len(best_img) != 0):
      img1 = best_img
      list_path[id] = []
      flag = True
	  
  for i in range(len(list_path)):
    if len(list_path[i]) > 0:
      back_up.append(list_path[i])

  return img1, back_up, flag

# Function to concatenate images of different sizes horizontally
def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.shape[1] + im2.shape[1], max(im1.shape[0], im2.shape[0])), color=0) # create a new image with the greatest height and sum of widths
    dst.paste(Image.fromarray(im1), (0, 0)) # paste image 1
    dst.paste(Image.fromarray(im2), (im1.shape[1], 0)) # paste image 2 under image 1
    return asarray(dst)

# Function to concatenate different size images vertically
def get_concat_v(im1, im2):
    dst = Image.new('L', (max(im1.shape[1], im2.shape[1]), im1.shape[0] + im2.shape[0]), color=0) # create a new image with the largest width and sum of heights
    dst.paste(Image.fromarray(im1), (0, 0)) # paste image 1
    dst.paste(Image.fromarray(im2), (0, im1.shape[0])) # paste image 2 next to image 1
    return asarray(dst)

# Best combination - vertical or horizontal
def best_match_position(img1, img2, pag_size = page_r):
  flag = True
  img_h = get_concat_h(img1, img2)
  d_h = densi_image(img_h)
  img_v = get_concat_v(img1, img2)
  d_v = densi_image(img_v)

  if d_v > d_h:
    if img_v.shape[0] <= pag_size[0]:
      return img_v, flag
    else:
      if img_h.shape[1] <= pag_size[1]:
        return img_h, flag
      else:
        flag = False
        return img1, flag
  else:
    if img_h.shape[1] <= pag_size[1]:
      return img_h, flag
    else:
      if img_v.shape[0] <= pag_size[0]:
        return img_v, flag
      else:
        flag = False
        return img1, flag

# Best rotation
def best_match(img1, img2, pag_size = page_r):
  best_img, flag = best_match_position(img1, img2, pag_size)
  best_den = densi_image(best_img)

  angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
  for a in angles:
    img_r, flag1 = best_match_position(img1, cv2.rotate(img2, a), pag_size)
    if (densi_image(img_r) > best_den) and flag1 == True:
      best_img = img_r
      best_den = densi_image(img_r)
      flag = True

  return best_img, flag

# Best match
def best_match_Image(img, path, pag_size = page_r, dist = distance):
  img2     = call_Image(path[0])
  best_img, flag = it_Fit(img, img2, pag_size, dist)
  best_path = path[0]
  
  if len(best_img) > 0: # If there was a fit with image 2
    b_img, f2 = best_match(img, img2, pag_size)
    if f2 == True:
      if (flag == False) or (densi_image(b_img) > densi_image(best_img)):
        best_img = b_img
        flag = True
    best_den = densi_image(best_img)

  else: # If there wasn't a fit with image 2
    best_img, flag = best_match(img, img2, pag_size)
    best_den = densi_image(best_img)

  for i in range(len(path)-1):
    img2 = call_Image(path[i+1])
    new_img, flag1 = it_Fit(img, img2, pag_size, dist)
    if len(new_img) > 0:
      if flag1 == True:
        if (densi_image(new_img) > best_den) or (flag == False):
          best_img = new_img
          best_den = densi_image(new_img)
          best_path = path[i+1]
          flag = True
    new_img, flag1 = best_match(img, img2, pag_size)
    if flag1 == True:
      if (densi_image(new_img) > best_den) or (flag == False):
        best_img = new_img
        best_den = densi_image(new_img)
        best_path = path[i+1]
        flag = True

  return best_img, best_path, flag

# Start with the best 2 matching images
def start_couple(path, pag_size = page_r, dist = distance):
  img1 = call_Image(path[0])
  new_path = [x for x in path if x != path[0]]
  new_img, drop_path, f = best_match_Image(img1, new_path, pag_size, dist)

  for i in range(len(path)-1):
    img = call_Image(path[i+1])
    aux_path = [x for x in path if x != path[i+1]]
    img_aux, drop_aux, f = best_match_Image(img, aux_path, pag_size, dist)
    if densi_image(img_aux) > densi_image(new_img):
      new_img = img_aux
      drop_path = drop_aux
      new_path = [x for x in path if x != path[i+1]]

  new_path = [x for x in new_path if x != drop_path]
  return new_img, new_path

# Start with matching couple of the images (only for the 'couple_first' heuristic)
def couple_outPath(list_img, pag_size, dist):
  img1 = list_img[0]
  img2 = list_img[1]
  best_img, flag = it_Fit(img1, img2, pag_size, dist)
  pos = 1

  if len(best_img) > 0: # If there was a fit with the image
    b_img, f2 = best_match(img1, img2, pag_size)
    if f2 == True:
      if (flag == False) or (densi_image(b_img) > densi_image(best_img)):
        best_img = b_img
        flag = True
    best_den = densi_image(best_img)

  else: # If there wasn't a fit with the image
    best_img, flag = best_match(img1, img2, pag_size)
    best_den = densi_image(best_img)
  
  for i in range(len(list_img)-2):
    img2 = list_img[i+2]
    new_img, flag1 = it_Fit(img1, img2, pag_size, dist)
    if len(new_img) > 0:
      if flag1 == True:
        if (densi_image(new_img) > best_den) or (flag == False):
          best_img = new_img
          best_den = densi_image(new_img)
          pos = i+2
          flag = True
    new_img, flag1 = best_match(img1, img2, pag_size)
    if flag1 == True:
      if (densi_image(new_img) > best_den) or (flag == False):
        best_img = new_img
        best_den = densi_image(new_img)
        pos = i+2
        flag = True
  return best_img, pos, flag

# Check if fit one image of the couple or both separated (only for the 'couple_first' heuristic)
def check_false(img, list_p, pagsize, dists):
  best_img, b_list, flag = it_Fit2(img1 = img, list_path = list_p[1].copy(), pag_size = pagsize, dist = dists)
  best_den = densi_image(best_img)
  pos = 1
  
  for i in range(len(list_p)-2):
    new_img, rest, flag1 = it_Fit2(img1 = img, list_path = list_p[i+2].copy(), pag_size = pagsize, dist = dists)
    if flag1 == True:
      if (densi_image(new_img) > best_den) or flag == False:
        best_img = new_img
        b_list = rest
        best_den = densi_image(new_img)
        pos = i+2
        flag = True
  return best_img, b_list, pos, flag

def reunitedImg(PATH, pag_size = page_r, dist = distance):
  list_I, rest_l = [], []
  img1 = call_Image(PATH[0])
  new_path = [x for x in PATH if x != PATH[0]]
  aux_img, drop_path, f = best_match_Image(img1, new_path, pag_size, dist)
  new_path = [x for x in new_path if x != drop_path]
  
  if len(new_path) > 0:
    list_I.append(aux_img)
    list_I.append(call_Image(new_path[0]))
    rest_l.append([PATH[0]]+[drop_path])
    rest_l.append([new_path[0]])
  else:
    list_I = aux_img
    rest_l = PATH
  return list_I, rest_l
  
def pairInTwo(PATH, pag_size = page_r, dist = distance):
  list_ofImages = []
  list_ofPaths = []

  while len(PATH) > 0:
    pair_path   = []
    if len(PATH) == 1:
      list_ofImages.append(call_Image(PATH[0]))
      list_ofPaths.append([PATH[0]])
      PATH = []
    else:
      aux_img = call_Image(PATH[0])
      pair_path.append(PATH[0])
      new_path = [x for x in PATH if x != PATH[0]]
      Img, dPath, f = best_match_Image(aux_img, new_path, pag_size, dist)
      if f == True:
        list_ofImages.append(Img)
        pair_path.append(dPath)
        PATH = [x for x in new_path if x != dPath]
      list_ofPaths.append(pair_path)
  return list_ofImages, list_ofPaths

def pairInPair(PATH, pag_size = page_r, dist = distance):
  list_ofImages, list_ofPaths = pairInTwo(PATH, pag_size, dist)
  list_ofImages2 = []
  list_ofPaths2  = []
  while True:
    if len(list_ofImages) == 1:
      list_ofImages2.append(list_ofImages[0])
      list_ofPaths2.append(list_ofPaths[0])
      list_ofImages.pop(0)
      list_ofPaths.pop(0)
      break
    new_img, pos, flag = couple_outPath(list_ofImages, pag_size, dist)
    if flag == True:
      list_ofImages2.append(new_img)
      list_ofPaths2.append(list_ofPaths[0]+list_ofPaths[pos])
      list_ofImages.pop(pos)
      list_ofImages.pop(0)
      list_ofPaths.pop(pos)
      list_ofPaths.pop(0)    
    if len(list_ofImages) == 0:
      break
  return list_ofImages2, list_ofPaths2

################################################################################## HEURISTIC #####################################################################################

################################################## Start with the best pair
def best_start(PATH, pag_size = page_r, dist = distance):
  start_time = datetime.now()
  list_ofPages = []
  flag = True
  new_img, new_path = start_couple(PATH, pag_size, dist)

  while True:
    aux_img, drop_path, flag = best_match_Image(new_img, new_path, pag_size, dist)
    if flag == True:
      new_img = aux_img
      new_path = [x for x in new_path if x != drop_path]
    else:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      if len(new_path) == 1:
        p = paperA4(page_size)
        img = call_Image(new_path[0])
        p[0:img.shape[0], 0:img.shape[1]] = img
        list_ofPages.append(p)
        break
      else:
        new_img, new_path = start_couple(new_path, pag_size, dist)
        flag = True
      
    if len(new_path) == 0:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      break
  end_time = datetime.now()
  print('--- End of execution ---> {}'.format(end_time - start_time))
  return list_ofPages
##########################################################################################################################################

####################################################### Start with the first image
def first_start(PATH, pag_size = page_r, dist = distance):
  start_time = datetime.now()
  list_ofPages = []
  flag = True
  new_img = call_Image(PATH[0])
  new_path = [x for x in PATH if x != PATH[0]]

  while True:
    aux_img, drop_path, flag = best_match_Image(new_img, new_path, pag_size, dist)
    if flag == True:
      new_img = aux_img
      new_path = [x for x in new_path if x != drop_path]
    else:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      if len(new_path) == 1:
        p = paperA4(pag_size)
        img = call_Image(new_path[0])
        p[0:img.shape[0], 0:img.shape[1]] = img
        list_ofPages.append(p)
        break
      else:
        new_img = call_Image(new_path[0])
        aux_path = [x for x in new_path if x != new_path[0]]
        new_path = aux_path
        flag = True

    if len(new_path) == 0:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      break
  
  end_time = datetime.now()
  print('--- End of execution ---> {}'.format(end_time - start_time))
  return list_ofPages
#####################################################################################################################################

################################################ Start with the lower white pixels density image
def worst_first(PATH, pag_size = page_r, dist = distance, alg = 1):
  # Organizar o path por ordem da menor densidade a maior
  d = []
  path_image = [None]*len(PATH)
  for p in PATH:
    d.append(densi_image(call_Image(p)))
  sort_index = np.argsort(d)
  i = 0
  for id in sort_index:
    path_image[i] = PATH[id]
    i = i+1
  if alg == 1:
    list_ofPages = first_start(path_image, pag_size, dist)
  if alg == 2:
    list_ofPages = couple_first(path_image, pag_size, dist)
  return list_ofPages
######################################################################################################################################

###################################################### Start with a random image
def random_first(PATH, pag_size = page_r, dist = distance):
  start_time = datetime.now()
  list_ofPages = []
  flag = True
  drop_path = random.choice(PATH)
  new_img = call_Image(drop_path)
  new_path = [x for x in PATH if x != drop_path]

  while True:
    aux_img, drop_path, flag = best_match_Image(new_img, new_path, pag_size, dist)
    if flag == True:
      new_img = aux_img
      new_path = [x for x in new_path if x != drop_path]
    else:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      if len(new_path) == 1:
        p = paperA4(pag_size)
        img = call_Image(new_path[0])
        p[0:img.shape[0], 0:img.shape[1]] = img
        list_ofPages.append(p)
        break
      else:
        drop_path = random.choice(new_path)
        new_img = call_Image(drop_path)
        aux_path = [x for x in new_path if x != drop_path]
        new_path = aux_path
        flag = True

    if len(new_path) == 0:
      p = paperA4(pag_size)
      p[0:new_img.shape[0], 0:new_img.shape[1]] = new_img
      list_ofPages.append(p)
      break
  
  end_time = datetime.now()
  print('--- End of execution ---> {}'.format(end_time - start_time))
  return list_ofPages
##########################################################################################################################################

######################################### Start with matching pair of the images
def couple_first(PATH, pag_size = page_r, dist = distance):
  start_time = datetime.now()
  list_ofPages  = []
  list_ofImages, list_ofPaths = pairInPair(PATH, pag_size, dist)

  while True:
    if len(list_ofImages) == 1:
      p = paperA4(pag_size)
      p[0:list_ofImages[0].shape[0], 0:list_ofImages[0].shape[1]] = list_ofImages[0]
      list_ofPages.append(p)
      break
      
    new_img, pos, flag = couple_outPath(list_ofImages, pag_size, dist)
    if flag == True:
      list_ofImages[0] = new_img
      list_ofImages.pop(pos)
      list_ofPaths.pop(pos)
    else:
      new_img, rest_list, pos, flag = check_false(img = list_ofImages[0].copy(), list_p = list_ofPaths.copy(), pagsize = pag_size, dists = dist)
      if flag == True:
        list_ofImages[0] = new_img
        if len(rest_list) == 1:
          if densi_image(call_Image(rest_list[0])) < densi_image(list_ofImages[1]):
            list_ofPaths[pos] = list_ofPaths[1]
            list_ofImages[pos] = list_ofImages[1]
            list_ofPaths[1] = rest_list
            list_ofImages[1] = call_Image(rest_list[0])
          else:
            list_ofPaths[pos] = rest_list
            list_ofImages[pos] = call_Image(rest_list[0])
        elif len(rest_list) > 1:
          rest_img, rest_l = reunitedImg(rest_list, pag_size, dist)
          if len(rest_list) == 3: 
            list_ofPaths[pos] = rest_l[0]
            list_ofPaths.append(rest_l[1])
            list_ofImages[pos] = rest_img[0]
            list_ofImages.append(rest_img[1])
          else:
            list_ofPaths[pos] = rest_l
            list_ofImages[pos] = rest_img
        else:
          list_ofImages.pop(pos)
          list_ofPaths.pop(pos)
      else:
        p = paperA4(pag_size)
        p[0:list_ofImages[0].shape[0], 0:list_ofImages[0].shape[1]] = list_ofImages[0]
        list_ofPages.append(p)
        list_ofImages.pop(0)
        list_ofPaths.pop(0)
  end_time = datetime.now()
  print('--- End of execution ---> {}'.format(end_time - start_time))
  return list_ofPages

def nestin_probl_funct(PATH, page_size = page_r, dist = distance, function_name = 'pac.FIRST_START'):
  match function_name:
    case 'pac.FIRST_START':
      list_ofPages = first_start(PATH, page_size, dist)
      return list_ofPages
    case 'pac.WORST_FIRST':
      list_ofPages = worst_first(PATH, page_size, dist, 1)
      return list_ofPages
    case 'pac.BEST_START':
      list_ofPages = best_start(PATH, page_size, dist)
      return list_ofPages
    case 'pac.RANDOM_FIRST':
      list_ofPages = random_first(PATH, page_size, dist)
      return list_ofPages
    case 'pac.COUPLE_FIRST':
      list_ofPages = worst_first(PATH, page_size, dist, 2)
      return list_ofPages
