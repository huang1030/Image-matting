import os
import cv2
  
path = "things/tu/"
print(path)
  
for filename in os.listdir(path):
  if os.path.splitext(filename)[1] == '.png':
#    print(filename)
    # print(filename)
    img = cv2.imread(path + filename)
    print(path + filename)
#    print(filename.replace(".png",".jpg"))
    newfilename = filename.replace(".png",".jpg")
#    print(newfilename)
    # cv2.imshow("Image",img)
    # cv2.waitKey(0)
    cv2.imwrite('things/tu/' + newfilename,img)