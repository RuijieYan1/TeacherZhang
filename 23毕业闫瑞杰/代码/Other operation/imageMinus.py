#图像减法 - 血液流动
import cv2
img1 = cv2.imread("Our method/ce201703.png")
img2 = cv2.imread("Our method/ce202003.png")

bitwiseXor = cv2.bitwise_xor(img1,img2)

cv2.imshow("bitwiseXor异或运算：",bitwiseXor)
cv2.imwrite("output/ce03.png", bitwiseXor)
cv2.waitKey(0)
cv2.destroyAllWindows()