import numpy as np 
import cv2 

#samp = np.zeros([300, 300, 3], dtype = 'uint8')
# Draw a green line from the top-left corner of our canvas
# to the bottom-right
#green = (0, 255, 0)
#cv2.line(samp, (0, 0), (300, 300), green)
#cv2.imshow("Samp", samp)
#cv2.waitKey(0) 

# Now, draw a 3 pixel thick red line from the top-right
# corner to the bottom-left
#red = (0, 0, 255)
#cv2.line(samp, (300, 0), (0, 300), red, 5)
#cv2.imshow("Samp", samp)
#cv2.waitKey(0)
# Reset our canvas and draw a white circle at the center
# of the canvas with increasing radii - from 25 pixels to
# 150 pixels
canvas = np.zeros([350, 350, 3], dtype = "uint8")
# Calculating the center of the image 
#(centerX,centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
#red = (0, 0, 255)

#print(centerX,centerY)
#for r in range(0, 175, 25):
	#cv2.circle(canvas, (centerX, centerY), r, red)

#cv2.imshow("Canvas", canvas)
#cv2.waitKey(0)


# Let's go crazy and draw 25 random circles
for i in range(0, 25):
	# randomly generate a radius size between 5 and 200,
	# generate a random color, and then pick a random
	# point on our canvas where the circle will be drawn
	radius = np.random.randint(5, high = 200)
	color = np.random.randint(0, high = 256, size = (3,)).tolist()
	pt = np.random.randint(0, high = 300, size = (2,))

	# draw our random circle
	cv2.circle(canvas, tuple(pt), radius, color, -1)

# Show our masterpiece
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)