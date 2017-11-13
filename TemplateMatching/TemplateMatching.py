#Yimeng Wang wym613@bu.edu
"""TemplateMatching_HW3"""
import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize):
    " src: source image, temp: template image, stepsize: the step size for sliding the template "
    mean_t = 0;
    var_t = 0;
    location = [0, 0];
    # Calculate the mean and variance of template pixel values
    mean_temp = np.mean(temp)
    var_temp = np.var(temp)
                    
    max_corr = 0;
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            mean_s = 0;
            var_s = 0;
            corr = 0;
            # Calculate the mean and variance of source image pixel values inside window
            mean_window = np.mean(src[i:i+temp.shape[0], j:j+temp.shape[1]])
            var_window = np.var(src[i:i+temp.shape[0], j:j+temp.shape[1]])
            # Calculate normalized correlation coefficient (NCC) between source and template
            mul = (src[i:i+temp.shape[0],j:j+temp.shape[1]]-mean_window)*(temp-mean_temp)
            sum_part = sum(sum(mul[i]) for i in range(len(mul)))
            corr = (1/float((temp.shape[0])*(temp.shape[1]))) * sum_part / ((var_temp)*(var_window))
            
            if corr > max_corr:
                max_corr, location = corr, [i, j]
    return location

# load source and template images
source_img = cv2.imread('source_img.jpg',0) # read image in grayscale
temp = cv2.imread('template_img.jpg',0) # read image in grayscale
location = TemplateMatching(source_img, temp, 20);
print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

# Draw a red rectangle on match to show the template matching result
for i in range(location[0],location[0]+temp.shape[0]):
         match_img[i][location[1]][2] = 255
         match_img[i][location[1]+temp.shape[1]][2] = 255
for i in range(location[1],location[0]+temp.shape[1]):
         match_img[location[0]][i][2] = 255
         match_img[location[0]+temp.shape[0]][i][2] = 255
        

# Save the template matching result image (match)
cv2.imwrite('match_img.jpg',match_img)

# Display the template image and the matching result
cv2.imshow('TemplateImage', temp)
cv2.imshow('TemplateMatching', match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
