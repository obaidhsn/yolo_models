import cv2
import numpy as np

# Load the image
image = cv2.imread('address_1_line1.png')

original_height, original_width = image.shape[:2]
    
# Double the dimensions
new_width = original_width * 5
new_height = original_height * 5

# Resize image to double the size
image = cv2.resize(image, (new_width, new_height))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to binarize the image
_, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("test.png", binary)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by their x-coordinate
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# Initialize a list to store the word images
word_images = []

# Initialize previous x-end value for calculating spaces between words
previous_x_end = None

# Define a threshold for what constitutes a "significant" space (in pixels)
# You can adjust this threshold depending on your image and requirements
space_threshold = 15  # Adjust this value based on the spacing in your image
padding_between_words = 50  # Padding with zeros between words (in pixels)

# Create a blank (white) space image for padding between words
max_height = image.shape[0]
space_image = np.ones((max_height, padding_between_words, 3), dtype=np.uint8) * 255

# Function to pad images to the same height
def pad_to_max_height(word_image, max_height):
    height, width = word_image.shape[:2]
    # If the word image is already of max height, no need to pad
    if height == max_height:
        return word_image
    # Create a white image of max height and current width
    padded_image = np.ones((max_height, width, 3), dtype=np.uint8) * 255
    # Place the word image in the top left corner
    padded_image[:height, :width] = word_image
    return padded_image

# Loop through contours and extract word images with spaces
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate space between current word and the previous word
    if previous_x_end is not None:
        space_between_words = x - previous_x_end
        
        # If the space is significant, add padding (zeros)
        if space_between_words > space_threshold:
            word_images.append(space_image)
    
    # Extract and store the word image
    word_image = image[y:y+h, x:x+w]
    padded_word_image = pad_to_max_height(word_image, max_height)
    word_images.append(padded_word_image)
    
    # Update the previous x-end value
    previous_x_end = x + w

# Concatenate all word images with necessary spaces in between
output_image = word_images[0]
for word_image in word_images[1:]:
    output_image = np.hstack((output_image, word_image))

# Save the final image with spaces between words
output_path = 'output_with_spaces.png'
cv2.imwrite(output_path, output_image)

# Optionally, display the final image
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
