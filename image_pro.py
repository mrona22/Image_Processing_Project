

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# Part 1: RGB Image #
class RGBImage:
    """
    # A Class to #
    """

    def __init__(self, pixels):
        """
        # Takes in a variable name and a matrix made up of lists, where each
        row has the same amount of elements and the most inner list is a list 
        of 3 ints with values between 0 and 255 #

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        if type(pixels) != list and len(pixels) == 0:
            raise TypeError
        if not all([isinstance(i, list) for i in pixels]):
            raise TypeError
        if not all([len(pixels[0]) == len(i) and len(i) > 0 for i in pixels]):
            raise TypeError
        if not all([len(i) == 3 for j in pixels for i in j]):
            raise SyntaxError
        if not all([max(i) <= 255 for j in pixels for i in j]):
            raise ValueError



        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        # takes in a instance of a class, return the number of columns and the 
        number of rows in a tuple#

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        # Makes a deep copy of the pixel of a instance of a class #

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[i[0], i[1], i[2]] for i in j] for j in self.pixels]


    def copy(self):
        """
        # creates a copy of the input instance, making a new instance thats
        the same.  #

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        copied_img = self.get_pixels()
        #new_instance = RGBImage(copied_img)

        #return new_instance
        return RGBImage(copied_img)

    def get_pixel(self, row, col):
        """
        # Returns the color of the pixel at the designated position. #

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        #if type(row) != int or type(col) != int:
            #raise TypeError
        if row < 0 or col < 0:
            raise ValueError
        try:
            return (self.pixels[row][col][0],self.pixels[row][col][1],
                self.pixels[row][col][-1])
        except ValueError:
            raise ValueError
        except IndexError:
            raise ValueError

        

    def set_pixel(self, row, col, new_color):
        """
        # Updates the designated pixels at the specific row and column if 
        the value isnt negative. #

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not (isinstance(row,int) and isinstance(col, int)):
            raise TypeError
        try:
            self.pixels[row][col]
        except ValueError:
            raise ValueError
        if not isinstance(new_color,tuple) or len(new_color) != 3:
            raise TypeError
        if not all([isinstance(i,int) for i in new_color]):
            raise TypeError
        if max(new_color) > 255:
            raise ValueError

        for i in range(len(self.pixels[row][col])):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
        return 


class ImageProcessingTemplate:
    """
    TODO: a class for images and to keep track of cost 
    """

    def __init__(self):
        """
        # A constructor,creates an instance attributes cost and sets it 
        to zero#

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        # Returns the instance attribute cost #

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        # A class method that returns a negative of a given image #

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """

        negative_img = [
        [[255 - i for i in j] 
        for j in k]
        for k in image.get_pixels()
        ]
        
        return RGBImage(negative_img)
        #new_instance = RGBImage(negative_img)
        #return new_instance


    def grayscale(self, image):
        """
        # A class method that takes the replaces the RGBI of a pixel with the 
        avarage of them. #

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        gray_scale_img = [[
        [sum(j) // len(j), sum(j) // len(j), sum(j) // len(j)] for j in i] 
        for i in image.get_pixels()]
        return RGBImage(gray_scale_img)


    def rotate_180(self, image):
        """
        # A class method that rotates the image. Returns a new class instance#

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        rotated_img = [i[::-1] for i in image.get_pixels()][::-1]

        #rotated_img = [i.reverse() for i in image.pixels].reverse()

        return RGBImage(rotated_img)


class StandardImageProcessing(ImageProcessingTemplate):
    """
    # A class for image processing, ImageProcessingTemplate is its super#
    """

    def __init__(self):
        """
        # constructor that initilizes cost, number of rotates coupon to zero
        as class attribute #

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.number_of_rotates = 0
        self.coupon = 0

    def negate(self, image):
        """
        # Negates an image and updates cost to 5#

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupon > 0:
            self.coupon -= 1

        else:
            self.cost = 5

        return ImageProcessingTemplate.negate(self,image)
        

    def grayscale(self, image):
        """
        # grayscale's the image and updates cost to 6 # 

        """
        if self.coupon > 0:
            self.coupon -= 1
        else :
            self.cost = 6
        return ImageProcessingTemplate.grayscale(self,image)

    def rotate_180(self, image):
        """
        # Rotates the given image for a cost of 10, rotating back to original
        is free #

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """

        self.number_of_rotates += 1
        if self.coupon > 0:
            self.coupon -= 1
        elif not self.number_of_rotates % 2 == 0:
            self.cost = 10
        else:
            self.cost = 0
        return ImageProcessingTemplate.rotate_180(self, image)



    def redeem_coupon(self, amount):
        """
        # Takes in an int, the next number of input times, there will be no 
        cost for a action#

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount,int):
            raise TypeError
        if amount <= 0:
            raise ValueError
        self.coupon += amount

        return 


class PremiumImageProcessing(ImageProcessingTemplate):
    """
    # A class with initialized cost, with super ImageProcessingTemplate#
    """

    def __init__(self):
        """

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        # Changes the pixels of the input picture1 to the pixel of input 
        picture 2 if the color matches the input color#

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        if not all([isinstance(chroma_image, RGBImage),isinstance(
            background_image, RGBImage)]):
            raise TypeError
        elif chroma_image.size() != background_image.size():
            raise ValueError

        for i in range(RGBImage.size(chroma_image)[0]):
            for j in range(RGBImage.size(chroma_image)[1]):
                if chroma_image.get_pixel(i,j) == color:
                    chroma_image.set_pixel(i, j, background_image.get_pixel(
                        i,j))
        return RGBImage(chroma_image.get_pixels())







    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        # Changes the background_image pixels from x_pos and y_pos to the
        pixels of sticker_image#

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if not all([isinstance(sticker_image, RGBImage),isinstance(
            background_image, RGBImage)]):
            raise TypeError

        if sticker_image.size()[0] > background_image.size()[0]:
            raise ValueError
        if sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError
        if not all([isinstance(x_pos,int), isinstance(y_pos,int)]):
            raise TypeError
        if sticker_image.size()[0] + x_pos > background_image.size()[0]:
            raise ValueError
        if sticker_image.size()[1] + x_pos > background_image.size()[1]:
            raise ValueError


        for i in range(sticker_image.size()[0]):
            for j in range(sticker_image.size()[1]):
                background_image.set_pixel(
                    x_pos + i, y_pos + j, sticker_image.get_pixel(i,j))

        return RGBImage(background_image.get_pixels())




class ImageKNNClassifier:
    """
    # A class that finds the group of the picture #
    """

    def __init__(self, n_neighbors):
        """
        # Takes in a class attribute that finds from the n nearest neibor the
        most common group #
        """
        self.n_neighbors = n_neighbors
        self.data = []


    def fit(self, data):
        """
        # creates a instance variable data, to store the data in#

        # make random training data (type: List[Tuple[RGBImage, str]])
        >>> train = []

        # create training images with low intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        ...     for _ in range(20)
        ... )

        # create training images with high intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        ...     for _ in range(20)
        ... )

        # initialize and fit the classifier
        >>> knn = ImageKNNClassifier(5)
        >>> knn.fit(train)

        # should be "low"
        >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
        low

        # can be either "low" or "high" randomly
        >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
        This will randomly be either low or high

        # should be "high"
        >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
        high
        """
        if len(data) < self.n_neighbors:
            raise ValueError
        if len(self.data) != 0:
            raise ValueError
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        # calculates the Eucledian distance from pixels of 2 images of the 
        same size# 
        """
        if not all([isinstance(image1, RGBImage), isinstance(
            image2, RGBImage)]):
            raise TypeError
        if image1.size() != image2.size():
            raise ValueError
        image_1_1d = [k 
        for i in image1.get_pixels() 
        for j in i 
        for k in j]

        image_2_1d = [k 
        for i in image2.get_pixels() 
        for j in i 
        for k in j]

        squared_sum = sum([(image_1_1d[i] - image_2_1d[i]) ** 2 
            for i in range(len(image_1_1d))])
        return squared_sum ** 0.5



    @staticmethod
    def vote(candidates):
        """
        # Returns the most common element from the second index of a list of
        tuples #
        """
        lst_of_labels = [i[1] for i in candidates]
        most_common = max(set(lst_of_labels), key = lst_of_labels.count)
        return most_common


    def predict(self, image):
        """
        # Predicts for an image what group of images it is the closest to #
        """
        if len(self.data) == 0:
            raise ValueError
        all_dis_n_labl = [(ImageKNNClassifier.distance(image, i[0]), i[1])
            for i in self.data]
        sorted_label = sorted(all_dis_n_labl, key=lambda x: x[0])
        m_common = ImageKNNClassifier.vote(sorted_label[:self.n_neighbors])
        return m_common



# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)

def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()
