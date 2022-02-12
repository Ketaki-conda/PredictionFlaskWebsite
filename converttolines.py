import cv2
import numpy as np


#import image
def convert_to_words(imagepath):
    image = cv2.imread(imagepath)
    image_1 = cv2.imread(imagepath)
    image_2 = cv2.imread(imagepath)
    # cv2.imshow('orig',image)
    # cv2.waitKey(0)

    # img = cv2.imread(imagepath)

    # dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    # cv2.imshow('denoise', dst)
    # cv2.waitKey(0)
    # cv2.imwrite("results/dst.png", dst)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # cv2.imwrite("results/gray.png", gray)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('second', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite("results/thresh.png", thresh)

    # dilation
    kernel = np.ones((1, 100), np.uint8)
    # kernel = np.ones((100, 10), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated', img_dilation)
    cv2.waitKey(0)
    # cv2.imwrite("results/dilated.png", img_dilation)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    count = 0
    c = 0
    for i, ctr in enumerate(reversed(sorted_ctrs)):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]
        # cv2.imwrite("lines/" + str(c) + ".png", roi)
        # c += 1 
        kernel_word = np.ones((100, 10), np.uint8)
        gray_word = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ret_word, thresh_word = cv2.threshold(gray_word, 127, 255,
                                              cv2.THRESH_BINARY_INV)

        img_dilation_word = cv2.dilate(thresh_word, kernel_word, iterations=1)

        ctrs_word, hier_word = cv2.findContours(img_dilation_word.copy(),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

        sorted_ctrs_words = sorted(ctrs_word,
                                   key=lambda ctr: cv2.boundingRect(ctr)[0])

        for j, ctr_word in enumerate(sorted_ctrs_words):
            x_word, y_word, w_word, h_word = cv2.boundingRect(ctr_word)
            cv2.rectangle(roi, (x_word, y_word), (x_word +
                        w_word, y_word + h_word), (84, 24, 62), 2)
            cv2.rectangle(image_1, (x + x_word, y + y_word), (x + x_word +
                        w_word, y + y_word + h_word), (84, 24, 62), 2)
            word = gray[y + y_word:y + y_word + h_word,
                        x + x_word:x + x_word + w_word]
            cv2.imwrite("words/" + str(count + 1) + ".png", word)
            count += 1

        # show ROI
        # cv2.imshow('segment no:'+str(i), roi)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        # cv2.rectangle(image_2, (x, y), (x + w, y + h), (90, 0, 255), 2)
        # cv2.waitKey(0)

        # cv2.imshow('marked areas', image_2)
        # cv2.waitKey(0)
        # cv2.imwrite("results/sentences.png", image_2)

        # cv2.imshow('marked areas', image_1)
        # cv2.waitKey(0)
        # cv2.imwrite("results/words.png", image_1)

        # cv2.imshow('marked areas', image)
        # cv2.waitKey(0)
        # cv2.imwrite("results/result.png", image)

# convert_to_words("abcde.jpeg")