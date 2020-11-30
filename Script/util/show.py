import cv2

def show(picture,name='default'):
    try:
        cv2.imshow(name,picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("***************************************")
        print("There was an error showing picture")
        print(e)
        if picture is None:
            print("Detected that the picture to show is None")
        print("***************************************")
