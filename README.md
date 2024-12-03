# cv-final-project

Objective: Create a filter that sharpens the foreground and blurs the background (basically the portrait filter on iphone)

  Pre-trained model choice: Openface/Facenet

    Code outline: 
      1. upload pre-trained model
      
        import dlib
        import numpy as np
        import cv2
        
      2. preprocess the image so that we separate the foreground and background from each other and apply filters

        cv2.imread(image)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
          1. detect landmarks on the face
              landmarks = predictor(gray, face)
              points = [(p.x, p.y) for p in landmarks.parts()]
          2. create face mask
            face_mask = get_face_mask(image, points)
          3. sharpen the foreground and blur the background
            ex: sharpening_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                blurring_kernel = GaussianBlur(image, (21,21), 0)
          5. combine

  Pre-trained model choice: MTCNN

    Code outline:
    #MTCNN stands for Multi-Task Cascaded Convolutional Neural Networks. 
    #It is designed to detect faces and facial landmarks in an image and 
    #has a cascade structure with three neural networks. MTCNN works well 
    #in identifying faces in different angles and lighting. 
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN

    #STEP 1: import images from the stereo camera image dataset and 
    #extract the bounding box for the face and facial landmarks using 
    #MTCNN

    #STEP 2: after detecting the face, we can crop the face region and 
    #enhance it using cv2 functions

    #STEP 3: blur the background of the image using a Gaussian blur from 
    #cv2

    #STEP 4: combine the two images of the enhanced face and blurred 
    #background, so that the enhanced face is placed back into the image. 
    #To blend the two images nicely, we can use cv2.addWeighted. 

    #initialize MTCNN
    detector = MTCNN()

    #load image
    img = cv2.imread("image_path.jpg")

    #detect faces
    faces = detector.detect_faces(img)

    #enhance subject and blur background
    x, y, w, h = face['box']
    landmarks = face['keypoints']

    #STEPS 3 and 4
