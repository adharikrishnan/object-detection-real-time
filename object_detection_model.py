# Imports OpenCV 
import cv2 as cv

# Imports the definitions from the defs.py file
import defs


class SSD_Object_Detection():

    # Class Constructor
    def __init__(self, ScreenName, camera = 0, Accuracy_Threshold = 0.45):

        # Initializes the name of the Camera Screen
        self.ScreenName = ScreenName

        # The accuracy threshold is used as the threshold for object detection
        # By default it is set to 0.45 (45%)
        self.Accuracy_Threshold = Accuracy_Threshold

        # Sets the input source to the default webcam (index 0 , the default value for the camera parameter)
        # If the system has more than a single camera, specify the index as a second parameter when creating an instance
        self.cSource = cv.VideoCapture(camera) 

        # Creates the screen that reads from the Camera
        cv.namedWindow(self.ScreenName,cv.WINDOW_NORMAL)
        
    # Helper function that reads in the txt file and stores
    # the contents into an array and returns it
    def Read_Labels(self,file_name):
        with open(file_name) as fp:
            class_labels = fp.read().split("\n")
        return class_labels

    # Performs pre-processing on the input image (frame) and feeds this into the SSD Neural Network
    # The function then retuns any objects detected
    def Object_Detection(self,NeuralNet, frame):
        
        # Dimensions of the input into the Neural Network
        inputDimensions = (300,300)

        # Preprocesses the image and returns a blob representation of the image for inference 
        fBlob = cv.dnn.blobFromImage(frame, 1.0, size=inputDimensions, mean = (0,0,0), swapRB = True, crop=False)

        # Passes in the blob representation of image as input to the Neural Network
        NeuralNet.setInput(fBlob)

        # Performs inference and returns any objects detected with the given input
        object_detections = NeuralNet.forward()

        return object_detections


    # Helper Function used to annotate a bounding box
    # This is used to display labels for the objects detected and the accuracy for each detection
    def Annotation(self, frame, display_text, x, y, colors):

        # Calculates text dimensions and baseline to scale bounding box that will store this image accordingly  
        text_size = cv.getTextSize(display_text, cv.FONT_HERSHEY_COMPLEX,0.5, 1)
        dimensions = text_size[0]
        baseline = text_size[1]

        # Creates a white boudning box to store the text
        cv.rectangle(frame,(x, y - dimensions[1] - baseline), (x + dimensions[0], y + baseline), (255,255,255), cv.FILLED)

        # Creates the annotation in the the color specified by the colors parameter
        cv.putText(frame, display_text, (x, y - 3), cv.FONT_HERSHEY_COMPLEX, 0.5, colors, 1, cv.LINE_AA)


    # This function displays the objects detected by putting a bounding box around them with
    # the object label and accuracy of the prediction 
    def Display_Detections(self, frame, object_detections):

        # Stores the dimensions for the input image in two variables
        fHeight = frame.shape[0]
        fWidth = frame.shape[1]

        # Reads in the class labels for the detection model and stores then
        # a list to reference
        Class_Labels = self.Read_Labels(defs.CLASS_LABELS)

        # Iterates throught the objects detected by the model
        for x in range(object_detections.shape[2]):

            # Stores the Class ID for the object detected to for labeling
            Class_ID = int(object_detections[0,0,x,1])

            # Stores the Accuracy for the detection 
            Accuracy = float(object_detections[0,0,x,2])

            # Converts normalised coordinates for the detections to real coordinates
            # to create the bounding boxes for the detections
            xPos = int(object_detections[0,0,x,3] * fWidth)
            yPos = int(object_detections[0,0,x,4] * fHeight)
            Width = int(object_detections[0,0,x,5] * fWidth - xPos)
            Height = int(object_detections[0,0,x,6] * fHeight - yPos)
            
            # Checks if the accuracy of the object detected is beyond the threshold
            # and only then is the bounding box and lable displayed, signifying the object has been detecetd
            if Accuracy > self.Accuracy_Threshold:

                # This if statement is used to prevent the array containing the labels going out of index 
                if Class_ID < 80:

                    # Annotates the label for the object within a white bounding box
                    self.Annotation(frame, "{}".format(Class_Labels[Class_ID-1]), xPos, yPos, (0,0,0))

                    # Annotates the Accuracy for the detection
                    self.Annotation(frame, "Accuracy %.4f" % Accuracy , xPos + 100, yPos, (255,0,0))

                    # Creates the bounding box rectangle around the detected object
                    cv.rectangle(frame, (xPos, yPos), (xPos + Width, yPos + Height), (255,255,255), 2)   


    # Function that runs the model by taking in the webcam feed as input
    def Run_Model(self):

        # Reads in the Tensorflow implementaion of the file and creates an instance of the Neural Network
        NeuralNet = cv.dnn.readNetFromTensorflow(defs.SSD_FROZEN_INFERENCE_GRAPH_V2, defs.SSD_CONFIG_FILE_V2)
    

        # Camera Feed loop, which runs until the window is closed or the exit key is pressed
        while self.cSource.isOpened():
            
            # Reads in the current frame of the camera feed, and returns the status of the frame
            # and the frame itself
            cFramePresent ,cFrame = self.cSource.read()

            # filps the camera to prevent the mirror effect
            cFrame = cv.flip(cFrame,1)
            
            # Performs inference on the current frame and returns the objects detected
            objects_detected = self.Object_Detection(NeuralNet, cFrame)

            # Display the boudning box for the objects detected that pass the accuracy threshold
            self.Display_Detections(cFrame, objects_detected)

            # Checks for the 'Esc' key being pressed
            # If it is the loop is broken and the feed is stopped
            if cv.waitKey(1) == 27:
                break

            # Checks if the 'p' key has been pressed
            # If it is, it pauses the video feed
            # it can be resumed by pressing any key
            if cv.waitKey(1) == ord('p'):
                cv.waitKey(0)    

            # Error detection
            # checks for any errors from the current frame
            # if the value is false, the feed is stopped
            if not cFramePresent:
                print('Camera Error, Feed Stopping')
                break

            # Displays the camera feed to screen
            cv.imshow(self.ScreenName, cFrame)

        # Destroys the feed and window after the loop is broken 
        self.cSource.release()
        cv.destroyWindow(self.ScreenName) 
    
