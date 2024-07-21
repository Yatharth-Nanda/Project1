The 9thaprilclassifier program is desgined as follows 

1.) Upon pressing the spacebar , a 5 sec timer countdown is initiaed after which the frame is stored as an image and read again for the model to process and give landmarks 



These landmarks are then preprocessed to normalize their coordinates and prepare them for classification. The script includes functions for capturing frames from the webcam feed, allowing users to trigger the capture by pressing the spacebar. After capturing a frame, it processes the landmarks for gesture recognition using a pre-trained model. The detected gestures are then displayed on the processed frame for visual feedback.

Upon a succesful detection , three other windows are opened 
1.) Captured frame : to show the captured frame 
2.) Hand Landmarks : to show the marked hand position and debugging 
3.) Processed Frame: with the output 