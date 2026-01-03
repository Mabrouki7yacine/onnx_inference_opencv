#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> load_class_labels(string labelsPath) {
    vector<string> classLabels;
    ifstream labelsFile(labelsPath);
    if (!labelsFile.is_open()) {
        cerr << "Error: Could not open the class labels file!" << endl;
        return {};
    }
    string line;
    while (getline(labelsFile, line)) {
        classLabels.push_back(line);
    }
    labelsFile.close();
    return classLabels;
}

Mat load_image(string imagePath) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Error: Could not load the input image!" << endl;
        return {};
    }
    return image;
}

// this process_image func was set up for ResNet-50
void process_image(Mat image, Net net, vector<string> classLabels){
    Mat inputBlob;
    Size inputSize(224, 224);  // ResNet-50 expects 224x224 images
    blobFromImage(image, inputBlob, 1.0 / 255.0, inputSize, Scalar(0, 0, 0), true, false);

    // Set the blob as input to the network
    net.setInput(inputBlob);

    // Perform forward pass to get predictions
    Mat output = net.forward();

    // Find the class with the highest confidence
    Point classId;
    double confidence;
    minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classId);

    // Output
    cout << "Predicted Class ID: " << classId.x << endl;
    cout << "Confidence: " << confidence << endl;
    if (classId.x >= 0 && classId.x < classLabels.size()) {
        cout << "Predicted Class Name: " << classLabels[classId.x] << endl;
    } else {
        cout << "Class ID is out of range!" << endl;
    }  
}

int main() {

    VideoCapture cap(0);

    // Check if the webcam is opened successfully
    if (!cap.isOpened()) {
        cerr << "Error: Unable to access the webcam" << endl;
        return -1;
    }
    // Paths to the ONNX model and class labels
    string onnxModelPath = "/home/yacine/path/to/model.onnx";
    string labelsPath = "/home/yacine/path/to/classes.txt";

    // Load the ONNX model
    Net net = readNetFromONNX(onnxModelPath);
    if (net.empty()) {
        cerr << "Error: Could not load the ONNX model!" << endl;
        return -1;
    }
    cout << "ONNX model loaded successfully!" << endl;

    // Load class labels
    vector<string> classLabels = load_class_labels(labelsPath);

    cout << "Press 'q' to exit the video stream." << endl;
    namedWindow("Webcam", WINDOW_AUTOSIZE);

    Mat frame;
    while (true) {
        // Capture each frame from the webcam
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Received empty frame" << endl;
            break;
        }

        flip(frame, frame, 1);
        process_image(frame, net, classLabels);
        imshow("Webcam", frame);

        // Break the loop if 'q' is pressed
        if (waitKey(30) == 'q') {
            cout << "Exiting video stream..." << endl;
            break;
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}
