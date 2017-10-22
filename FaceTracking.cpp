#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class FaceDetector {
public:
	virtual void detect_faces(cv::Mat frame, std::vector<cv::Rect>& faces) {};
};

class FaceCascadeClassifier : public FaceDetector {
private:
	cv::CascadeClassifier face_classifier;
public:
	FaceCascadeClassifier(std::string template_file) {
		face_classifier.load(template_file);
	}
	void detect_faces(cv::Mat frame, std::vector<cv::Rect>& faces) override {
		face_classifier.detectMultiScale(frame, faces, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));// , cv::Size(120, 120));
	}
};