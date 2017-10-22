#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class BodyDetector {
public:
	virtual void detect_bodies(cv::Mat frame, std::vector<cv::Rect>& eyes) {};
};

class BodyCascadeClassifier : public BodyDetector {
private:
	cv::CascadeClassifier classifier;
public:
	BodyCascadeClassifier(std::string template_file) {
		classifier.load(template_file);
	}
	void detect_bodies(cv::Mat frame, std::vector<cv::Rect>& bodies) {
		int min_size = frame.rows / 10;
		classifier.detectMultiScale(frame, bodies, 2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(min_size, min_size));
	}
};