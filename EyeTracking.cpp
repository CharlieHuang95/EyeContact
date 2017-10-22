#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class EyeDetector {
public:
	virtual void detect_eyes(cv::Mat frame, std::vector<cv::Rect>& eyes) {};
};

class EyeCascadeClassifier : public EyeDetector {
private:
	cv::CascadeClassifier classifier;
public:
	EyeCascadeClassifier(std::string template_file) {
		classifier.load(template_file);
	}
	void detect_eyes(cv::Mat frame, std::vector<cv::Rect>& eyes) {
		cv::Size min_size(1, 1);
		cv::Size max_size(frame.cols, frame.rows);
		classifier.detectMultiScale(frame, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, min_size);//, max_size);
	}
};