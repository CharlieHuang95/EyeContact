#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class PupilDetector {
public:
	virtual void detect_pupil(cv::Mat frame, std::vector<cv::Vec3f>& pupils) {};
};

class PupilHoughDetector : public PupilDetector {
public:
	PupilHoughDetector() {}
	void detect_pupil(cv::Mat frame, std::vector<cv::Vec3f>& pupils) {
		cv::Mat morphed_eye;// = face(eye_rect);
		int morph_elem = 2;
		int morph_size = 3;
		int operation = 3;
		cv::cvtColor(frame, morphed_eye, CV_BGR2GRAY);
		cv::Mat element = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
		/// Apply the specified morphology operation
		morphologyEx(morphed_eye, morphed_eye, operation, element);

		// Canny(eye_area, eye_area, 10 /*lowThreshold*/, 30 /*lowThreshold*ratio*/, 3 /*kernel_size*/);
		HoughCircles(morphed_eye, pupils, CV_HOUGH_GRADIENT, 4, frame.cols, 100, 20, 1, 20);
	}
};