// EyeContactThesis.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "BodyTracking.cpp"
#include "FaceTracking.cpp"
#include "EyeTracking.cpp"
#include "PupilTracking.cpp"
#include "SVM.cpp"
#include "Utility.cpp"

// Control the techniques used with the following
// 1 is cascade classifier
#define BODY_ALG 0
// 1 is cascade classifier
#define FACE_ALG 1
// 1 is cascade classifier
#define EYES_ALG 1
// 1 is morphological opening
#define PUPI_ALG 1
#define EYEC_ALG 0

// Extract eyes from 
#define IS_TRAINING 0
// 0 is SVM
// 1 is Bayes Classifier
#define MODEL 1

#define RES_WIDTH 320
#define RES_HEIGHT 240

const std::string path_to_classifiers = "C:\\Program Files\\OpenCV\\2.0\\opencv\\sources\\data\\haarcascades\\";

void run_algorithm() {
	BodyDetector* body_detector = NULL;
	FaceDetector* face_detector = NULL;
	EyeDetector* eye_detector = NULL;
	PupilDetector* pupil_detector = NULL;

	// Setup necessary files/modules
	switch (BODY_ALG) {
	case 1:
		body_detector = new BodyCascadeClassifier(path_to_classifiers + "haarcascade_upperbody.xml");
		break;
	default:
		body_detector = new BodyDetector();
		break;
	}
	switch (FACE_ALG) {
	case 1:
		face_detector = new FaceCascadeClassifier(path_to_classifiers + "haarcascade_frontalface_alt2.xml");
		break;
	default:
		face_detector = new FaceDetector();
		break;
	}
	switch (EYES_ALG) {
	case 1:
		eye_detector = new EyeCascadeClassifier(path_to_classifiers + "haarcascade_eye.xml");
		break;
	default:
		eye_detector = new EyeDetector();
		break;
	}
	switch (PUPI_ALG) {
	case 1:
		pupil_detector = new PupilHoughDetector();
		break;
	default:
		pupil_detector = new PupilDetector();
		break;
	}

	// Open webcam
	cv::VideoCapture cap(0);

	// Check if everything is ok
	if (!cap.isOpened())
		return;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, RES_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, RES_HEIGHT);
	cap.set(CV_CAP_PROP_FPS, 30);

	cv::Mat frame, eye_tpl;
	cv::Rect eye_bb;
	std::vector<cv::Rect> bodies, faces, eyes;
	std::vector<cv::Vec3f> circles;
	std::cout << circles.size();

	CvSVM svm;
	svm.load("eye_looking_svm"); // saving
	CvNormalBayesClassifier bayes_classifier;
	bayes_classifier.load("bayes_classifier");

	int num_frames = 0;
	std::clock_t start = std::clock();

	while (cv::waitKey(10) != 'q')
	{
		if (IS_TRAINING) {
			if (!load_image_color(frame)) {
				break;// Read the file
			}
		} else {
			cap >> frame;
		}
		if (frame.empty()) {
			std::cout << "There was an error with loading the image.";
			break;
		}

		// Flip the frame horizontally
		cv::flip(frame, frame, 1);

		// Detect bodies
		body_detector->detect_bodies(frame, bodies);
		for (int i = 0; i < bodies.size(); ++i) {
			cv::rectangle(frame, bodies[i], CV_RGB(0, 255, 0));
		}
		// Detect faces
		face_detector->detect_faces(frame, faces);

		for (int i = 0; i < faces.size(); i++)
		{
			// For each face, detect the eyes
			// Assume that the eyes are located in the top portion
			cv::Rect top_portion = cv::Rect(faces[i].x, faces[i].y + faces[i].height / 10, faces[i].width, faces[i].height / 2);
			cv::Mat face = frame(top_portion);
			cv::imshow("video2", face);
			int min_size = face.rows / 5;
			eye_detector->detect_eyes(face, eyes);
			cv::rectangle(frame, faces[i], CV_RGB(0, 255, 0));
			int num_eye = 1;
			for (cv::Rect eye : eyes) {
				cv::Mat eye_area;

				// Take the middle one third of the eye
				cv::Rect eye_rect = eye + cv::Point(0, eye.height / 3);
				eye_rect.height /= 3;

				cv::Mat just_eye = face(eye_rect);
				cv::cvtColor(just_eye, just_eye, CV_BGR2GRAY);
				resize(just_eye, just_eye, cv::Size(30, 10));
				if (IS_TRAINING == -1) {
					std::string filename = "C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\"
											+ std::to_string(cur_image - 1) + "_" + std::to_string(num_eye++) +".jpg";
					cv::imwrite(filename, just_eye);
				}
				cv::Mat temp = flatten_image_1D(just_eye);
				cv::Scalar color(0, 255, 0);

				if (IS_TRAINING == 0) {
					//float response = svm.predict(temp);
					float response2 = bayes_classifier.predict(temp);
					//std::cout << response;
					std::cout << response2;
					if (response2 == 1) {
						color = cv::Scalar(0, 0, 255);
					}
				}
				cv::rectangle(frame, eye + cv::Point(faces[i].x, faces[i].y + faces[i].height / 10), color);
				//pupil_detector->detect_pupil(just_eye, circles);
				//cv::imshow("video2", eye_area);
				for (size_t j = 0; j < circles.size(); j++) {
					cv::Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
					int radius = cvRound(circles[j][2]);
					// circle center
					cv::Point new_center = center + cv::Point(eye_rect.x, eye_rect.y) + cv::Point(faces[i].x, faces[i].y + faces[i].height / 10);

					circle(frame, new_center, 3, color, -1, 8, 0);
					// circle outline
					circle(frame, new_center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
				}
			}
		}
		num_frames++;
		double fps = (double)CLOCKS_PER_SEC / (std::clock() - start);
		start = std::clock();
		putText(frame, ftos(fps, 3), cv::Point(10, 25), 0, 1, cv::Scalar(255, 255, 255));
		cv::imshow("video", frame);
	}
	delete body_detector;
	delete face_detector;
	delete eye_detector;
	delete pupil_detector;
}


int main(int argc, char** argv)
{
	if (IS_TRAINING == 1) {
		train_bayes_classifier();
		train_svm();
	} else {
		run_algorithm();
	}
	return 0;
}