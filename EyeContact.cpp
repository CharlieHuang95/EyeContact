// EyeContactThesis.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <regex>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "BodyTracking.cpp"
#include "FaceTracking.cpp"
#include "EyeTracking.cpp"
#include "PupilTracking.cpp"
#include "ClassificationModeler.cpp"
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
// #define MODEL 1

#define RES_WIDTH 320
#define RES_HEIGHT 240

const std::string path_to_classifiers = "C:\\Program Files\\OpenCV\\2.0\\opencv\\sources\\data\\haarcascades\\";

void run_algorithm(ImagesDataWrapper images) {
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

	// Saves the camera input
	cv::Mat frame;

	// Saves bounding boxes of the captured objects
	std::vector<cv::Rect> bodies, faces, eyes;

	// Saves a representation of the eye pupil
	std::vector<cv::Vec3f> circles;

	CvSVM svm;
	svm.load("eye_looking_svm");
	CvNormalBayesClassifier bayes_classifier;
	bayes_classifier.load("bayes_classifier");
	GenericClassifier* classifier;
	classifier = new BayesClassifier();
	classifier->load("bayes_classifier_v1");

	// Record the number of frames that have been recorded
	int num_frames = 0;
	// Record the start time
	std::clock_t start = std::clock();

	while (cv::waitKey(10) != 'q') {
		ImageData my_image;
		if (IS_TRAINING == -1) {
			if (!images.has_image()) {
				break;// Read the file
			} else {
				my_image = images.get_image();
				frame = cv::imread(my_image.full_path);
			}
		} else {
			cap >> frame;
		}
		if (frame.empty()) {
			std::cout << "There was an error with loading the image.";
			cv::waitKey(1000);
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

		for (int i = 0; i < faces.size(); i++) {

			// Detect eyes. Assume that the eyes are located in the top portion
			cv::Rect top_portion = cv::Rect(faces[i].x, faces[i].y + faces[i].height / 10, faces[i].width, faces[i].height / 2);
			cv::Mat face = frame(top_portion);
			//cv::imshow("video2", face);

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
					std::string directory = my_image.is_match ? "Match" : "NonMatch";
					std::string filename = "C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\"
											+ directory + "\\" + my_image.filename + "_" + std::to_string(num_eye++) +".jpg";
					cv::imwrite(filename, just_eye);
				}
				cv::Mat temp = flatten_image_mat(just_eye);
				cv::Scalar color(0, 255, 0);

				if (IS_TRAINING == 0) {
					//float response = svm.predict(temp);
					float response2 = classifier->predict(temp);
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
		double fps = (double) CLOCKS_PER_SEC / (std::clock() - start);
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
	ImagesDataWrapper images;
	if (IS_TRAINING == -1) {
		std::vector<std::string> match =
			collect_files("C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Faces\\Match");
		std::vector<std::string> non_match =
			collect_files("C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Faces\\NonMatch");
		std::regex file_regex("([a-zA-Z0-9_]*)\\.[a-zA-Z0-9_]*");
		for (std::string s : match) {
			std::smatch m;
			std::regex_search(s, m, file_regex);
			if (m.size() > 0) {
				std::ssub_match base_sub_match = m[0];
				std::string base = base_sub_match.str();
				images.add_images(ImageData(s, base, true));
				//std::cout << base << '\n';
			}
		}
		for (std::string s : non_match) {
			std::smatch m;
			std::regex_search(s, m, file_regex);
			if (m.size() > 0) {
				std::ssub_match base_sub_match = m[0];
				std::string base = base_sub_match.str();
				images.add_images(ImageData(s, base, false));
				//std::cout << base << '\n';
			}
		}

		// Make copies of the relevant parts of the image
		run_algorithm(images);
	} else if (IS_TRAINING == 1) {
		// Prepare training data
		TrainingData training_data;
		std::vector<std::string> match =
			collect_files("C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\Match");
		std::vector<std::string> non_match =
			collect_files("C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\NonMatch");
		for (std::string s : match) {
			cv::Mat training_image = cv::imread(s);
			training_data.add_testcase(training_image, true);
		}
		for (std::string s : non_match) {
			cv::Mat training_image = cv::imread(s);
			training_data.add_testcase(training_image, false);
		}

		GenericClassifier* classifier;
		classifier = new BayesClassifier();
		classifier->train(training_data);
		classifier->save("bayes_classifier_v1");
		//train_svm();
	} else {
		run_algorithm(images);
	}
	return 0;
}