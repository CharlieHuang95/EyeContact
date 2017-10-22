// Create a Support Vector Machine

// EyeContactThesis.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define MAX_IMAGES 60
static int cur_image = 1;

int labels[MAX_IMAGES] = { 
1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

bool load_image_color(cv::Mat &frame) {
	if (cur_image <= MAX_IMAGES) {
		std::string filename = "C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Faces\\" + std::to_string(cur_image) + ".jpg";
		frame = cv::imread(filename, CV_LOAD_IMAGE_COLOR);   // Read the file
		cur_image++;
		return true;
	} else {
		return false;
	}
}

cv::Mat flatten_image_1D(cv::Mat& frame) {
	cv::Mat training_examples(1, 30 * 10, CV_32FC1);
	int ii = 0;
	for (int a = 0; a < 10; ++a) {
		for (int b = 0; b < 30; ++b) {
			training_examples.at<float>(0, ii++) = frame.at<uchar>(a, b);
		}
	}
	return training_examples;
}

void train_bayes_classifier() {
	//cv::Mat training_examples(MAX_IMAGES, 30 * 10, CV_32FC1);
	float images[MAX_IMAGES][300];
	// Iterate through every image
	for (int i = 1; i <= MAX_IMAGES; ++i) {
		std::string filename = "C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\" + std::to_string(i) + ".jpg";
		cv::Mat frame = cv::imread(filename, 0);   // Read the file
		std::cout << "Width: " << frame.rows << " height: " << frame.cols;
		// Compress image to a vector
		int ii = 0;
		for (int a = 0; a < 10; ++a) {
			for (int b = 0; b < 30; ++b) {
				images[i - 1][ii++] = frame.at<uchar>(a, b);
			}
		}
		//imshow("training exp", training_examples);
		cv::waitKey(10000);
	}

	cv::Mat trainingData(MAX_IMAGES, 300, CV_32FC1, images);
	cv::Mat labelsMat(MAX_IMAGES, 1, CV_32S, labels);

	CvNormalBayesClassifier bayes_classifier = CvNormalBayesClassifier(trainingData, labelsMat);
	bayes_classifier.save("bayes_classifier");
}

void train_svm() {
	//cv::Mat training_examples(MAX_IMAGES, 30 * 10, CV_32FC1);
	float images[MAX_IMAGES][300];
	// Iterate through every image
	for (int i = 1; i <= MAX_IMAGES; ++i) {
		std::string filename = "C:\\Users\\Charlie\\source\\repos\\EyeContact_Thesis\\EyeContact_Thesis\\TrainingSet\\Eyes\\" + std::to_string(i) + ".jpg";
		cv::Mat frame = cv::imread(filename, 0);   // Read the file
		std::cout << "Width: " << frame.rows << " height: " << frame.cols;
		// Compress image to a vector
		int ii = 0;
		for (int a = 0; a < 10; ++a) {
			for (int b = 0; b < 30; ++b) {
				images[i - 1][ii++] = frame.at<uchar>(a, b);
			}
		}
		//cv::waitKey(10000);
	}
	
	cv::Mat trainingData(MAX_IMAGES, 300, CV_32FC1, images);
	cv::Mat labelsMat(MAX_IMAGES, 1, CV_32S, labels);

	CvSVM svm;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.gamma = 3;
	params.degree = 3;
	svm.train(trainingData, labelsMat, cv::Mat(), cv::Mat(), params);
	svm.save("eye_looking_svm");
}