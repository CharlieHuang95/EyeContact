/* This program is used to train classification models. */

#include "stdafx.h"
#include <atlstr.h> 
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <sys/types.h>
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

#include "Utility.cpp"

/*
Create an object that prepares training data. Usage should be:

TrainingData training_data = TrainingData(num_features);
training_data.add_testcase(training_image, label);

*/

class TrainingData {
private:
	int num_features;
	cv::Mat training_set;
	cv::Mat labels;
public:
	TrainingData() {}
	void add_testcase(cv::Mat& training_image, int label) {
		training_set.push_back(flatten_image_mat(training_image));
		labels.push_back(label);
	}
	cv::Mat get_training_set() {
		cv::Mat converted;
		training_set.convertTo(converted, CV_32FC1);
		return converted;
	}
	cv::Mat get_labels() {
		cv::Mat converted;
		labels.convertTo(converted, CV_32S);
		return converted;
	}
};

/* 
Create a generic classification model object. Usage should be:
	
	GenericClassifier *classifier;
	if (DESIRED_CLASSIFIER == 1) { classifier = new DerivedClassifier(); }
	TrainingData training_data;
	classifier->train(training_data);
	classifier->save(filename);
	classifier->load(filename);
	int value = classifier->predict(input_data);
*/

class GenericClassifier {
public:
	virtual void train(TrainingData training_data) {}
	virtual void save(std::string filename) {}
	virtual void load(std::string filename) {}
	virtual int predict(cv::Mat input_data) { return 0; }
};

class BayesClassifier : public GenericClassifier {
private:
	CvNormalBayesClassifier classifier;
public:
	BayesClassifier() {}
	void train(TrainingData training_data) {
		// Train the model
		cv::Mat t = training_data.get_training_set();
		cv::Mat l = training_data.get_labels();
		std::cout << "About to train Bayes Classifier.\n"
			      << "Training set is: " << t.rows << "x" << t.cols << "\n"
			      << "Labels set is: " << l.rows << "x" << l.cols << "\n";
		classifier.train(t, l);
		std::cout << "Bayes Classifier has been trained.\n";
	}
	void save(std::string filename) {
		classifier.save(filename.c_str());
	}
	void load(std::string filename) {
		classifier.load(filename.c_str());
	}
	int predict(cv::Mat input_data) {
		return classifier.predict(input_data);
	}
};

std::vector<std::string> collect_files(std::string directory) {
	std::vector<std::string> files;
	HANDLE hFile = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA FindFileData;
	CString strTemp;
	CString cstr(directory.c_str());
	strTemp.Format(_T("%s\\%s"), cstr, _T("*.*"));
	hFile = FindFirstFile(strTemp, &FindFileData);
	CString strFilePath;

	if (INVALID_HANDLE_VALUE != hFile) {
		do {
			//Skip directories
			if (FILE_ATTRIBUTE_DIRECTORY & FindFileData.dwFileAttributes)
				continue;

			strFilePath.Format(_T("%s\\%s"), cstr, FindFileData.cFileName);
			CT2CA pszConvertedAnsiString(strFilePath);
			std::string s(pszConvertedAnsiString);
			files.push_back(s);
			strFilePath.Empty();
		} while (FindNextFile(hFile, &FindFileData));
		FindClose(hFile);
	} else {
		std::cout << "Could not find any files.\n";
	}
	return files;
}

#define MAX_IMAGES 60

int labels[MAX_IMAGES] = { 
1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

void train_bayes_classifier() {
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

	svm.save("eye_looking_svm");
}