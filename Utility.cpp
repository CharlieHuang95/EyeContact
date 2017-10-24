#pragma once

class ImageData {
public:
	ImageData() {}
	ImageData(std::string full_path, std::string filename, bool is_match) :
		full_path(full_path), filename(filename), is_match(is_match) {}
	std::string full_path;
	std::string filename;
	bool is_match;
};

class ImagesDataWrapper {
private:
	std::vector<ImageData> images;
	int current_image = 0; // Increment this pointer
public:
	ImagesDataWrapper() {}
	void add_images(ImageData image_data) {
		images.push_back(image_data);
	}
	void add_images(std::vector<ImageData> filenames) {
		for (ImageData image_data : filenames) {
			images.push_back(image_data);
		}
	}
	bool has_image() {
		return current_image < images.size();
	}
	ImageData get_image() {
		if (has_image())
			return images.at(current_image++);
	}
};

std::vector<float> flatten_image_vec(cv::Mat& frame) {
	std::vector<float> flattened; // (1, frame.cols * frame.rows, CV_32FC1);
	int ii = 0;
	for (int a = 0; a < 10; ++a)
		for (int b = 0; b < 30; ++b)
			flattened.push_back(frame.at<uchar>(a, b));
	return flattened;
}

cv::Mat vector_to_mat(std::vector<std::vector<float>> vec) {
	assert(vec.size() > 0);
	cv::Mat mat;
	//mat.push_back(= cv::Mat(vec.size(), vec[0].size());
	return mat;
}

cv::Mat flatten_image_mat(cv::Mat& frame) {
	cv::Mat flattened(1, frame.cols * frame.rows, CV_32FC1);
	int ii = 0;
	for (int a = 0; a < frame.rows; ++a)
		for (int b = 0; b < frame.cols; ++b)
			flattened.at<float>(0, ii++) = frame.at<uchar>(a, b);
	return flattened;
}


std::string ftos(float f, int nd) {
	std::ostringstream ostr;
	int tens = std::stoi("1" + std::string(nd, '0'));
	ostr << round(f*tens) / tens;
	return ostr.str();
}