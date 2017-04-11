#include <time.h>
#include "annotation_class.h"

annotated_class::annotated_class(string p, string attr, bool mk_save_dir ){
	path = p;
	attribute = attr;
	path_annotation = path + "/annotations/" + attribute;
	path_sample = path + "/samples/" + attribute;

	if (PathFileExists(path_annotation.c_str()) || mk_save_dir == true) _mkdir(path_sample.c_str());

	n_samples = 1;

	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);
}

annotated_class::~annotated_class(){
}

string annotated_class::get_annotation_path(){ 
	return path_annotation; 
}

string annotated_class::get_sample_path() { 
	return path_sample; 
}

string annotated_class::get_file_path(const int frame){
	char temp[100];
	sprintf_s(temp, 100, "%s/%08d.yml", path_annotation.c_str(), frame);

	return temp;
}

string annotated_class::get_attribute() { 
	return attribute; 
}

void annotated_class::generate_samples(const Mat src, Size sz, double occupancy){
	unsigned char* p_mask = img_mask.data;
	int h = img_mask.rows - sz.height + 1;
	int w = img_mask.cols - sz.width + 1;

	for (int y = 0; y < h; y++){
		for (int x = 0; x < w; x++){
			Rect rect(x, y, sz.width, sz.height);
			Mat img_patch = img_mask(rect);

			Scalar avg = mean(img_patch / 255);
			if (avg[0] > occupancy){
				Mat img_sample = src(rect);
				char saved_path[100];
				sprintf_s(saved_path, 100, "%s/%08d.jpg", path_sample.c_str(), n_samples);
				imwrite(saved_path, img_sample, params);
				n_samples++;
			}
		}
	}
}

void annotated_class::generate_samples(const Mat src, Size sz, double occupancy, Rect roi){
	unsigned char* p_mask = img_mask.data;
	int h = img_mask.rows - sz.height + 1;
	int w = img_mask.cols - sz.width + 1;

	for (int y = 0; y < h; y++){
		for (int x = 0; x < w; x++){
			Rect rect(x, y, sz.width, sz.height);
			Mat img_patch = img_mask(rect);
			Mat img_roi = img_patch(roi);
			Scalar avg = mean(img_roi / 255);
			if (avg[0] > occupancy){
				Mat img_sample = src(rect);
				char saved_path[100];
				sprintf_s(saved_path, 100, "%s/%08d.jpg", path_sample.c_str(), n_samples);
				imwrite(saved_path, img_sample, params);
				n_samples++;
			}
		}
	}
}

void annotated_class::generate_random_samples(const Mat src, Size sz, double occupancy, int num){
	srand((unsigned int)time(NULL));
	int y_bound = src.rows - sz.height;
	int x_bound = src.cols - sz.width;
	for (int i = 0; i < num;){
		int x = rand() % x_bound;
		int y = rand() % y_bound;
		Rect rect(x, y, sz.width, sz.height);

		Mat img_patch = img_mask(rect);			
		Scalar avg = mean(img_patch / 255);
		if (avg[0] >= occupancy){
			Mat img_sample = src(rect);
			char saved_path[100];
			sprintf_s(saved_path, 100, "%s/%08d.jpg", path_sample.c_str(), n_samples);
			imwrite(saved_path, img_sample, params);
			n_samples++;
			i++;
		}
	}		
}