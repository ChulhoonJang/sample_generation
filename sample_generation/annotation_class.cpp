#include <time.h>
#include "annotation_class.h"
#include <opencv2\imgproc\imgproc.hpp>

annotated_class::annotated_class(string p, string attr, int _n_class, Scalar color, bool mk_save_dir){
	path = p;
	attribute = attr;
	path_annotation = path + "/annotations/" + attribute;
	path_sample = path + "/samples/" + attribute;

	//if (PathFileExists(path_annotation.c_str()) || mk_save_dir == true) _mkdir(path_sample.c_str());

	n_samples = 1;

	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);

	n_class = _n_class;

	gt_color = color;
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

void annotated_class::generate_gt_images(){
	Mat bgr[3];
	cvtColor(img_mask, img_gt, CV_GRAY2BGR);
	split(img_gt, bgr);
	bgr[0] = bgr[0] * gt_color[0];
	bgr[1] = bgr[1] * gt_color[1];
	bgr[2] = bgr[2] * gt_color[2];
	merge(bgr, 3, img_gt);
}

void annotated_class::generate_samples(const Mat src, Size sz, double occupancy){
	unsigned char* p_mask = img_mask.data;
	int h = img_mask.rows - sz.height + 1;
	int w = img_mask.cols - sz.width + 1;
		
	samples.clear();
	for (int y = 0; y < h; y++){
		for (int x = 0; x < w; x++){
			Rect rect(x, y, sz.width, sz.height);
			Mat img_patch = img_mask(rect);

			Scalar avg = mean(img_patch / 255);
			if (avg[0] > occupancy){				
				samples.push_back(rect);
				n_samples++;
			}
		}
	}	
}

void annotated_class::generate_samples(const Mat src, Size sz, double occupancy, Rect roi){
	unsigned char* p_mask = img_mask.data;
	int h = img_mask.rows - sz.height + 1;
	int w = img_mask.cols - sz.width + 1;
	
	samples.clear();
	for (int y = 0; y < h; y++){
		for (int x = 0; x < w; x++){
			Rect rect(x, y, sz.width, sz.height);
			Mat img_patch = img_mask(rect);
			Mat img_roi = img_patch(roi);
			Scalar avg = mean(img_roi / 255);
			if (avg[0] > occupancy){				
				samples.push_back(rect);
				n_samples++;
			}
		}
	}	
}

void annotated_class::generate_random_samples(const Mat src, Size sz, double occupancy, int num){
	srand((unsigned int)time(NULL));
	int y_bound = src.rows - sz.height;
	int x_bound = src.cols - sz.width;
	
	samples.clear();
	for (int i = 0; i < num;){
		int x = rand() % x_bound;
		int y = rand() % y_bound;
		Rect rect(x, y, sz.width, sz.height);

		Mat img_patch = img_mask(rect);			
		Scalar avg = mean(img_patch / 255);
		if (avg[0] >= occupancy){			
			samples.push_back(rect);
			n_samples++;
			i++;
		}
	}		
}

void annotated_class::open_csv_file(int frame){
	char temp[100];
	sprintf_s(temp, 100, "%s/%08d.csv", path_sample.c_str(), frame);
	csv_file.open(temp, ios::out | ios::ate | ios::app);
}

void annotated_class::close_csv_file(){
	csv_file.close();
}