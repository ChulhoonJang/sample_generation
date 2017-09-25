#include <direct.h>
#include <iostream>
#include <fstream>
#include "Shlwapi.h"
#include <opencv\highgui.h>

using namespace std;
using namespace cv;

class annotated_class{
public:
	annotated_class(string p, string attr, int _n_class, Scalar color, bool mk_save_dir = false);
	~annotated_class();
	
	string get_annotation_path();
	string get_sample_path();
	string get_file_path(const int frame);
	string get_attribute();
	void generate_gt_images();

	void generate_samples(const Mat src, Size sz, double occupancy);
	void generate_samples(const Mat src, Size sz, double occupancy, Rect roi);
	void generate_random_samples(const Mat src, Size sz, double occupancy, int num);
	void open_csv_file(int frame);
	void close_csv_file();

public:
	Mat img_mask;
	Mat img_gt;
	vector<Rect> samples;
	int n_class;

private:
	string path;
	string attribute;
	string path_annotation;
	string path_sample;
	vector<int> params;
	int n_samples;	
	ofstream csv_file;	
	Scalar gt_color;
};