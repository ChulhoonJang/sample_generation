#include <direct.h>
#include <iostream>
#include "Shlwapi.h"
#include <opencv\highgui.h>

using namespace std;
using namespace cv;

class annotated_class{
public:
	annotated_class(string p, string attr, bool mk_save_dir=false);
	~annotated_class();
	
	string get_annotation_path();
	string get_sample_path();
	string get_file_path(const int frame);
	string get_attribute();
	void generate_samples(const Mat src, Size sz, double occupancy);
	void generate_samples(const Mat src, Size sz, double occupancy, Rect roi);
	void generate_random_samples(const Mat src, Size sz, double occupancy, int num);

public:
	Mat img_mask;

private:
	string path;
	string attribute;
	string path_annotation;
	string path_sample;
	vector<int> params;
	int n_samples;
};