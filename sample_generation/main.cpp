#include <stdlib.h>
#include <time.h>
#include <direct.h>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

vector<Point> polygon;
vector<vector<Point>> polygon_groups;

void import_annotation(FileStorage & f, string name){
	FileNode fn = f[name];
	FileNodeIterator it = fn.begin(), it_end = fn.end();

	polygon_groups.clear();
	for (int idx = 0; it != it_end; ++it, idx++){
		vector<Point> p;
		vector<int> x_val, y_val;
		(*it)["x"] >> x_val;
		(*it)["y"] >> y_val;
		for (int i = 0; i<(int)x_val.size(); i++){
			p.push_back(Point(x_val[i], y_val[i]));
		}
		polygon_groups.push_back(p);
	}
}

void generate_polygon_mask(Mat src, Mat & dst, vector<Point> poly){
	Mat img_bin = Mat::zeros(src.size(), CV_8UC1);

	vector<vector<Point>> pPoly;
	pPoly.push_back(poly);
	fillPoly(img_bin, pPoly, Scalar(255, 255, 255));
	dst = img_bin.clone();
}

void generate_polygon_mask(Mat src, Mat & dst, vector<vector<Point>> polygon_groups){
	Mat img_bin = Mat::zeros(src.size(), CV_8UC1);
	fillPoly(img_bin, polygon_groups, Scalar(255, 255, 255));
	dst = img_bin.clone();
}

int main(int argc, char *argv[], char *envp[])
{
	string root, attribute;
	string image_dir, annotation_dir, attribute_dir, saved_dir, class_0_dir, class_1_dir, class_2_dir, class_3_dir;
	int patch_x(24), patch_y(24), roi_x(5), roi_y(5);
	Rect rect_ego(74, 54, 156, 52);

	if (argc < 2) return 0;
	root = argv[1];
	attribute = argv[2];

	image_dir = root + "/images";
	annotation_dir = root + "/annotations";
	attribute_dir = annotation_dir + '/' + attribute;
	saved_dir = root + "/samples";
	class_0_dir = saved_dir + "/0"; // free space
	class_1_dir = saved_dir + "/1"; // marker
	class_2_dir = saved_dir + "/2"; // vehicle
	class_3_dir = saved_dir + "/3"; // curb

	_mkdir(saved_dir.c_str());
	_mkdir(class_0_dir.c_str());
	_mkdir(class_1_dir.c_str());
	_mkdir(class_2_dir.c_str());
	_mkdir(class_3_dir.c_str());

	vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);	
	
	FileStorage f_info;
	string frame_yml = annotation_dir + "/frame.yml";
	if (!f_info.open(frame_yml, FileStorage::READ)) return 0;
	FileNode f_info_n = f_info["frame"];
	FileNodeIterator f_it = f_info_n.begin();
	
	unsigned int n_c_0(1), n_c_1(1), r(0);
	for (; f_it != f_info_n.end();f_it++){
		int frame = (*f_it);

		// load image
		char image_path[100];
		sprintf_s(image_path, 100, "%s/%08d.jpg", image_dir.c_str(), frame);

		char file_name[100];
		sprintf_s(file_name, 100, "%s/%08d.yml", attribute_dir.c_str(), frame);
		FileStorage fr;
		
		Mat img, img_debug, img_mask;
		img = imread(image_path);
		if (img.data == NULL) continue;
		if (!fr.open(file_name, FileStorage::READ)) continue;
				
		img_debug = img.clone();
		rectangle(img_debug, rect_ego, CV_RGB(255, 0, 0));
		import_annotation(fr, attribute);				
		generate_polygon_mask(img,img_mask, polygon_groups);

		unsigned char* p_mask = img_mask.data;
		int h = img_mask.rows - patch_y + 1;
		int w = img_mask.cols - patch_x + 1;

		for (int y = 0; y < h; y++){
			for (int x = 0; x < w; x++){
				Rect rect(x, y, patch_x, patch_y);
				Mat img_patch = img_mask(rect);

				Rect roi((patch_x - roi_x + 1) / 2, (patch_y - roi_y + 1) / 2, roi_x, roi_y);
				Mat img_roi = img_patch(roi);
				Scalar avg = mean(img_roi/255);
				if (avg[0] > 0.75){
					rectangle(img_debug, rect, CV_RGB(0, 255, 0));
					
					Mat img_sample = img(rect);
					char saved_path[100];
					sprintf_s(saved_path, 100, "%s/%08d.jpg", class_1_dir.c_str(), n_c_1);
					imwrite(saved_path, img_sample, params);
					n_c_1++;
				}
			}
		}

		srand(time(NULL));
		int y_bound = img.rows - patch_y;
		int x_bound = img.cols - patch_x;
		for (int i = 0; i < 3000 ;){
			int x = rand() % x_bound;
			int y = rand() % y_bound;
			Rect rect(x, y, patch_x, patch_y);
			Rect intersected = rect & rect_ego;
			if (intersected.width > (patch_x/2) && intersected.height > (patch_y/2)) continue;

			Mat img_patch = img_mask(rect);			
			Scalar avg = mean(img_patch / 255);
			if (avg[0] == 0.0){
				//rectangle(img_debug, rect, CV_RGB(0, 255, 0));
				Mat img_sample = img(rect);
				char saved_path[100];
				sprintf_s(saved_path, 100, "%s/%08d.jpg", class_0_dir.c_str(), n_c_0);
				imwrite(saved_path, img_sample, params);
				n_c_0++;
				i++;
			}
		}		
	}
}